#!/usr/bin/env bash
# fleet_dispatch.sh — 跨机协同调度器
#
# 工作原理：
#   1. 扫描当前 kanban 看板里 status=ready 的任务
#   2. 按 assignee 映射到对应远程机器（通过 Tailscale SSH）
#   3. 把任务 body 通过 SSH 发给远程机器的 hermes 执行（hermes chat -q）
#   4. 远程 hermes 的输出回传，本机代为 kanban comment + complete
#
# 远程机器不需要访问本机 kanban.db —— 本机脚本当桥梁。
# 远程 hermes 只负责执行任务指令，不读写 kanban 状态。
#
# 用法：
#   ./fleet_dispatch.sh            # 扫描并派发所有 ready 任务（一次）
#   ./fleet_dispatch.sh --loop 30  # 每 30 秒扫一次，持续运行
#   ./fleet_dispatch.sh --task t_xxx  # 只派发指定任务
#
# assignee 到远程机器的映射在下面的 FLEET_MAP 里配置。

set -euo pipefail

# ========== 配置：assignee -> SSH 目标 ==========
# 格式："assignee_name|ssh_user|tailscale_ip|remote_hermes_cmd"
# remote_hermes_cmd 是远程机器上调用 hermes 的命令（确保 PATH 含 ~/.local/bin）
# ===== 按你的环境修改下面的映射 =====
FLEET_MAP=(
  "worker-<name1>|<ssh_user>|<tailscale_ip>|<remote_hermes_path>"
  "worker-<name2>|<ssh_user>|<tailscale_ip>|<remote_hermes_path>"
  # 示例：
  # "worker-beefy|alice|100.64.0.10|/home/alice/.local/bin/hermes"
  # "worker-mini|bob|100.64.0.11|/home/bob/.local/bin/hermes"
)
# ================================================

# 本机 hermes 命令
LOCAL_HERMES="<path_to_hermes_on_hub>"

# ========== 工具函数 ==========

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# 按 assignee 名查 FLEET_MAP，输出 "user|ip|hermes_cmd"
lookup_fleet() {
  local assignee="$1"
  for entry in "${FLEET_MAP[@]}"; do
    local name="${entry%%|*}"
    if [[ "$name" == "$assignee" ]]; then
      # 去掉首字段，剩下 "user|ip|hermes_cmd"
      echo "${entry#*|}"
      return 0
    fi
  done
  return 1
}

# 获取任务的 body（描述）。用 hermes kanban show 提取 Body 部分
get_task_body() {
  local task_id="$1"
  "$LOCAL_HERMES" kanban show "$task_id" 2>/dev/null \
    | awk '/^Body:/{f=1; sub(/^Body:[[:space:]]*/,""); print; next} f&&/^---|^Events|^Comments|^Runs|^Latest/{f=0} f{print}'
}

# 获取任务的 assignee
get_task_assignee() {
  local task_id="$1"
  "$LOCAL_HERMES" kanban show "$task_id" 2>/dev/null \
    | awk '/^  assignee:/{gsub(/^[[:space:]]*assignee:[[:space:]]*/,""); print; exit}'
}

# 列出所有 ready 任务 ID
list_ready_tasks() {
  "$LOCAL_HERMES" kanban list --status ready 2>/dev/null \
    | grep -oE 't_[a-f0-9]+' || true
}

# 派发单个任务到远程机器
dispatch_task() {
  local task_id="$1"
  local assignee body fleet user ip rhermes

  assignee=$(get_task_assignee "$task_id")
  [[ -z "$assignee" || "$assignee" == "default" ]] && {
    log "任务 $task_id assignee=$assignee，跳过（default 由本机 gateway 处理）"
    return 0
  }

  fleet=$(lookup_fleet "$assignee") || {
    log "任务 $task_id assignee=$assignee 无对应远程机器，跳过"
    return 0
  }
  IFS='|' read -r user ip rhermes <<<"$fleet"

  body=$(get_task_body "$task_id")
  [[ -z "$body" ]] && body="执行任务 $task_id"

  log "派发 $task_id -> $assignee ($user@$ip)"

  # 把 body 写到临时文件，scp 到远程，远程 hermes 从文件读
  # 这样完全避免 shell 引号嵌套和 SSH SetEnv 限制
  local tmpbody="/tmp/fleet_task_${task_id}.txt"
  printf '%s' "$body" > "$tmpbody"
  local remote_body="/tmp/fleet_task_${task_id}.txt"

  scp -o BatchMode=yes -o ConnectTimeout=15 "$tmpbody" "$user@$ip:$remote_body" 2>&1 | tail -1 || true

  local result
  if result=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$user@$ip" \
        "export PATH=\"\$HOME/.local/bin:\$PATH\";
         # 显式清掉代理，确保 hermes 对 Tailscale 内网直连（各机器都有 Clash 代理会劫持 100.x）
         unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy;
         export NO_PROXY='localhost,127.0.0.1,::1,100.64.0.0/10,<hub_tailscale_ip>,<worker_ips...>';
         export no_proxy=\"\$NO_PROXY\";
         BODY=\$(cat $remote_body);
         timeout 600 \"$rhermes\" chat --yolo -q \"\$BODY\" 2>&1;
         rm -f $remote_body" 2>&1); then
    log "$task_id 远程执行完成"
    # 把结果写回 kanban（评论 + 完成）
    "$LOCAL_HERMES" kanban comment "$task_id" "$result" >/dev/null 2>&1
    "$LOCAL_HERMES" kanban complete "$task_id" >/dev/null 2>&1
    log "$task_id 已完成并写回 kanban"
  else
    log "$task_id 远程执行失败: $result"
    # 失败时只评论错误，不标 done —— 保持 retryable 或 block 供操作者处理
    "$LOCAL_HERMES" kanban comment "$task_id" "远程执行失败: $result" >/dev/null 2>&1
    "$LOCAL_HERMES" kanban block "$task_id" "远程执行失败，待排查" >/dev/null 2>&1
    log "$task_id 已标记为 blocked（远程执行失败）"
  fi
  rm -f "$tmpbody"
}

# ========== 主逻辑 ==========

main() {
  local loop=0 interval=30 single_task=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --loop) loop=1; interval="${2:-30}"; shift 2;;
      --task) single_task="$2"; shift 2;;
      *) echo "用法: $0 [--loop N] [--task t_xxx]"; exit 1;;
    esac
  done

  while true; do
    if [[ -n "$single_task" ]]; then
      dispatch_task "$single_task"
      break
    fi

    local tasks
    tasks=$(list_ready_tasks)
    if [[ -z "$tasks" ]]; then
      log "无 ready 任务"
    else
      while IFS= read -r tid; do
        [[ -n "$tid" ]] && dispatch_task "$tid"
      done <<<"$tasks"
    fi

    [[ $loop -eq 0 ]] && break
    log "等待 ${interval}s 后下一轮..."
    sleep "$interval"
  done
}

main "$@"
