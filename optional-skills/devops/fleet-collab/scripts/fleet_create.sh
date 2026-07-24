#!/usr/bin/env bash
# fleet_create.sh — 在任意机器上创建协同任务到 hub 的 fleet-collab 看板
#
# 原理：把 hermes kanban 的参数写到临时文件，scp 到 hub，hub 读取后执行。
# 这样完全避免 SSH 引号嵌套和参数拆分问题。
#
# 用法：
#   fleet_create.sh "任务标题" [--assignee worker-<name>] [--body "描述"]
#   fleet_create.sh --list              # 列出看板任务
#   fleet_create.sh --show t_xxxxx      # 查看任务详情
#
# 部署到每台机器。请按需修改下面的 HUB_* 和 IS_HUB 判断。

set -euo pipefail

# ===== 按你的环境修改以下三行 =====
HUB_USER="<hub_ssh_user>"
HUB_IP="<hub_tailscale_ip>"
HUB_HERMES="<path_to_hermes_on_hub>"
# ================================

# 判断当前是否在 hub 上（改成你的 hub 主机名）
IS_HUB=0
hostname | grep -qi "<hub_hostname>" && IS_HUB=1

# 核心函数：在本机执行 hermes kanban <subcmd> <args...>
# 参数通过临时文件 + scp 传输，避免 shell 引号问题
run_on_hub() {
  local subcmd="$1"; shift
  local tmpargs="/tmp/fleet_args_$$"
  # 每行一个参数写入临时文件
  printf '%s\n' "$@" > "$tmpargs"

  if [[ $IS_HUB -eq 1 ]]; then
    # 本机：用 while-read 循环读参数（兼容 macOS bash 3.2）
    local ARGS=()
    while IFS= read -r line; do ARGS+=("$line"); done < "$tmpargs"
    rm -f "$tmpargs"
    "$HUB_HERMES" kanban "$subcmd" "${ARGS[@]}"
  else
    # 远程：scp 参数文件到 hub，hub 读取后执行
    local remote_tmp="/tmp/fleet_args_remote_$$"
    scp -o BatchMode=yes -o ConnectTimeout=15 "$tmpargs" \
        "$HUB_USER@$HUB_IP:$remote_tmp" >/dev/null 2>&1
    rm -f "$tmpargs"
    # 远程用 while-read 兼容 macOS bash 3.2
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$HUB_USER@$HUB_IP" \
      "A=(); while IFS= read -r L; do A+=(\"\$L\"); done < $remote_tmp; rm -f $remote_tmp; $HUB_HERMES kanban $subcmd \"\${A[@]}\""
  fi
}

case "${1:-}" in
  --list|ls)
    if [[ $IS_HUB -eq 1 ]]; then
      "$HUB_HERMES" kanban list
    else
      ssh -o BatchMode=yes -o ConnectTimeout=15 "$HUB_USER@$HUB_IP" "$HUB_HERMES kanban list"
    fi
    ;;
  --show)
    shift
    if [[ $IS_HUB -eq 1 ]]; then
      "$HUB_HERMES" kanban show "$@"
    else
      ssh -o BatchMode=yes -o ConnectTimeout=15 "$HUB_USER@$HUB_IP" "$HUB_HERMES kanban show $*"
    fi
    ;;
  "")
    echo "用法: $0 \"任务标题\" [--assignee <profile>] [--body \"描述\"]"
    echo "      $0 --list | --show t_xxx"
    exit 1
    ;;
  *)
    run_on_hub create "$@"
    ;;
esac
