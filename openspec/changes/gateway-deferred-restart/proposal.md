type: spec-proposal
project: hermes-agent (upstream PR)
task: gateway-deferred-restart
status: reviewed (ChatGPT x3 + Codex x1) → final

# Proposal: gateway deferred self-restart

## 背景（Problem）

Hermes agent 无法自主重启 gateway。terminal 工具守卫按文本扫描拦截一切
"从 gateway 进程内杀宿主"的命令（hermes gateway restart、launchctl kickstart、
osascript 转发 Terminal 等），报错：
"Blocked: command or referenced script cannot restart, stop, or uninstall the
gateway from inside the gateway process... Run `hermes gateway restart` from a
separate shell outside the running gateway."

用户聊天授权绕不过该守卫（G1/G2 拦截语义是安全边界，本变更不允许放松）。
当前唯一通路是 computer_use 打桌面 Terminal.app（进程树外）执行延迟重启——
脆弱、依赖 UI 自动化、跨平台不成立。

痛点：config.yaml 仅启动时读取（实测确认），每次换 LLM API key / 改配置都要
用户手动跑命令或靠 UI 自动化绕行。

## 变更（Solution）

新增 agent 专用的、安全的、延迟生效的自重启通道：

`hermes gateway defer-restart`（无参数）

语义：请求当前 gateway 在当前对话 turn 完成后执行 drain-first 优雅重启，
由 supervisor（launchd/systemd/s6）拉起。CLI 在请求受理后立即返回。

### 两层架构（Codex 建议采纳）

1. **defer-restart 请求协议**：目标绑定、request_id、授权/provenance、冷却、
   审计、受理确认（本变重新增）
2. **gateway restart executor**：接收已验证请求后复用现有 request_restart()、
   drain、退出码 75 链路（零改动或最小改动）

### 请求归因协议（ChatGPT 第三轮定稿：请求文件 + SIGUSR1 唤醒）

SIGUSR1 不携带请求上下文，纯信号无法归因。协议：

1. CLI 先做目标解析 + supervisor 能力预检 + 冷却预算检查
2. CLI 原子写 `<hermes_home>/restart_requests/pending-<request_id>.json`
   （schema_version / request_id / service_identity / target_generation
   （pid+start_time）/ requested_at / actor_ref（自报线索，不作授权）），
   0700 目录 + 0600 文件 + tmp+rename+目录 fsync，拒符号链接/校验 owner
3. SIGUSR1 仅作唤醒（保留既有语义兼容，见不变量 I3）
4. gateway 侧：读请求文件 → 校验 generation/service identity/schema/授权 →
   原子认领 → 走现有 request_restart() → 持久记录 ACCEPTED(request_id)
5. CLI 轮询确认自身 request_id 被正确代际接受 → exit 0；证不出 → exit 6

**受理线性化点**：gateway 完成身份验证 + 原子认领 + 重启预算占用、确认可排程
重启的时刻。受理结果必须在该 generation 最终退出前持久化（active-work 为零的
竞态下 CLI 仍可读到受理记录——按 request_id 保留持久结果，不在新进程启动时清除）。

### 投递边界

active-work 归零 = 最终回复已提交平台发送 API（源码证据：turn 内发送全部
await；收尾先等 stream 最终 edit 完成才释放计数）。保证限于"平台 API 接受"，
不承诺"用户可见到达"。超时路径（restart_after_turn_timeout）不承诺无损。

### 消息恢复口径（F4-rev，修正初版表述）

Gateway shutdown 会将**成功 flush 的** pending messages 持久化
（shutdown_flush.py → pending_messages/，#72680 防丢失），新进程
recover_pending_to_db() 恢复到会话历史。该机制保障已 flush 消息的数据可恢复，
但不保证消息自动作为新 turn 执行，也不保证跨平台 exactly-once。
消息重新执行依赖平台重投/用户重发。异常退出（SIGKILL/断电）不承诺全量可恢复。

## 协议不变量（ChatGPT 第三轮钉死，全部进测试）

- I1：只有同一目标 generation 的有效请求才能获得 ACCEPTED；接受结果必须在
  该 generation 退出前可靠持久化
- I2：同一 Service Identity 在冷却窗口内不得因 generation 更替、不同 profile
  或重复请求获得额外 deferred restart 预算
- I3：新增 deferred 请求处理不得改变旧 SIGUSR1 调用的兼容语义；deferred 请求
  校验失败时不得经 legacy fallback 意外触发重启（防投放无效请求文件制造重启）

## 威胁模型（覆盖对象）

普通 shell 用户、同一 hermes home 下其他 profile、cron 子进程、恶意本地进程、
PID 复用、supervisor 配置变更。actor_ref 不作授权输入，授权不信任 CLI 自报；
cron 覆盖是 v1 已知缺口（解耦 follow-up change 处理），冷却窗口为上界。

## Non-goals（明确不做）

- 不加 N 秒延迟参数（v1 turn 结束驱动）
- 不做持久化消息重投队列（原 N5 留后续专项）
- 不改 /restart、`hermes gateway restart`、SIGUSR1 既有语义（I3）
- 不新增 core tool、不新增环境变量配置
- Windows 第一版显式报"不支持"
- 不动本机 workaround skill 配方

## 可翻转点（Reversible decisions）

- 冷却窗口 600s、受理确认窗口 ~5s、退出码映射（0/2/3/4/5/6）
- 状态文件新增字段（revision / accepted_request_id）
- 请求文件目录名与过期清理时限

## 讨论记录归档

- chatgpt-round1.txt：第一轮架构评审（5 设计问题、7 blocking、事实核实清单）
- chatgpt-round2.txt：第二轮（F1-F7 核实结果、B8/B9 修正、有条件冻结裁决）
- codex-spec-review-round1.txt：Codex 双保险评审（1 blocking 协议缺口、
  F4 事实纠正、7 项修改清单）
- chatgpt-round3.txt：第三轮（分歧裁决选 (a) 请求文件协议、F4-rev 确认、
  协议不变量 I1-I3、"同意冻结，无需第四轮"）
