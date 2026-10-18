type: spec-tasks
project: hermes-agent (upstream PR)
task: gateway-deferred-restart

# Tasks: gateway-deferred-restart

## T1 CLI 子命令骨架
- [ ] gateway parser 注册 `defer-restart`（无参数；拒绝任何参数 → exit 2）
- [ ] 处理器流水：解析目标 → 预检 → 冷却/幂等 → 写请求文件 → SIGUSR1 → 确认 → 报告
- [ ] 退出码 0/2/3/4/5/6 + 输出文案

## T2 目标解析与身份
- [ ] 复用 profile→进程解析（_matches_current_profile / find_profile_gateway_processes）
- [ ] generation identity = PID + 进程启动时间（防 PID 复用）
- [ ] service identity = (canonical hermes home, supervisor type, supervisor service id)
- [ ] multiplex：共享宿主的 profile 归并同一 service identity / 授权范围 / 冷却范围

## T3 supervisor 能力预检
- [ ] launchd：plist 存在 + job 已加载 + KeepAlive/重启语义
- [ ] systemd：unit 存在 + 重启策略覆盖 75 + 非 start-limited
      （含 RestartPreventExitStatus / SuccessExitStatus 组合）
- [ ] s6：supervise 在管
- [ ] 无 supervisor / Windows / foreground-detached → exit 4，不发信号
- [ ] 预检绑定信号目标（不得先查 A 再给无关 PID 发信号）

## T4 请求归因协议（请求文件 + 受理确认）
- [ ] restart_requests/ 目录 0700 + 文件 0600 + tmp+rename+目录 fsync；
      拒符号链接、校验 owner/普通文件类型/路径边界
- [ ] 请求文件 schema：schema_version / request_id / service_identity /
      target_generation / requested_at / actor_ref（自报线索，不作授权）
- [ ] gateway 侧请求消费：校验代际/service identity/schema/授权 → 原子认领 →
      request_restart() 成功排程后才写 ACCEPTED（线性化点语义）
- [ ] ACCEPTED 持久化先于最终退出；accepted_request_id 按 request_id 可查、
      跨代际保留（retention 规则），新进程启动不清理受理证据
- [ ] 并发：每代际至多一个有效请求；并发请求显式结局（accepted/duplicate/
      cooldown/expired），不静默丢弃
- [ ] 崩溃语义：写后未发信号 → 过期清理不触发重启；claim 前 gateway 崩溃 →
      文件留给过期清理；过期清理不删 accepted/unknown 结果记录
- [ ] CLI 确认谓词：自身 request_id + 目标代际匹配 → exit 0；窗口 ~5s；
      陈旧/损坏 JSON 不当正常状态 → exit 6（禁止立即重试）

## T5 跨进程冷却 + 幂等（不变量 I2）
- [ ] 冷却存储：hermes home 受保护目录 JSON，原子替换 + 文件锁
- [ ] 键 = 稳定 service identity；600s 自受理时刻起算
- [ ] 同 request_id 内部重试返回原决策不占预算；独立调用=新请求
- [ ] 测试：窗口内拒绝（exit 5）、重启后冷却仍生效、multiplex 双 profile 共享预算

## T6 SIGUSR1 兼容（不变量 I3）
- [ ] 无有效 deferred 请求 → 旧 SIGUSR1 行为逐字节不变
- [ ] deferred 请求校验失败 → 拒绝 + 记日志，绝不 fall through 到无条件重启
- [ ] signal handler 不做阻塞文件 I/O；读取/锁/状态提交在事件循环异步任务里

## T7 守卫扩展（不放松 G1/G2）
- [ ] （移出本 PR）G3 cron 三维禁用与 audit 日志：留 spike 分支，
      作为解耦 follow-up change 独立送审
- [ ] 覆盖缺口文档：cron payload 可驱动 deferred 通道为 v1 已知缺口，
      冷却窗口为上界（每 service 每窗口至多一次）
- [ ] 回归测试：G1/G2 现有 15 探针行为逐字节不变（拼接/折行/组合 fail-closed）

## T8 重启链路测试
- [ ] request_restart 幂等 + 异步排程不阻塞事件循环（回归，非新假设）
- [ ] shutdown 清理不取消 restart task（回归——已知实现已跳过自身清理）
- [ ] drain：active-work 归零后才 stop；stream 最终 edit 在释放前完成
- [ ] active-work=0 窗口的受理交接竞态测试（I1）
- [ ] 超时路径沿用现有恢复语义

## T9 文档（具体文件）
- [ ] `hermes gateway defer-restart --help` 与 argparse epilog
- [ ] website/docs 对应 gateway CLI 章节（消息恢复口径按 F4-rev：
      成功 flush 可恢复、不自动重投为新 turn、无 exactly-once、异常退出不承诺）
- [ ] 平台支持矩阵：macOS launchd / Linux systemd / s6 契约支持，Windows 明确不支持

## T10 测试计划与覆盖声明
- [ ] macOS launchd 真实 E2E：触发 → 回复送达 → 重启 → 会话续接，四步留证
- [ ] 契约测试（不依赖真实 supervisor）：systemd/s6 预检判定、Windows 拒绝、
      multiplex 归并、无 supervisor 拒绝
- [ ] 未能真实 E2E 的 supervisor 在 PR 里明确标注测试覆盖限制
