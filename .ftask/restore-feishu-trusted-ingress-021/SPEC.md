# SPEC — restore-feishu-trusted-ingress-021

> The agent fills this by running the BOUNDED dimensional clarifier (see the
> Spec-first protocol: score Objective/Metric/Target/Scope, ask one question
> per round at the weakest, exit at ambiguity ≤20%), reads it back, and only
> runs `ftask spec restore-feishu-trusted-ingress-021 --approve` once sunke says OK. No code until
> approved. This is the non-coder's real review gate.
>
> The 'How will I know it works' section is the Karpathy gate — `--approve`
> parses it and refuses to flip status if Surface / Acceptance scenarios /
> Regression guards are empty or placeholder. Filling this section honestly
> is what lets the LLM LOOP toward done instead of guessing.

## Done — ONE measurable sentence (fill LAST, after the interview)
> The crisp goal the interview converges to. Must be checkable, not vague —
> it doubles as the future drive-to-green stop condition.
> e.g. 'subscribe-v3 import maps all 7 fields; imported row count = source ±0'.
- Hermes Agent 0.21.3 的 Feishu adapter 恢复可选的进程内签名 ingress ticket/admission 契约，启用 admitter 时 WebSocket 与 Webhook 消息/卡片事件先验签并 fail-closed，未启用时保持原生行为，且定向测试全部通过。

## What sunke wants (plain language)  [Objective]
- 把旧版个人 Hermes 提交 `817264a67b` 中 multitenancy 依赖的 Feishu 受信入口安全边界移植到当前 0.21.3，并提交 PR 后合并。

## Out of scope (what we will NOT do)  [Scope]
- 不修改 multitenancy 插件自身代码，不重新设计飞书消息卡片，不改变普通 Hermes 用户未安装 admitter 时的 Feishu 行为，不处理其他平台或飞书凭据配置问题。

## 任务类型 fix

## 根因 (fix 必填 — 不写根因就会同一个 bug 修两遍)
> 这个缺陷的真实成因是什么?在哪一层进入系统?哪些调用方共享同一个根因?
- 更新到 0.21.3 主线时，个人发布提交 `817264a67b` 中的 `TrustedFeishuIngressTicket`、`_trusted_ingress_admitter` 与 WebSocket/Webhook 入口封装没有进入新主线；multitenancy 因缺少这一显式核心契约在注册阶段 fail-closed，无法接管飞书消息与 CardKit。

## 拷问(写 Done 前) — 运行 /grilling 与 sunke 对齐到共识;共识即落 Done。(T0/T1 可跳过)

## How will I know it works (Karpathy gate — required to approve)

### Surface (which user-facing surface — pick one or more)
- [ ] web — Interceptor / agent-browser harness
- [ ] cli — fresh shell + actual command
- [ ] api — curl against real endpoint
- [x] lib — 5-line consumer script
- [ ] none — pure doc/config change (no simulate step)

### Visual target (web surface only — 钉死"长成什么样才算对")
> 仅 web surface 任务需填: 参考图路径 / 设计稿 URL / 一句可视判定(如"侧边栏宽 240px、企业名居中")。
> 这是前端视觉验收的对比基准 — 没有它,"看着对"无法机器核验。
- 

### Acceptance scenarios (each = observable user action + observable outcome)
Format: 'user does X → observe Y' (use → to separate action from outcome)
- 外部插件在 FeishuAdapter 上安装 admitter 后发送 WebSocket 消息或卡片回调 → adapter 生成进程内签名 ticket，只有 admitter 返回 admission 才把封装事件交给下游。
- 通过 Webhook 发送同类事件 → 使用同一 ticket/admission 边界，验签失败或 admission 为空时不执行下游处理。
- 对 ticket 篡改账户、身份、时间或签名 → `is_valid` 返回 false，事件被拒绝。
- 不安装 admitter 使用原生 Feishu → 消息、卡片、评论、会议和 reaction 继续走 0.21.3 现有处理路径。
- multitenancy 在装有本修复的 0.21.3 上注册 → 能找到并安装 `_trusted_ingress_admitter`，不再以 “Feishu core lacks trusted ingress contract” 退出。

### Regression guards (what must NOT break — list things to recheck)
- WebSocket 与 Webhook 的事件类型覆盖保持一致，message read 与 bot membership 回调不受影响。
- 未安装 admitter 的标准 Hermes Feishu 行为保持向后兼容。
- 启用 admitter 后消息与业务卡片 fail-closed，reaction 保持既有回调语义，comment/VC 不获得新的业务权限。
- 原有去重、批处理、per-chat serialization 和 approval/update-prompt 回调保持通过。

### Targeted tests (repo-relative paths; one per bullet, or `full-suite`)
> The direction model lists only tests affected by this task. Invalid/missing targets block; full-suite is CI-only.
- tests/gateway/test_feishu_trusted_ingress.py
- tests/gateway/test_feishu.py
- tests/gateway/test_feishu_approval_buttons.py

## Plan (long tasks only — ordered route + live progress; T1/short may leave empty)
> Steps DERIVED from the Done line (not a chat-plan). Tick `[ ]`→`[x]` as you go.
> This is the compaction-survival anchor: after an auto-compact, read this to see
> exactly which steps are done and what's next — never re-run finished steps.
- [ ] 将 `817264a67b` 的 ticket/envelope/admission 语义适配到 0.21.3 当前 Feishu adapter，明确 opt-in 与 fail-closed 分界。
- [ ] 接入当前 WebSocket/Webhook handler、消息事件 metadata 传播及卡片回调，同时保持未安装 admitter 的原生路径。
- [ ] 移植并补充安全回归测试，运行定向测试和 multitenancy 真实注册探针。
- [ ] 通过 ftask TEST / LEAK / REVIEW 门禁，提交 PR、合并并读回结果。

## Dead ends (filled DURING work — approaches tried & rejected, don't retry)
> Append one line per rejected approach: `approach → why it failed`. Read
> this before each new attempt so the same wrong path isn't tried twice.
- 直接用 0.21.3 `hermes plugins doctor --ci` 验证 multitenancy → Doctor 隔离环境不加载 bundled Feishu platform，触发 `live Feishu adapter module did not materialize`；真实集成验收改为先加载当前 bundled platform 再注册插件。
