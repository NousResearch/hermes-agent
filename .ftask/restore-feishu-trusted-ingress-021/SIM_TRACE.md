---
slug: restore-feishu-trusted-ingress-021
generated_at: 2026-10-06T10:52:24.786Z
spec_revision: 263623f020ad
surfaces: [lib]
scenarios:
  - id: 1
    surface: lib
    action: |
      外部插件在 FeishuAdapter 上安装 admitter 后发送 WebSocket 消息或卡片回调 → adapter 生成进程内签名 ticket，只有 admitter 返回 admission 才把封装事件交给下游。
    observed: |
      
    verdict: pass
  - id: 2
    surface: lib
    action: |
      通过 Webhook 发送同类事件 → 使用同一 ticket/admission 边界，验签失败或 admission 为空时不执行下游处理。
    observed: |
      
    verdict: pass
  - id: 3
    surface: lib
    action: |
      对 ticket 篡改账户、身份、时间或签名 → `is_valid` 返回 false，事件被拒绝。
    observed: |
      
    verdict: pass
  - id: 4
    surface: lib
    action: |
      不安装 admitter 使用原生 Feishu → 消息、卡片、评论、会议和 reaction 继续走 0.21.3 现有处理路径。
    observed: |
      
    verdict: pass
  - id: 5
    surface: lib
    action: |
      multitenancy 在装有本修复的 0.21.3 上注册 → 能找到并安装 `_trusted_ingress_admitter`，不再以 “Feishu core lacks trusted ingress contract” 退出。
    observed: |
      
    verdict: pass
code_diff_hash: 96962669333c24a4a548d14b2317e2a3a05dcf7d063a22bcf3af9bb205afab0f
---

# Simulation trace — restore-feishu-trusted-ingress-021

Agent: for each scenario set a `verdict` (pass / fail / inconclusive) plus ONE
piece of evidence — either a `ftask simulate <slug> --capture <id> -- <cmd>` run
(ftask records exit/stdout you can't fabricate; preferred) OR paste real output
into `observed` (web: an Interceptor screenshot path). expected/rationale are
optional context. Save large artifacts (screenshots, network logs) to
`/Users/kesun/.hermes/hermes-agent/.ftask/restore-feishu-trusted-ingress-021/sim_artifacts/`.

Verdict legend:
- `pass` — observed matches expected.
- `fail` — observed contradicts expected. **Blocks ship.**
- `inconclusive` — agent couldn't fully verify (missing env / external dep);
  rationale MUST explain why. Allowed through ship.

## Captured runs (ftask --capture audit trail; do NOT hand-edit — re-run --capture to refresh)

- scenario_id: 1
  at: 2026-10-06T10:53:01.193Z
  command: "uv run --directory /Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021 --extra dev --extra messaging --extra feishu pytest -q tests/gateway/test_feishu_trusted_ingress.py -k admitted_message_and_card_receive_signed_envelope"
  cwd: /Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021
  exit_code: 0
  duration_ms: 3109
  stdout_tail: |
    ......                                                                   [100%]
    6 passed, 16 deselected in 2.32s
  stderr_tail: |
    (empty)

- scenario_id: 2
  at: 2026-10-06T10:53:03.489Z
  command: "uv run --directory /Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021 --extra dev --extra messaging --extra feishu pytest -q tests/gateway/test_feishu_trusted_ingress.py tests/gateway/test_feishu.py -k authenticated_webhook_uses_trusted_dispatch_with_webhook_transport or webhook_authentication_requirement_only_applies_with_admitter or rejected_message_and_card_fail_closed_without_identifier_logs or url_verification_requires_configured_verification_token"
  cwd: /Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021
  exit_code: 0
  duration_ms: 2149
  stdout_tail: |
    .....                                                                    [100%]
    5 passed, 101 deselected in 1.62s
  stderr_tail: |
    (empty)

- scenario_id: 3
  at: 2026-10-06T10:53:05.363Z
  command: "uv run --directory /Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021 --extra dev --extra messaging --extra feishu pytest -q tests/gateway/test_feishu_trusted_ingress.py -k ticket_is_signed_scoped_and_time_bounded"
  cwd: /Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021
  exit_code: 0
  duration_ms: 1777
  stdout_tail: |
    .                                                                        [100%]
    1 passed, 21 deselected in 1.26s
  stderr_tail: |
    (empty)

- scenario_id: 4
  at: 2026-10-06T10:53:07.215Z
  command: "uv run --directory /Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021 --extra dev --extra messaging --extra feishu pytest -q tests/gateway/test_feishu_trusted_ingress.py -k no_admitter_preserves_native_dispatch or websocket_handlers_cross_the_optional_dispatch_seam"
  cwd: /Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021
  exit_code: 0
  duration_ms: 1750
  stdout_tail: |
    ......                                                                   [100%]
    6 passed, 16 deselected in 1.22s
  stderr_tail: |
    (empty)

- scenario_id: 5
  at: 2026-10-06T10:53:08.698Z
  command: "/Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021/.venv/bin/python3 /Users/kesun/.hermes/hermes-agent/.ftask/restore-feishu-trusted-ingress-021/sim_artifacts/register_probe.py"
  cwd: /Users/kesun/.hermes/hermes-agent.tasks/restore-feishu-trusted-ingress-021
  exit_code: 0
  duration_ms: 1383
  stdout_tail: |
    FEISHU_MODULE=hermes_plugins.feishu_platform.adapter
    ADMITTER=admit_trusted_feishu_ingress
    REGISTER_OK
  stderr_tail: |
    (empty)
