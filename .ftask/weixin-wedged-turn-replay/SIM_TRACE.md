---
slug: weixin-wedged-turn-replay
generated_at: 2026-09-24T18:41:01.174Z
spec_revision: a29737a3f079
surfaces: [lib]
scenarios:
  - id: 1
    surface: lib
    action: |
      agent 在工具调用前遇到一个会阻塞的 subdirectory hint 路径 → hint 读取在短边界内放弃且工具执行链继续/受控失败，不会无限占用 worker。
    observed: |
      Captured run 1: 67 focused tests passed; bounded hint reads include the no-unbounded-is_file regression.
    verdict: pass
  - id: 2
    surface: lib
    action: |
      gateway watchdog 判定 turn inactivity timeout → 旧 turn 被标记为终止且在稍后解除 I/O 阻塞后也不能继续执行后续工具或覆盖响应。
    observed: |
      Captured run 2: an interrupt released during real subdirectory-hint result commit prevents the next tool dispatch.
    verdict: pass
  - id: 3
    surface: lib
    action: |
      同一 inbound message 因 watchdog timeout 失败 → transcript 形成单一 user/assistant 关闭边界，平台重投或队列 drain 不会把该旧请求重新当作新的 user turn 执行。
    observed: |
      Captured run 3: 28 gateway timeout, failure ownership, and session race tests passed.
    verdict: pass
code_diff_hash: 4d306ab39e1b7a4210523c4b9a86c7258810d4de19cd18bfd73bd15d8cff82b0
---

# Simulation trace — weixin-wedged-turn-replay

Agent: for each scenario set a `verdict` (pass / fail / inconclusive) plus ONE
piece of evidence — either a `ftask simulate <slug> --capture <id> -- <cmd>` run
(ftask records exit/stdout you can't fabricate; preferred) OR paste real output
into `observed` (web: an Interceptor screenshot path). expected/rationale are
optional context. Save large artifacts (screenshots, network logs) to
`/Users/kesun/.hermes/hermes-agent/.ftask/weixin-wedged-turn-replay/sim_artifacts/`.

Verdict legend:
- `pass` — observed matches expected.
- `fail` — observed contradicts expected. **Blocks ship.**
- `inconclusive` — agent couldn't fully verify (missing env / external dep);
  rationale MUST explain why. Allowed through ship.

## Captured runs (ftask --capture audit trail; do NOT hand-edit — re-run --capture to refresh)

- scenario_id: 1
  at: 2026-09-25T14:36:19.978Z
  command: "zsh -lc uv run --extra dev pytest -q tests/agent/test_subdirectory_hints.py"
  cwd: /Users/kesun/.hermes/hermes-agent.tasks/weixin-wedged-turn-replay
  exit_code: 0
  duration_ms: 3045
  stdout_tail: |
    ....................................................................     [100%]
    68 passed in 2.59s
  stderr_tail: |
    (empty)

- scenario_id: 2
  at: 2026-09-25T14:36:22.800Z
  command: "zsh -lc uv run --extra dev pytest -q tests/agent/test_tool_call_guardrail_runtime.py -k 'interrupt_during_subdirectory_hint_commit'"
  cwd: /Users/kesun/.hermes/hermes-agent.tasks/weixin-wedged-turn-replay
  exit_code: 0
  duration_ms: 1149
  stdout_tail: |
    .                                                                        [100%]
    1 passed, 13 deselected in 0.88s
  stderr_tail: |
    (empty)

- scenario_id: 3
  at: 2026-09-25T14:36:29.557Z
  command: "zsh -lc uv run --extra dev pytest -q tests/gateway/test_abandoned_turn_process_cleanup.py tests/gateway/test_failure_writer_ownership.py tests/gateway/test_session_race_guard.py"
  cwd: /Users/kesun/.hermes/hermes-agent.tasks/weixin-wedged-turn-replay
  exit_code: 0
  duration_ms: 5116
  stdout_tail: |
    ............................                                             [100%]
    28 passed in 4.85s
  stderr_tail: |
    (empty)

