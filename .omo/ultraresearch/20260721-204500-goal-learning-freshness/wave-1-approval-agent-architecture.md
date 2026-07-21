# Wave 1 — approval queue and AI-agent boundary

## Digest

- Current queue claim/release behavior is at-most-once for normal successful
  application.  Failed application is requeued; held claims are not
  automatically replayed.
- Explicitly confirmed goal outcomes are already session/root scoped,
  freshness-checked, pull-only candidates.  They do not auto-inject memory or
  modify skills.
- Primary guidance supports post-approval revalidation and decision audit
  without storing raw sensitive payloads: OpenAI
  [HITL](https://openai.github.io/openai-agents-python/human_in_the_loop/),
  [running agents](https://openai.github.io/openai-agents-python/running_agents/),
  [tracing](https://openai.github.io/openai-agents-python/tracing/),
  [MCP elicitation](https://modelcontextprotocol.io/specification/2025-06-18/client/elicitation),
  [OWASP MCP](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html),
  and [NIST AI RMF](https://airc.nist.gov/docs/AI_RMF_Playbook.pdf).
- The `verification_evidence` suite is red in the home-contained temporary
  directory because its workspace detector selects the home Git ancestor;
  this is an environment baseline, not a diff in this worktree.

## EXPAND

- LEAD: pending write에는 payload digest·expiry·대상 precondition이 없음 — WHY: 장시간 대기 뒤의 승인이나 대상 변경이 stale action으로 실행될 수 있음 — ANGLE: `tools/write_approval.py` stage/approve 경로에 canonical digest와 revalidation contract 설계.
- LEAD: reject/cancel/success/requeue의 terminal decision은 append-only 감사 ledger로 보존되지 않음 — WHY: NIST식 human override·adjudication audit과 incident 복구 근거가 불완전함 — ANGLE: raw payload 비보존 decision receipt schema 및 failure-ordering 테스트.
- LEAD: `blocked`·`cancelled` outcome terminal kind는 정의됐지만 production emitter가 없음 — WHY: goal 종료 원인의 감사 범위와 user-visible outcome reporting이 불완전함 — ANGLE: `GoalManager.clear/pause`와 gateway cancellation 경로에서 nonreusable audit receipt를 명시 기록할지 검토.
- LEAD: verification-evidence 테스트는 home Git ancestor가 temp marker root를 가리는 Windows 환경 문제로 실패함 — WHY: 현재 receipt freshness 회귀를 신뢰성 있게 검증할 수 없음 — ANGLE: `_git_root`와 `_marker_root` 우선순위 및 home-contained temp workspace 회귀 테스트.
- DEAD END: LangGraph HITL 문서의 구 URL은 redirect만 반환되어, 본 보고의 근거에는 사용하지 않음.
