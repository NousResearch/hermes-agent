# Wave 1 — architecture librarian digest

## Key findings

- Magentic-One distinguishes a task ledger from a progress ledger and bounds
  stalled work with replanning. A2A models long-running work with explicit
  state updates, artifacts, input/authorization waits, and idempotent
  delivery expectations.
- LangGraph and Temporal both support preserving intent/history across resume
  but require idempotent external effects. Google ADK separates deterministic
  trajectory/outcome evaluation from rubric quality checks.
- WebArena-Verified favors deterministic structural/network-trace evaluation
  over an LLM judge for critical conditions. AgentLab records versions for
  reproducibility. OpenTelemetry task semantics are still a proposal and
  should not become a hard schema dependency.
- The worker recommends a small `GoalLedger v0`: goals, commitments, receipts,
  verification records, and bounded learning candidates. Its larger schema is
  useful as a north-star but is not yet selected for direct implementation;
  local compatibility and the existing evidence ledger take precedence.

## Sources

- https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/magentic-one.html
- https://a2a-protocol.org/latest/specification/
- https://docs.langchain.com/oss/python/langgraph/functional-api
- https://docs.langchain.com/oss/python/langgraph/interrupts
- https://docs.temporal.io/workflow-execution
- https://adk.dev/evaluate/
- https://github.com/ServiceNow/webarena-verified
- https://github.com/ServiceNow/AgentLab
- https://openai.github.io/openai-agents-python/tracing/

## Worker EXPAND markers (verbatim)

- LEAD: Hermes의 실제 task/event/trace·DB 인터페이스 — WHY: GoalLedger의 안전한 삽입 지점과 마이그레이션 범위가 아직 미확인 — ANGLE: `rg -n "goal|task|event|trace|receipt|eval|sqlite" <Hermes-repo>`
- LEAD: 현재 Hermes의 고위험 도구 승인 경로 — WHY: `awaiting_approval` 상태와 정책 게이트를 기존 계약에 정확히 연결해야 함 — ANGLE: `rg -n "approval|clarify|permission|destructive|merge" <Hermes-repo>`
- LEAD: A2A Python SDK의 현행 타입/상태 매핑 — WHY: 대외 에이전트 상호운용을 구현할 때 내부 GoalLedger를 A2A Task로 손실 없이 변환해야 함 — ANGLE: `site:github.com a2a python sdk TaskState artifact`
- LEAD: OpenTelemetry GenAI task 규약의 표준화 상태 — WHY: 내부 telemetry 이름을 미래 표준으로 매핑할 시점이 달라짐 — ANGLE: `site:github.com/open-telemetry/semantic-conventions gen_ai.task status`
