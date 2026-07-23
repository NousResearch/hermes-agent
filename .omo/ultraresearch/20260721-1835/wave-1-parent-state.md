# Wave 1 — parent state and source cross-check

## Verified local findings

- `hermes_cli/goals.py` persists `GoalState` per session with an explicit
  completion contract, subgoals, wait/resume state, budget, and CAS-backed
  wait claiming. Older records load without newer optional fields.
- The active goal loop records a `judge_done_unconfirmed` outcome receipt
  only after a judge returns `done`; it does not promote a receipt to learning
  automatically.
- `agent/verification_evidence.py` keeps outcome receipts in a separate
  SQLite ledger. Goal text is stored only as a SHA-256 digest. A receipt is
  reusable only after same-session confirmation and current passing evidence;
  the reusable read path excludes receipts after a later workspace edit.
- `list_reusable_outcome_receipts()` is implemented as an explicit, pull-only
  API but currently has no production presentation call site. CLI, gateway,
  and TUI only expose `/goal confirm`.
- Existing `subgoals` are durable textual criteria but do not carry explicit
  lifecycle state or a progress ledger.

## External source cross-check

- LangGraph describes durable checkpoints/interrupts as the basis for human
  approval and restartable long-running state:
  https://docs.langchain.com/oss/python/langgraph/persistence
  https://docs.langchain.com/oss/python/langgraph/interrupts
- OpenAI Agents SDK documents durable human approval with serializable run
  state and warns that persisted state must be handled as data:
  https://openai.github.io/openai-agents-python/human_in_the_loop/
- OWASP's agentic guidance identifies persistent memory/context as a security
  boundary; this supports retaining Hermes's explicit, non-injected receipt
  design: https://genai.owasp.org/download/49059/

## Provisional direction

The smallest compatible extension is an evidence-aware goal control plane:
durable, user-visible subgoal progress plus explicit receipt-learning
visibility. It must not inject prior outcomes into prompts, alter memory or
skills, imply a model judge is an approval, or disclose another session's
receipt data.

## EXPAND

- LEAD: add a bounded status/summary API for outcome receipts — WHY: the
  existing pull-only API is otherwise not operable from Hermes user surfaces
  — ANGLE: inspect all three command transports and receipt privacy tests.
- LEAD: structured subgoal lifecycle — WHY: current textual criteria have no
  durable progress/replanning signal — ANGLE: preserve backward-compatible
  state JSON and judge completion semantics.
