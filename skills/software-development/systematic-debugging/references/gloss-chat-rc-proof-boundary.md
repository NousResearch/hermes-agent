# Gloss Chat RC Proof Boundary — Session Note

## Context

User asked whether Gloss had all intended features and whether they were all working, specifically noting prior chat issues.

## Durable lesson

For feature-completeness/status questions, distinguish three proof levels:

1. **Source presence / implementation evidence** — files, UI controls, commands, feature flags, receipt entries exist.
2. **Automated proof** — targeted gates, unit tests, contract tests, and static validation pass.
3. **Live product proof** — headed app launch with real provider/model, real user workflow, persistence/reload, packaging launch, and performance timing.

Do not answer "yes, working" from levels 1–2 alone. Say "automated RC green" or "release-candidate" until level 3 has been exercised.

## Gloss-specific example

Automated chat proof that was green:

- `validate_source_send_gate.py` — source-list loading/error/partial state does not hard-disable send.
- `validate_frontend_event_routing.py` — frontend terminal handlers are not filtered by active notebook ID.
- `validate_chat_terminal_contract.py` — backend terminal event contract is present.
- `commands::chat::tests` — provider done-frame / terminal metadata regressions covered.
- Provider tests and frontend contract tests passed.

Known chat fixes represented by those gates:

- `SourceScope::None` must remain valid; no-retrieval chat cannot be blocked by source state.
- Backend emits exactly one terminal event: `chat:done`, `chat:error`, or `chat:cancelled`.
- Terminal events match by `messageId`, not active notebook ID.
- Assistant message persistence happens before `chat:done`; persistence failure emits `chat:error`.
- Provider `done=true` can terminalize without waiting for HTTP EOF.

Still not live-proven by those gates:

- Real chat in the headed desktop app with actual configured provider/model.
- Import source → embed/index/retrieve → answer with citations.
- No-retrieval chat through the GUI.
- Stop/regenerate/continue flows in a real session.
- Notebook switch during/after streaming.
- Assistant persistence after app reload.
- AppImage launch and performance timing.

## Recommended wording

Use blunt states:

- **Automated RC green**: gates/tests pass.
- **Release-candidate**: intended release-scope code paths appear present and automated proof is green.
- **Release-proven**: headed live smoke + real provider/model + packaging/perf have passed.

If the artifact receipt says something exists, verify the current filesystem artifact before repeating it as current truth.
