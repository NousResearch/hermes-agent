# Spec-orchestrator procedure

Load `references/protocol.md` first.

## Job

Turn a raw idea into a frozen-ready packet and run the task graph until a human can merge with SHA-tied evidence.

## Steps

1. Research the repo before asking product questions. Never ask what the code can answer.
2. Interview in short rounds (1-4 questions, options + recommendation first) until two engineers would ship the same observable behaviour.
3. Draft the packet with `templates/packet.md`. Size to one day of agent work or less; larger work becomes a chain of packets/units.
4. Create the root kanban card in **`triage`** with the full body. Do not assign a spawnable builder yet if that would land the card in `ready` before freeze. Prefer `--triage`.
5. Show the human the packet. They freeze `triage` → `ready` on the unchanged body.
6. After freeze, create an ordinary **build** child task:
   - body includes Factory role: builder, packet version, AC/NG summary or pointer to root
   - workspace: `worktree` for code
   - assignee: builder profile
   - parent: root (or prior unit)
7. When build completes with handoff, create an ordinary **review** child task:
   - Factory role: reviewer
   - pin packet version + full SHA from build handoff
   - assignee: reviewer profile
   - **do not** set kernel status `review`
8. On `changes-requested`, create a new build-fix task then a new review task. Do not ask the reviewer to push fixes.
9. When review returns approve-evidence, present the human merge checklist: verdict SHA == head SHA, CI policy, residual risk.
10. After human merge, complete/close the graph with PR URL + packet version.

## Never

- Self-freeze
- Specify/Decompose the approved packet
- Put factory review on kernel `review` status
- Merge
