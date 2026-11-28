# ATM Patch Requirements — the contract an updated patch MUST satisfy

This document is the knowledge base for the escalation path of the fork-maintenance
pipeline: when the mechanical rebase of the ATM patch stack onto upstream fails (or
tests fail after it), an agent is given this document and expected to produce an
updated patch. It states WHAT the patch must provide and WHY, so the patch can be
re-derived even if upstream refactors everything it currently touches.

## What the patch is

`NousResearch/hermes-agent` has no public way for an external process to inject a
message into a running gateway session. The ATM patch adds exactly one seam:

**`GatewayRunner.inject_internal_message(...)`** — a public, keyword-only async API
that the `hermes-atm` package (pip; source in atm-core `crates/hermes-atm`) calls to
deliver agent-team-mail nudges into a profile's chat session. Everything else in the
stack exists to support or test that seam.

Current stack shape as of v0.23.1 (12 commits above the base tag; replay ALL of them on every rebase):
1. `feat: reseat public injection seam onto mixin-split run.py` — the API + hook exposure + tests
2. `fix(gateway): P1 review fixes HGF-001..004 (injection seam)` — review-round hardening
3. `style(tests): strip trailing whitespace (HGF-005)`
4. `test(gateway): startup-emit stub absorbs upstream heartbeat-restore addition`
5. `docs(atm): release-publishing model (vX-atm) + pipeline corrections`
6. `docs(atm): HGF-003 verify commands for the v0.22.4 pyproject layout`
7. `docs(readme): add Synaptic Canvas fork banner` — the README fork banner is a
   STACK PATCH (Rand ruling 2026-10-23): ALL fork content rides the stack and is
   replayed by the rebase; nothing lives main-only. (It was originally main-only
   and got silently reverted by a whole-tree sync merge — PR #33 — which is the
   incident that produced this rule.)
8. `fix(kanban): end the owning executor when a card is closed externally` —
   fork-only fleet fix (zombie headless executors kept running after their card
   went terminal; live incident 2026-11-05 t_2a7d0b3e). Originally fork PR #37;
   landed on the stack per the banner ruling (nothing lives main-only), PR #37
   closed as landed-via-stack. **UPSTREAM SUBMISSION CANDIDATE** — when it
   merges upstream AND is contained in the base tag, DROP it from the stack at
   the next replay (same removal rule as the beads fork rows); its 5 files then
   leave the diffstat invariant.
9. `docs(atm): stack shape 10 above base — kanban fix rides the stack (this
   commit)` — the shape/invariant update for item 8.
10. `cron: keep --no-agent script jobs off the async-delegation fire path` —
    fork-only fleet fix (hx-dk4, from grecon-bd8 2026-10-06 family; repro
    2026-11-11 runtime-v5): direct `hermes cron run` on a --no-agent --script
    job dispatched an async delegation whose CLI owner exited before terminal
    recording — script never ran, run row stuck 'claimed', later fires blocked
    by "already being fired". Gate sends no-agent jobs to the synchronous
    path (terminal state in-process). Bug verified LIVE on upstream main
    (fb5d314b10) — **UPSTREAM SUBMISSION CANDIDATE**, same removal rule as
    item 8: when it merges upstream AND is contained in the base tag, DROP it
    at the next replay; its 2 files then leave the diffstat invariant.
11. `docs(atm): stack shape 12 above base — cron no-agent fire-path fix rides
    the stack (this commit)` — the shape/invariant update for item 10.

Stack diffstat invariant: `git diff vX^{commit}..stack-tip` touches exactly
**13 files** as of v0.23.1 — the 6 ATM/banner files (5 seam/contract +
`README.md`), the kanban fix's 5 (`hermes_cli/kanban_db.py`,
`hermes_cli/kanban_db_termination.py`,
`tests/hermes_cli/test_kanban_complete_worker_termination.py`,
`tests/tools/test_kanban_worker_self_exit.py`, `tools/kanban_tools.py`), and
the cron fire-path fix's 2 (`tools/cronjob_tools.py`,
`tests/tools/test_cronjob_run_background.py`).
The kanban 5 leave the invariant when item 8 retires; the cron 2 leave when
item 10 retires (upstream merge + base containment). A sync candidate touching
fewer files dropped a patch; more files grew the stack without a contract
update.

## The contract (every item is load-bearing)

1. **Public API surface** — `GatewayRunner` must expose:
   ```python
   async def inject_internal_message(
       *, profile: str, platform: Platform, chat_id: str, text: str,
       notice_text: Optional[str] = None,
       mode: Literal["queue", "steer"] = "queue",
   ) -> None
   ```
   plus `InjectInternalMessageError(code, chat_id, detail)`. hermes-atm validates at
   install time that `inject_internal_message` is callable on the runner and calls it
   with exactly these keywords. Changing names/signature breaks every deployed
   hermes-atm wheel — do not.

2. **Hook exposure** — the `gateway:startup` hook context dict must contain
   `"gateway_runner": self`. This is how hermes-atm's installed hook obtains the
   runner without private imports. (Upstream may rename the emit site; the key in the
   context dict must survive.)

3. **Profile resolution, fail-closed** — resolve the target profile via the runner's
   profile-adapter map (or the active profile). Unknown profile / empty adapter map
   MUST raise `InjectInternalMessageError` — never fall through to a default chat.
   Misrouted injection = message delivered to the wrong Telegram chat. Fail closed.

4. **Modes** — `"queue"`: enqueue via the platform adapter's normal inbound-message
   path (fire-and-forget). `"steer"`: if the profile's agent is mid-turn and exposes
   a steer capability, inject into the running turn; otherwise fall back to queue.
   Queue is the default and the only mode hermes-atm currently uses in production.

5. **Visible notice is soft-fail** — when `notice_text` is provided, send it via
   `adapter.send(chat_id, notice_text, metadata={"notify": True})` BEFORE routing the
   event; check the result's `success` and log a warning (with the result error) on
   failure; catch exceptions and log. A notice failure must NEVER prevent the main
   event from routing. (This is commits 2–3 of the stack.)

6. **Tests are the contract's enforcement** —
   `tests/gateway/test_inject_internal_message.py` (26 tests: queue, steer,
   isolation, error cases, notice soft-fail). The updated patch must keep all 26
   green. Only change a test when upstream semantics genuinely force it, and then
   re-verify hermes-atm compatibility (its hook + runtime must still work — see
   "Definition of done").

## Known conflict hotspots

- `gateway/run.py` import block (upstream churns `datetime`/`typing` imports; ours
  adds `Literal`, `Optional`). Union the imports — this is the most common conflict
  and is always trivially resolvable.
- The `inject_internal_message` method body sits in `GatewayRunner` (a huge class
  upstream refactors freely). If upstream moves adapter access or the event-routing
  entry point (`adapter.handle_message` today), re-wire the seam to the new
  internals while keeping the public surface identical.
- The `gateway:startup` emit site (search: `hooks.emit("gateway:startup"`).

## How to update the patch (escalation procedure)

1. Work in a scratch clone. NEVER in `~/Documents/forks/hermes-agent` directly,
   never force-push anything (force-push needs interactive user auth — you run unattended), never touch branch protection.
2. `git fetch upstream main`; start from `upstream/main`; attempt
   `git rebase upstream/main` of the `atm/stack` branch.
3. Resolve conflicts per the contract above — the contract, not the old diff, is the
   spec. If upstream now provides an equivalent public injection API, prefer
   adapting hermes-atm to it and SHRINKING the patch (goal state: patch size zero).
4. Verify (deps frozen from the repo's own `uv.lock`; Python per the repo's
   `.python-version` — 3.14 as of v0.22.4, `requires-python >=3.11,<3.15`).
   pytest lives in the `dev` DEPENDENCY-GROUP (not an extra), and the repo sets
   `default-groups = []`, so it must be requested explicitly (HGF-003, updated
   for the v0.22.4 pyproject layout):
   `uv sync --frozen --extra messaging --group dev` then
   `uv run --frozen --extra messaging --group dev python -m pytest tests/gateway/test_inject_internal_message.py`
   → all collected tests pass (31 as of v0.22.4: 26 seam + 5 P1 regressions),
   plus `tests/gateway/test_hooks.py` (6 as of v0.22.4) and any test file the
   conflict touched. Gate on ZERO failures; totals drift across releases.
5. Definition of done: tests green AND a live check that
   `hermes_atm.HermesAtmRuntime.from_gateway_runner` accepts the runner (import
   `hermes_atm`, construct against a stub runner exposing the API — the seam tests
   include this shape) — then push the branch and open a PR to `main`; the review
   pipeline (contessa → qwen reviewer) takes it from there.
6. If the contract itself cannot be satisfied (upstream removed a capability the
   seam needs), STOP and escalate to Rand with a written analysis — do not ship a
   behavioral compromise.
