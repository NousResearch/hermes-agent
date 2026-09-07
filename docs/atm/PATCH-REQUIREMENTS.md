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

Current stack shape (3 commits + this doc commit):
1. `feat: expose public gateway injection seam` — the API + hook exposure + tests
2. `fix(gateway): notify visible internal-message notices` — notice delivery hardening
3. `test(gateway): cover soft visible-notice failure`

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
4. Verify (all with Python 3.11, deps frozen from the repo's own `uv.lock`):
   `uv sync --frozen --no-dev --extra messaging` then
   `python -m pytest tests/gateway/test_inject_internal_message.py` → 26 passed,
   plus `tests/gateway/test_hooks.py` and any test file the conflict touched.
5. Definition of done: tests green AND a live check that
   `hermes_atm.HermesAtmRuntime.from_gateway_runner` accepts the runner (import
   `hermes_atm`, construct against a stub runner exposing the API — the seam tests
   include this shape) — then push the branch and open a PR to `main`; the review
   pipeline (contessa → qwen reviewer) takes it from there.
6. If the contract itself cannot be satisfied (upstream removed a capability the
   seam needs), STOP and escalate to Rand with a written analysis — do not ship a
   behavioral compromise.
