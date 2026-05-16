# Step 1: codify-undersea-harness-operating-model

## Objective
Update the Undersea Friends operating model so automation and structural changes are planned and executed through harness phases/steps instead of ad-hoc profile instructions.

## Scope
- Modify existing docs only where appropriate:
  - `docs/HARNESS.md`
  - `~/.hermes/shared-memory/undersea-friends/README.md`
  - `~/.hermes/shared-memory/undersea-friends/FRIENDS_INDEX.md` if role routing needs clarification.
- Do not print, move, or copy secrets.
- Do not restart gateways or modify live process state.

## Required content
- Define Nemo as the execution/verification gate for harness phases.
- Define Manta as the front-door planner for AX/research/automation framing.
- Define Whale as canonicalization, data quality, and long-term structure guardian.
- Define Shark as security/permissions/secret-risk reviewer.
- Define Octopus as multi-tool automation/prototype helper.
- Keep direct user-to-profile conversation allowed; harness is for automation/change work, not every casual message.

## Acceptance Criteria
- Existing documentation states that material automation, file/config changes, gateway/provider/process work, and cross-profile handoffs should use a harness phase or a handoff packet.
- Documentation includes the minimum handoff packet fields: `goal / why / current state / paths / risks / execution steps / verification / report format`.
- No secret values are added to any document.
- Run: `python -m json.tool phases/index.json >/dev/null` and `python -m json.tool phases/0-undersea-friends-harness-structure/index.json >/dev/null`.
