# Account Segmentation Handoff

> Keep accounts split by role.

**Support handoff**

**Decision:** split by role, not convenience.

## What is already decided

- One QA account stays the sandbox.
- One high-trust research account stays locked down.
- Add business accounts only with a named purpose and owner.
- Browser/profile separation is the floor.
- No cookie reuse. No shared passwords. MFA everywhere.
- No VPN blocking is a fact, not the security model.

## Recommended structure

- `qa-testing` for messy flows.
- `business-admin` / `business-support` for live ops.
- `research-secure` for sensitive research.
- `recovery-admin` for recovery only.

## What still needs doing

- Assign owners for future business accounts.
- Record recovery before business accounts matter.
- Choose the browser profile naming on the MacBook.
- Decide whether any account needs extra hardening.

## Discoverability surfaces

- Use `docs/control/2026-05-17_account-setup-start-here.md` as the front door.
- The index and manifest point here.
- The QC checklist is ready.

## Next safe move

Create business accounts only after purpose, owner, recovery path, profile, and trust level are named.
