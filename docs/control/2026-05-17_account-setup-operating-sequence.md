# Account Setup Operating Sequence

> Safest order for the first account lanes.

**Support sequence**
**Decision owner:** Haz

## Goal

Create the accounts in order. No role bleed. No credential reuse. No recovery confusion.

## Operating order

### 1) Create the browser profiles first

Create these profiles before login:

- `profile-qa-testing`
- `profile-research-secure`
- `profile-recovery-admin`
- `profile-business-admin-<brand>` when business rollout begins
- `profile-business-support-<brand>` when business rollout begins

Rules:

- one family, one profile
- no cross-use
- keep secure profiles minimal

### 2) Set up the recovery lane

Use the recovery profile to set up recovery.

Do:

- assign recovery email
- assign recovery phone if needed
- generate backup codes
- store backup material outside this repo
- confirm recovery works

### 3) Set up the secure research lane

Use the secure profile for the high-trust account.

Do:

- log in only there
- keep extensions minimal
- keep sign-ins limited
- confirm MFA works

### 4) Set up the testing lane

Use the testing profile for the messy account.

Do:

- create or confirm the testing account
- keep it low-friction
- store nothing valuable there
- treat it as resettable

### 5) Only then create business accounts

Create business accounts only when each one has a real workflow.

For each business account, confirm:

- purpose
- owner
- recovery path
- MFA method
- browser profile
- allowed device(s)
- trust level

If any are missing, stop.

## Creation sequence for a new business account

1. Name the role and brand
2. Decide the owner
3. Choose the browser profile
4. Set the recovery path
5. Enable MFA
6. Record the safe metadata
7. Test sign-out and recovery
8. Confirm isolation

## Pause triggers

Stop and review if:

- two roles share one profile
- recovery details are unclear
- MFA is untested
- business accounts appear without a real workflow
- convenience starts beating separation

## Handoff

When this sequence is followed, fill in the registry draft and fix any missing fields before the account matters.
