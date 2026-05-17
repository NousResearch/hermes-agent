# Recovery + MFA Setup Checklist

> Make each account recoverable and harder to compromise. No secrets in repo.

**Support checklist**
**Decision owner:** Haz

## Purpose

Keep recovery explicit. Test MFA. No secrets in repo.

## Checklist

### Before setup

- [ ] Owner named.
- [ ] Purpose named.
- [ ] Recovery path known.
- [ ] Browser profile assigned.
- [ ] Allowed device(s) known.
- [ ] Trust level written down.

### Recovery setup

- [ ] Recovery email assigned.
- [ ] Recovery phone assigned if needed.
- [ ] Backup codes saved outside this repo.
- [ ] Recovery access is separate from daily login.

### MFA setup

- [ ] MFA enabled.
- [ ] MFA method chosen on purpose.
- [ ] Primary MFA works.
- [ ] Backup MFA is tested.
- [ ] Account still works without the main device.

### Validation

- [ ] Sign out and re-check recovery.
- [ ] Recovery works without role mix.
- [ ] Recovery profile is not daily-use.
- [ ] No live credentials or backup codes in repo.

## Pause triggers

Stop and fix the setup if:

- recovery path is unclear
- backup codes are not stored safely elsewhere
- MFA is on but untested
- the account shares a profile
- recovery and daily login blur together

## Safe handling

Do not store passwords, backup codes, QR codes, or live credential material in this document.
