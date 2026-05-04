# Skill Intake Policy Matrix

## Default recommendation

Use `conservative` as the default profile.

Reason:
- favors governance over convenience
- blocks risky imports early
- keeps conversion subordinate to audit

## Policy profiles

### conservative
- critical => rejected
- high => rejected
- medium => review_required
- low => approved_with_warning
- info => approved

### balanced
- critical => rejected
- high => review_required
- medium => approved_with_warning
- low => approved
- info => approved

### permissive
- critical => review_required
- high => approved_with_warning
- medium => approved
- low => approved
- info => approved

## Additional admission rules

1. Conversion never implies approval.
2. Re-audit is mandatory after conversion.
3. Compatibility and security must remain separate axes.
4. A structurally valid skill may still be rejected on risk.
5. Repo/source reputation may inform review later, but must not override critical findings in V2.

## Risk examples

### critical
- explicit secret exfiltration
- instructions to bypass system/developer controls
- remote arbitrary execution with hidden payloads

### high
- curl|bash
- sudo without clear necessity
- download-and-run without checksum
- reading SSH keys, cloud creds, browser cookies, Keychain, `.env`

### medium
- unpinned dependencies
- implicit telemetry/webhooks
- dangerous wildcard file operations
- strong repo-specific coupling or hidden assumptions

### low
- weak metadata
- missing compatibility notes
- missing pitfalls/verification guidance

## Outcome mapping

- A/B + low => admit or approve with warning
- B/C + low => convert then re-audit
- any + medium => human review
- any + high => quarantine/reject by policy
- D + critical => reject
