# P0 Contract Pack

This directory freezes the phase-0 contract artifacts for the Hermes Cron
failover/control-plane work.

Delivered artifacts:

- `job-metadata.schema.json`
- `jobs-registry.schema.json`
- `evidence.schema.json`
- `verdict.schema.json`
- `audit.schema.json`
- `state-machine.json`
- `state-machine.md`
- `risk-register.md`
- `examples/`
- `fixtures/`
- `validate_phase0.py`

Validation:

```bash
python /Users/ryanchao/.hermes/worktrees/cron-control-p0/docs/cron-control/p0/validate_phase0.py
```

Scope:

- Documentation and validation artifacts only.
- No runtime source files are changed here.
