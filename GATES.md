# Canonical Readiness Gate

The only canonical readiness gate for the autonomy foundation layer is:

```bash
python scripts/run_readiness.py
```

This script is backed by:

- `/Users/jonathannugroho/Developer/PersonalProjects/Hermes-agent/hermes-agent/autonomy_policy.yaml`
- `/Users/jonathannugroho/Developer/PersonalProjects/Hermes-agent/hermes-agent/.github/workflows/readiness.yml`

There are no non-blocking shortcuts in this gate. Any failing command is blocking.

## Required GitHub Protection

This local gate is not sufficient by itself. The GitHub repo must also enforce:

- protect `main`
- require pull requests before merge
- require the `Readiness` workflow as a required status check
- block direct pushes to `main`
- block force-pushes to `main`
