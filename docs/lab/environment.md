# IT Automation Lab Environment

## Base Requirements

Use the repository root as the lab workspace.

Recommended tools:

- Python 3.11
- project virtual environment: `.venv` or `venv`
- Node.js 20+ for UI and integration scripts
- Git
- ripgrep
- shellcheck for shell script review, when available

## Python Environment

Prefer the project virtual environment:

```bash
source .venv/bin/activate
```

If this checkout uses `venv` instead:

```bash
source venv/bin/activate
```

## Test Runner

Use the canonical test wrapper, not direct `pytest`:

```bash
scripts/run_tests.sh
```

Targeted examples:

```bash
scripts/run_tests.sh tests/agent/
scripts/run_tests.sh tests/path/to/test_file.py::test_name -v
```

## Node-based Areas

Some subprojects have their own Node dependencies, for example:

- `ui-tui/`
- `web/`
- `scripts/whatsapp-bridge/`

Run their package-manager commands from the relevant subdirectory, not blindly from the repository root.

## Local Configuration

- User config: `~/.hermes/config.yaml`
- Local secrets: `~/.hermes/.env`
- Example env file: `.env.example`

Do not commit real secrets.

## Lab Validation Checklist

Before running an automation experiment:

- [ ] Confirm the current Git branch and working tree.
- [ ] Confirm whether the target is local, staging, or production.
- [ ] Read the related runbook.
- [ ] Run in read-only or dry-run mode first when available.
- [ ] Capture verification output.
- [ ] Update docs if behavior or commands changed.
