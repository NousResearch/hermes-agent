# Local Gitea Actions CI image/toolcache hardening

Use this reference when S006/local Gitea Actions has progressed past scheduling, Node startup, and checkout, but fails inside validation jobs because the local act_runner job image/toolcache does not match GitHub-hosted runner assumptions.

## Symptom pattern

This is a later-stage runner/tooling issue when all of these are true:

- Dedicated CI job image is active, for example `ubuntu-latest:docker://crypto-bot-ci-runner:python313-node20-go`.
- `actions/checkout` runs through Node successfully.
- Checkout fetches the S006 ref from `http://crypto-bot-gitea:3000/preston/crypto_bot` successfully.
- Logs do **not** contain `Cannot find: node in PATH`.
- Logs do **not** contain `Could not resolve host: crypto-bot-gitea` or checkout `exit code 128`.

Typical failures after those blockers are fixed:

```text
Version 3.13 was not found in the local cache
The version '3.13' with architecture 'arm64' was not found for this operating system.
Failure - Main Set up Python
```

and:

```text
scripts/validation/validate-governance-baseline.sh: line 26: rg: command not found
ERROR: missing governance marker 'MAPE-K' in docs/MAPEK_PROGRAM.md
```

Interpret these as CI image/toolcache hardening issues, not product-code failures and not checkout/network regressions.

## Durable fix pattern

Harden the dedicated job image and runner toolcache contract rather than editing product workflows by default:

1. Add missing validation tools used by product scripts to the dedicated CI image, including `ripgrep` (`rg`).
2. Make `actions/setup-python` find Python `3.13` on local linux/arm64. Options to evaluate in order:
   - pre-populate the toolcache path and metadata expected by `actions/setup-python` for Python 3.13 arm64;
   - set the appropriate runner/toolcache environment for act_runner job containers if supported;
   - as a last resort, adjust the dedicated CI image/runner config so the already-installed Python 3.13 is discoverable without requiring network downloads.
3. Keep the fix in Hermes control-plane infrastructure unless the workflow itself is explicitly approved for mutation.
4. Add regression coverage so the Dockerfile/config asserts required tools and toolcache setup are present.

## Evidence sequence for reruns

Before dispatching another S006 rerun:

1. Run `crypto_bot_gitea_runner_recovery.py --inspect --format json` and require `PASS` with:
   - dedicated image label detected;
   - `runner_config_network_detected: true`;
   - old host-mode `ubuntu-latest:host` absent.
2. Run the read-only PR/CI audit for exact S006 branch/head.
3. Dispatch only `validate.yml` for `hermes/dev13-006-daemon-trust-contract-mapping` when separately approved.
4. Poll Gitea Actions tasks until terminal states.
5. Read logs by `action_run_job.id`, not task id, when using the jobs logs endpoint. In the local DB, `action_run_job.task_id` maps to the API task id, while the logs endpoint expects the job id.
6. Summarize new blockers separately from resolved blockers. Explicitly state whether Node, DNS, checkout, Python toolcache, and `rg` failures are present or absent.

## Token hygiene

If a temporary Gitea token is required for dispatch/log reads:

- create a narrowly-scoped short-lived token;
- never print or persist its value;
- record only the token name and action outcome;
- clean it up immediately after use;
- if the API cannot delete it, remove the local token record by exact token name in the local Gitea DB and record that cleanup without token material.

## Governance

S006 CI rerun approval authorizes only the bounded workflow rerun/log-read evidence path. It does not authorize product-file edits, workflow edits, PR metadata/comment/check/status mutation beyond Gitea naturally recording the dispatched run, merge, push, S007A, or broad CI infrastructure mutation. Runner/image hardening is a separate control-plane implementation step and must follow the current runner recovery approval gates before another rerun.