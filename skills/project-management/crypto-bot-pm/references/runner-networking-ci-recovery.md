# Runner job-container networking for local Gitea CI

Use this reference when S006/local Gitea Actions gets past runner scheduling and JavaScript action startup but `actions/checkout` fails because the job container cannot resolve the local Gitea hostname.

## Symptom pattern

- The dedicated CI job image is already in use, e.g. runner label `ubuntu-latest:docker://crypto-bot-ci-runner:python313-node20-go`.
- Logs no longer show `Cannot find: node in PATH`.
- Logs do show `node .../actions/checkout.../dist/index.js` and `git version ...`.
- Checkout fails with DNS/network evidence such as:

```text
fatal: unable to access 'http://crypto-bot-gitea:3000/preston/crypto_bot/': Could not resolve host: crypto-bot-gitea
```

Treat this as runner infrastructure, not a product workflow failure.

## Durable fix pattern

For `gitea/act_runner`, configure the per-job containers to join the same Docker network as local Gitea:

```yaml
container:
  network: "crypto-bot-gitea-net"
```

In this control plane, the bounded recovery helper should render/install a managed config in the runner data volume, for example `/data/config.yaml`, and start/register the runner with `CONFIG_FILE=/data/config.yaml` so the image entrypoint passes `--config` during registration and daemon startup.

## Fail-closed inspection

Inspection should fail when a registered runner has the dedicated image label but lacks a readable config containing an uncommented `container.network` value for `crypto-bot-gitea-net`. Avoid substring-only checks that can pass on comments such as `# network: "crypto-bot-gitea-net"`.

Recommended evidence fields:

- `runner_config_path`
- `runner_job_container_network`
- `docker_config_exit_code`
- `runner_config_network_detected`
- `workflow_dispatch_invoked: false` unless a separate CI rerun approval exists

## Execution pitfalls

- The recovery helper defaults to inspect mode. Passing `--approval-phrase` alone is not execution; approved live repair must include `--execute --approval-phrase "..."`. If the result JSON says `"mode": "inspect"` and `"steps": []`, no runtime mutation happened.
- After recovery, run a fresh `--inspect` and require both the dedicated image label and `runner_config_network_detected: true` before treating the runner as healthy.
- For DNS/network validation, a direct probe from the dedicated CI job image on `crypto-bot-gitea-net` is good evidence: resolve `crypto-bot-gitea` and run a read-only `git ls-remote` for the S006 branch. Redact/avoid credential-bearing URLs; local unauthenticated read-only Gitea probes are acceptable when the repo is locally readable.
- If a control-plane preflight fails on source/runtime skill parity, repair and commit the skill/reference sync before runner or CI actions. Do not bypass parity failures just because the runtime action was approved.
- If a command pipeline fails after writing JSON evidence, inspect the evidence file before retrying. This prevents repeating the same failed command shape and catches cases where the helper ran in inspect mode by mistake.

## Governance

Runner networking repair approval is separate from S006 CI rerun approval. Do not dispatch workflows, mutate PR/check/status state, merge, push product branches, or begin S007A until the relevant approval explicitly covers that action.