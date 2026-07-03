# Harbor Terminal-Bench Runner

`scripts/run_harbor_terminal_bench.sh` launches Harbor Terminal-Bench runs with
Hermes Agent against an OpenAI-compatible autoscaler gateway.

The script is intentionally generic: it does not hardcode personal usernames,
hostnames, private key paths, or internal gateway IPs. Provide those values via
environment variables when running it.

If `HARBOR_DIR` does not exist, the script clones the patched Harbor fork/ref
from `HARBOR_REPO_URL` and `HARBOR_REF`. The bundled
`scripts/patches/harbor-hermes-custom-endpoint.patch` is kept as a fallback for
unpatched upstream Harbor checkouts and is only applied when
`APPLY_HARBOR_PATCH=1`.

## Required Environment

```bash
export AUTOSCALER_SSH_TARGET="user@example-host"
export AUTOSCALER_SSH_KEY="$HOME/.ssh/id_ed25519"
export AUTOSCALER_REMOTE_GATEWAY="gateway-host-or-ip:30090"
```

You can copy the example environment file:

```bash
cp scripts/harbor-terminal-bench.env.example .env.harbor-terminal-bench
set -a
source .env.harbor-terminal-bench
set +a
```

## Optional Environment

```bash
export HARBOR_DIR="../harbor"              # Harbor checkout path
export HARBOR_REPO_URL="git@github.com:NousResearch/harbor-fork.git"
export HARBOR_REF="hermes-custom-endpoint"
# export APPLY_HARBOR_PATCH="1"            # Only for unpatched upstream Harbor
export HERMES_MODEL="hermes-large"         # Autoscaler model id
export LOCAL_PORT="30090"                  # Local SSH tunnel port
export N_TASKS="10"                        # Unset for a full Terminal-Bench run
export N_CONCURRENT="1"                    # Harbor concurrency
export EXCLUDE_TASK_NAME="gpt2-codegolf"   # Excluded by default for smoke runs
```

## Run A Smoke Task

```bash
INCLUDE_TASK_NAME=cancel-async-tasks \
  ./scripts/run_harbor_terminal_bench.sh
```

## Run A 10-Task Batch

```bash
N_TASKS=10 ./scripts/run_harbor_terminal_bench.sh
```

## Run A Larger Batch

```bash
N_TASKS=50 N_CONCURRENT=4 ./scripts/run_harbor_terminal_bench.sh
```

## Run The Full Dataset

Leave `N_TASKS` unset so Harbor does not receive `--n-tasks`:

```bash
unset N_TASKS
N_CONCURRENT=1 ./scripts/run_harbor_terminal_bench.sh
```

## Head Node Guard

The script refuses to run on hostnames that look like head nodes, for example
hosts containing `-hn1` or `head`, unless explicitly overridden:

```bash
ALLOW_HEAD_NODE_RUN=1 ./scripts/run_harbor_terminal_bench.sh
```

Only use the override when you are sure the host is an appropriate place to run
Docker/Harbor workloads.

## Notes

- Harbor task containers use `http://host.docker.internal:<LOCAL_PORT>/v1` by
  default because Hermes runs inside Docker.
- `OPENAI_API_KEY` defaults to `dummy`; it is only used to populate Hermes'
  custom provider config for the autoscaler endpoint.
- Set `NO_TUNNEL=1` if a local tunnel or gateway is already running.
- By default the script owns the inference tunnel. It checks `/v1/models`
  periodically and restarts the SSH tunnel if it stops responding during a long
  eval.
