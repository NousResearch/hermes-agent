# Governed builder orchestrator

The builder orchestrator turns an owner-registered implementation contract into
one restricted Hermes Kanban worker. The worker receives an immutable execution
packet, can edit only allowed repository paths, and must pass the registered
validation profile before the adapter records completion evidence.

The adapter does not push, open or merge pull requests, approve its own work, or
accept arbitrary filesystem paths and shell commands from callers.

## Operator workflow

The adapter must already be running with an owner-only runtime configuration.
The CLI defaults to `~/.hermes/builder-adapter/runtime.json`; set
`HERMES_BUILDER_ADAPTER_CONFIG` or pass `--config` to use another file.

Check the local service and list the jobs the owner has registered:

```console
hermes orchestrate health
hermes orchestrate cycles
```

Prepare a new job proposal without activating or starting it:

```console
hermes orchestrate prepare \
  --repo /absolute/path/to/repository \
  --repository-id my-project \
  --cycle FEATURE_EXAMPLE_001 \
  --contract FEATURE-EXAMPLE-001 \
  --goal "Add the requested behavior" \
  --accept "Focused tests pass" \
  --accept "Existing behavior remains compatible" \
  --allow "src/example.py" \
  --allow "tests/test_example.py" \
  --branch "feat/example-001" \
  --worktree "/absolute/new/worktree/path"
```

`prepare` inspects the clean repository, pins its current commit and canonical
remote, rejects repository-wide write access, and writes an owner-only JSON
proposal under `~/.hermes/builder-jobs/pending/`. It creates no worktree,
changes no governance state, and starts no worker. The printed SHA-256 identifies
the exact proposal submitted for governance review.

Activate a reviewed proposal:

```console
hermes orchestrate activate ~/.hermes/builder-jobs/pending/FEATURE_EXAMPLE_001.json
```

Activation verifies the proposal hash and pinned repository again, writes a
cycle-specific contract and path manifest into the clean governance repository,
commits those artifacts, creates the isolated linked worktree, and atomically
registers the cycle in the owner-only runtime configuration. It does not start a
builder. Reload and verify the supervised adapter after activation:

```console
hermes orchestrate restart
```

The restart is graceful and uses the already-loaded macOS launch agent. It does
not reconstruct or print the adapter secret. Readiness succeeds only when the
replacement process reports the exact registered-cycle configuration
fingerprint, then the job can be started.

Start one registered job:

```console
hermes orchestrate start CYCLE_ID
```

The command prints the generated dispatch ID and the exact status command. A
specific dispatch UUID can be supplied with `--dispatch-id` when recovering an
idempotent request whose response was lost.

Monitor the job and retrieve its evidence:

```console
hermes orchestrate status DISPATCH_ID --cycle CYCLE_ID
hermes orchestrate evidence DISPATCH_ID --cycle CYCLE_ID
```

Cancel only when the job should genuinely stop:

```console
hermes orchestrate cancel DISPATCH_ID --cycle CYCLE_ID
```

Cancellation terminates the worker process tree and archives the native task;
it is not a pause operation.

## What is registered before `start`

`start` deliberately cannot invent a task contract. The owner-controlled
runtime and governance snapshot bind:

- objective and acceptance criteria;
- repository, branch, worktree, and exact starting commit;
- permitted paths;
- builder model and tool policy;
- validation commands and isolation policy;
- runtime, heartbeat, and retry limits.

This separation keeps the convenient operator command from becoming an
unrestricted remote-code-execution interface. Preparing new contracts is an
administrative operation; starting and monitoring an already registered job is
an operator operation.

## Authentication

Authenticated commands use the active key registered for the current local Unix
user. The secret is read only from its approved environment variable and is
never printed or written by the CLI. If several keys match, select one with
`--key-id`.
