# Optional native continuation (unreleased)

The stable board is manual/cooperative. Native continuation is an optional
integration that additionally requires a compatible Hermes host and the
separately supplied external plugin `session-coord-native`. Nothing in the board
release implies that the required Hermes API is upstream, accepted, installed,
or active in an already running process.

## Setup gate

Use the repository-root helper. Profile selection is always explicit:

```text
python3 hermes_setup.py check --profile default --plugin-path /absolute/plugin
python3 hermes_setup.py check --all-profiles --plugin-path /absolute/plugin --json
python3 hermes_setup.py setup --profile default --profile work --plugin-path /absolute/plugin
```

`check` is read-only. `setup` preflights all selected profiles before the first
mutation and then uses these sanctioned boundaries:

```text
hermes -p PROFILE plugins doctor /ABS/PLUGIN --ci
hermes -p PROFILE plugins install file:///ABS/PLUGIN --ref FULL_SHA --no-enable
hermes -p PROFILE plugins enable session-coord-native --no-allow-tool-override
hermes -p PROFILE config set delegation.wait_for_all true
hermes -p PROFILE config get delegation.wait_for_all --json
hermes -p PROFILE session-coord native-check --json
```

The local plugin source must be a clean Git worktree with a committed HEAD. That
is necessary because the supported installer clones a `file://` Git URL; Doctor
and installation must evaluate identical immutable bytes.

A config value is not capability proof. `native-check` must return a JSON object
with:

- `supported: true` only after real native-source registration;
- a non-empty string list in `surfaces`;
- `wait_for_all_supported: true` only when the public effective-policy helper
  exists;
- actual Boolean `wait_for_all: true` after setup;
- `activation: "fresh_process_only"`;
- a string `reason`.

Strings such as `"true"`, integers, null, malformed JSON, or duplicate JSON keys
fail the gate. Native-source support and joined-delegation support are separate
facts. The helper writes the managed native enrollment only after every selected
profile and requested watchdog pass.

Successful setup means **installed and configured on disk; safe restart still
required**. It never restarts a CLI, TUI, Desktop, or gateway process and never
changes a model/provider.

## Yield and continue

After successful setup, the separate managed native block permits a claim like:

```text
python3 <session_coord_path> claim --id ID --res KEY --yield \
  --checkpoint /absolute/existing-checkpoint.json
```

A `--yield` request stores the immutable complete resource set, checkpoint path,
and exact receiver target. Exit 75 means the current turn must stop immediately:
no polling, mutation, extra tool call, or synthetic follow-up.

The native host polls only at a proven idle, live, owned session boundary with no
queued human turn. It reserves the turn slot before the plugin's pre-model commit.
A successful commit admits a normal new turn containing the exact continuation
command:

```text
python3 <session_coord_path> continue --id BOARD --event EVENT
```

`continue` revalidates the durable event and atomically reacquires the full set.
No historical message, system prompt, or provider setting is rewritten.

## Identity and delivery invariants

- Targets contain exact profile home, surface, session id, and lineage supplied
  by the runtime; never infer them from a directory name or current shell.
- A receipt proves native prompt admission, not task completion.
- A false commit performs no model/history action.
- An uncertain outcome remains unknown and is not retried blindly.
- Stop, new, switch, close, and exit serialize cancellation before identity
  rotation. Explicit cancel tombstones the event.
- A yielded subagent process is not resumed. Its exact parent receiver
  redispatches a fresh child from checkpoint.
- An offline receiver remains pending; native continuation does not start an
  offline Hermes process.

## Registration compatibility

An exact absolute `profile_home`, receiver `session_id`, and target `kind` remain
required. A profile label is optional for legacy targets: the receiver checks the
exact profile home rather than guessing a label from a directory name. When a
`profile` label is supplied, it must be a non-empty string and match the host's
profile label.

New registrations preserve supplied `compression_lineage`, or seed it with only
the known receiver session id. A child actor's identity is never substituted for
its parent receiver. Explicit lineage must be a non-empty string list containing
the receiver; the host rejects ancestry outside its own known lineage. No
unobserved ancestors are inferred.

Existing stored targets without these optional metadata fields remain readable
under the same exact-home/session checks. Validation never backfills or rewrites
those payloads, their hashes, or existing wait episodes. Invalid metadata that is
present is rejected rather than replaced with a default.

## One shared watchdog

The repository installer only copies `coord_resume_watchdog.py`. Scheduling is
explicit and plugin-owned:

```text
hermes -p default session-coord watchdog-setup --json --check
hermes -p default session-coord watchdog-setup --json
```

`hermes_setup.py --watchdog` invokes these commands through profile `default`
only. The default cron store is the sole owner of the shared board's one recurring
job; the helper never creates per-profile jobs. The plugin must inspect enabled
and disabled copies in that owner store, refuse duplicates or semantic drift,
and read back scheduler state. Check mode does not mutate. Jobs created manually
in other profile stores are outside this helper's proof and must be removed by the
operator.

The expected semantic job is named `session-coord-resume-watchdog`, runs every two
minutes, executes `coord_resume_watchdog.py` with `no_agent: true`, and delivers
locally. The script only calls `wake-reconcile --json`; it uses no model, network,
or provider.

After a create timeout or nonzero result, do not issue another create. Run the
read-only check, report observed persistence and recovery, and resolve duplicates
before retrying.

## Recovery

| Status | Recovery |
|---|---|
| `unsupported_host` | Install a Hermes build with the public native-turn-source registrar, then rerun `check` |
| joined policy unsupported | Install the build containing the public effective delegation policy helper; do not set an unproven key |
| `action_needed` enrollment | Compare the preserved custom block with the managed example; merge manually, then rerun |
| `partial` config/plugin setup | Rerun `check` first; do not hand-edit `config.yaml` |
| uncertain watchdog create | Run `watchdog-setup --json --check`; never create a second job while state is unknown |
| pending event while receiver offline | Start a fresh configured receiver normally; reconciliation does not launch Hermes |
| explicit Stop/cancel | Keep the tombstone; do not redeliver the canceled event |

The legacy `--wait` flow remains valid and is not backfilled into native events.
Board-only users should continue using manual `check`/`inbox` or bounded shell
waiting.
