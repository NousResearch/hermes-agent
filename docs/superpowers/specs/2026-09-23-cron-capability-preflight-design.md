# Cron capability preflight + local-job failure alerts

Date: 2026-09-23. Status: implemented.

## Incident

`Ready-for-Staging Auto-Merger — BA` (`cron:2d9eee30cb90`) was built on
2026-09-16 in a Slack session with a `builder`-role user. That role had
read-only Jira and (because of the MCP alias bug fixed 2026-09-17) no visible
GitHub tools. The agent first refused ("cannot safely create or enable this
agent yet"), was then told to go ahead without a terminal, and created the job
with `enabled_toolsets: [jira, messaging, file]`, the tools it could see
itself. Ownership then moved to Leetal (`rd`, which grants everything the job
needs), but the job's toolsets were never widened.

For a week the job ran every 5 minutes with `last_status: ok`. It processed 17
candidate events, merged nothing, and posted none of its blocker notices:

- `messaging` is stripped from every cron run, so `send_message` never existed.
- It had no `jira_write`, so no Jira comment or transition.
- Merge and branch update are approval-gated, and `approvals.cron_mode: deny`
  blocks them headless unless the job lists them in `unattended_approved_tools`.
- It was `deliver: local`, so none of this reached a human.

None of these steps (the MCP merge, the cron strip, the owner-role ceiling,
the approval gate) is visible at creation time, and each one fails silently.

### Second incident: drift after creation

`Agentic Drive signed order forms → MRR sheet` (`cron:e5f851dec63d`) was
built correctly on 2026-06-29. It had no owner, so it ran uncapped with
`terminal` and wrote sheet updates through a writer script until 08-31. Then
two later changes broke it, and neither one checked the job:

1. **2026-09-14:** an admin claimed the job and transferred it to a user whose
   role has no `terminal`. The RBAC limit silently stripped `terminal`. On
   09-16 and 09-17 the agent searched for "execute shell command", found
   nothing, and left its plans unapplied in `/tmp`.
2. **2026-09-18:** the gateway began running from `.venv`. The collector
   imports `google` and `pypdf`, which exist only in the other venv
   (`venv`, the one the `hermes` CLI uses). Every run afterwards died on
   `ImportError`. The agent described the error in prose, and the run
   recorded `ok`.

A policy follows from this: **automated jobs never rely on `terminal`**,
which is limited to super admins. Any shell step in an automation must be a
deterministic script that a super admin attaches, not a command an LLM
chooses.

## Design

### One evaluation: `cron/capability_preflight.evaluate_job_capabilities`

It reproduces the scheduler's pipeline: `_resolve_cron_enabled_toolsets`
(per-job list with enabled MCP servers merged in), then
`apply_cron_toolset_ceiling(owner_grant)`, then removing
`_resolve_cron_disabled_toolsets`. It checks each **required** toolset against
the result.

Required toolsets are the union of:

1. the job's `required_toolsets` field, a new `cronjob` parameter;
2. `requires_toolsets` in `<workdir>/automation.yaml`, the automation bundle's
   manifest;
3. every name the job lists explicitly in `enabled_toolsets`. Naming a
   toolset the job can never receive is always a mistake.

A requirement fails when it is:

- unknown;
- stripped in cron (with a hint: `messaging` → `slack_post`);
- outside the owner's RBAC grant (floor toolsets always pass);
- missing from an explicit per-job list;
- or, at create/update time only, has no usable tool on the host (check_fn:
  missing credentials or binary, MCP server not connected).

`mcp-<server>` folds onto `<server>`, the spelling the resolver and the
ceiling use.

The report also returns `gated_unacked`: approval-gated tools the agent can
reach that are not in `unattended_approved_tools`. The existing
`_unattended_ack_error` only walks explicitly listed toolsets, so it misses
MCP servers merged in at run time. Rejecting on these would break creating
any job on a host with gated MCP tools, and the gate already blocks them, so
they are reported rather than rejected.

### Surfaces

| Surface | Requirements | Availability probe | On failure |
|---|---|---|---|
| `cronjob` create | declared + explicit | yes | reject |
| `cronjob` update touching toolsets, requirements, workdir, acks, no_agent | declared + explicit | yes | reject |
| `cronjob` update, other fields | declared + explicit | yes | `capabilities.warnings` |
| scheduler preflight (4th check) | declared + explicit | no | `blocked_config`, alert once |
| `ownership transfer` of `cron:` | declared + explicit | no | refuse unless `confirm_capability_loss` |

Create evaluates against the creator's grant, since the creator becomes the
owner. Update, runtime, and transfer evaluate against the owner from the
ownership registry, the same subject as the runtime ceiling.

The runtime check skips the availability probe, so a transient MCP disconnect
cannot block a run and send an alert. (A first version also ignored
`enabled_toolsets` at run time to protect legacy jobs; measuring every
enabled VM job showed the rule flags only the broken ones, so it was dropped.)

The rejection text tells the agent to report the missing access and who must
grant it, and never to drop the requirement. This targets the specific failure
here: an agent downgrading the job to what its current session could see.

### Failure visibility

- **`[FAILED]` marker.** A final response that starts with `[FAILED]` becomes
  `success=False` with error `Agent reported failure: <reason>`. That feeds
  `failure_streak`, the incident store, and the failure alert. The cron
  system hint tells the agent to use it. `_summarize_cron_failure_for_delivery`
  passes the reason through before its substring heuristics, so a reason
  mentioning "401" is not relabelled a provider failure.
- **Owner DM for local-only jobs.** When a failure alert is composed for a
  job with no delivery target, `_alert_owner_of_undelivered_failure` DMs the
  owner through `automation_ownership._send_dm`. It sends once per incident
  signature and then marks the incident `alerted`. `blocked_config` and drift
  alerts use their own alert-once markers instead. This covers both the normal
  path and the exception path in `run_one_job`.

### `slack_post_thread`

`thread_ts` is now optional: omitting it posts a new root message. The result
carries `message_ts` and a `permalink` (`chat.getPermalink`, best-effort).
Before this change, the only cron-safe Slack poster could not create the root
blocker posts the workflow needed, or return the receipt it wanted to link
from Jira.

### No shell for cron agents; `post_script`

`_resolve_cron_disabled_toolsets` adds `terminal` and `code_execution` unless
`cron.allow_agent_shell: true` is set. The subtraction is authoritative: it
removes those tools even from composite toolsets. A 14-day scan of cron
sessions on the VM found two jobs whose agents called `terminal` (MRR and the
weekly activity report). Both used it only to run a deterministic script on
the agent's output.

`post_script` is that step as a first-class hook. It runs after a successful
agent run, and on `[SILENT]` runs too, because queued work may exist. It gets
the response through `$HERMES_CRON_RESPONSE_FILE`, runs in the job's
`workdir`, and appends its stdout to the delivery. A non-zero exit fails the
run. It uses the same path containment and interpreter as `script`. The
create gate (`_rbac_creation_error`) treats it like `script`, so only a role
with `terminal`/`code_execution`, i.e. a super admin, can attach one. It is
rejected on `no_agent` jobs.

### Run-time checks now include listed toolsets

`include_enabled=True` at run time. Before enabling this, it was evaluated
against every enabled job on the VM: it flagged exactly the two broken jobs.

### Script-import preflight

`script_import_problems` parses the script's AST and collects unconditional
top-level imports. Imports inside `try` blocks count as optional, and
relative and sibling-module imports are skipped. Each remaining module is
checked with `importlib.util.find_spec` in the gateway interpreter. The check
applies at `cronjob` create/update, which rejects, and at the very start of
`run_job`, before any script executes. At run time the job is blocked with a
single alert that includes the exact `uv pip install --python <interpreter>`
command.

### Transfer gate

`ownership transfer` of a `cron:` item evaluates the job under the new
owner's grant *before* transferring. It refuses when the new role would strip
something the job needs, unless the caller passes
`confirm_capability_loss=true`.

## Not done

- **Inferring requirements from prompt prose.** This is unreliable. Instead,
  the `required_toolsets` schema text and the `scheduled-automation-design`
  skill (host-side) ask the author to map each action to a toolset.
- **Auto-resuming or auto-widening a job when ownership moves to a broader
  role.** Transfer reports; a human decides.
