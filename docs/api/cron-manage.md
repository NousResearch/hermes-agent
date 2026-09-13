# `cron.manage` (describe / update / history) and cron source files

The gateway's `cron.manage` method is the *person's* door to cron — Portal and
the TUI call it when someone clicks. The model reaches the same code through
its `cronjob` tool. Until now the method only offered `list`, `add`, `pause`,
`resume` and `remove`, and `list` caps every prompt at a 100-character
`prompt_preview` for the model's benefit. A person expanding a job card needs
the whole prompt, needs to save an edit to it, and wants the execution ledger
beside it — three round-trips `list` cannot answer, so a client that asked got
`unknown cron action` (4016) and had to show "may be truncated" forever.

## Actions

| Action | Params | Returns |
|--------|--------|---------|
| `describe` | `name` (job id, or a unique name) | `{success, job}` where `job` is the `list` shape plus the full `prompt`, the raw `inputs` / `outputs` / `side_effects` / `source_files` / `context_from` lists, and `source_files_resolved` (see below). **4404** unknown job, **4001** missing name |
| `update` | `name` (job id) + any of `prompt`, `job_name`, `schedule`, `deliver`, `repeat`, `skills`, `script`, `monitor_script`, `monitor_url`, `context_from`, `workdir`, `enabled_toolsets`, `inputs`, `outputs`, `side_effects`, `source_files` | The tool's `{success, job}` envelope. **4017** when the tool refuses the edit (its message is passed through), **4001** when no field was given |
| `history` | `name`, `limit?` (1–500, default 50) | `{success, job_id, job_name, count, runs[]}` — the execution ledger newest first (`status`, `claimed_at`, `started_at`, `finished_at`, `error`, …) |

`name` is already the job **identifier** on this method, so a rename travels
as `job_name`. Sending the new name as `name` addresses a job that doesn't
exist; sending the id as `job_name` renames the job to its own id. Neither
mistake fails loudly, which is why the asymmetry is spelled out here.

Every write is attributed to the `actor` param (default `human`) through
`cron.changesets.use_changeset_origin`, same as before.

## Source files — the code behind a job

A job's dataflow (`inputs` / `outputs` / `side_effects`) says what it reads
and writes. Its **source files** say which code does the reading and writing.

Two are known mechanically: `script` (for a `no_agent` job this *is* the job)
and `monitor_script`. The rest is agentic — a prompt that says "run
`~/.hermes/scripts/ingest.py`" names code the mechanism can't see — so the
creating agent declares it in a new job field:

```
source_files: ["ingest.py", "~/.hermes/hermes-agent/indexing/x402_snapshot.py"]
```

- Entries are **paths, not typed refs**: absolute, `~`-relative, or relative
  (relative resolves under `HERMES_HOME/scripts/`, mirroring `script`). A
  leading `file:` is tolerated and stripped; URLs are rejected with a pointer
  to `inputs`.
- Only shape is enforced at write time. Existence is *reported*, never
  required — a job is routinely declared before its script is committed.
- Available on `cronjob(action=create|update, source_files=[...])`, the tool
  schema (`CRONJOB_SCHEMA`), `cron.manage update`, and backfilled to `[]` on
  legacy records by `_normalize_job_record`.

### On the graph

`cron.graph` puts the merged list on every `cron` node as `source_files`:

```jsonc
{
  "id": "3f9a…", "kind": "cron", "label": "indexing/x402",
  "source_files": [
    {"path": "/Users/me/.hermes/scripts/w.sh", "declared": "w.sh", "role": "script",
     "root": "hermes", "rel": "scripts/w.sh", "exists": true},
    {"path": "/Users/me/.hermes/hermes-agent/indexing/x402.py", "declared": "~/.hermes/hermes-agent/indexing/x402.py",
     "role": "declared", "root": "repo", "rel": "indexing/x402.py", "exists": true},
    {"path": "/opt/elsewhere/gone.py", "declared": "/opt/elsewhere/gone.py",
     "role": "declared", "root": null, "rel": null, "exists": false}
  ]
}
```

- `role`: `script` → `monitor` → `declared`, mechanical first. Duplicates
  collapse on the resolved path with the mechanical role winning.
- `root` + `rel` address the file for `files.read` (see `files-browse.md`)
  when it lives under a browsable root. When roots nest — the repo checkout
  often lives inside `~/.hermes` — the **deepest** containing root wins, so a
  repo file is `repo:indexing/x.py` rather than `hermes:hermes-agent/indexing/x.py`.
  Both `null` means the file is listed but not openable from a client.
- `exists` is this host's view at graph-build time.

Source files are **node metadata, not nodes**: a script is what a job is *made
of*, not something it exchanges data with, so drawing it as a resource would
clutter the dataflow with edges that carry no data.

They are also deliberately **outside the graph commitment**. `cron/changesets.py`
and Portal's `CronGraphDigest` hash the node row independently and must agree
byte-for-byte; growing that row is a coordinated change on both sides, not a
side effect of adding metadata. Declaring code is therefore metadata on a
revision, not a revision — `test_source_files_stay_out_of_the_commitment` pins
this.

Builder: `cron.jobs.job_source_files(job)`; roots from
`tui_gateway.files_browse.file_roots()` with a data-home fallback when that
module isn't importable.
