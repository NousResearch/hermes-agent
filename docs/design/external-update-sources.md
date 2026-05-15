# External update sources design

Hermes core updates are handled by `hermes update`. External components — user-installed plugins, project plugins, external dashboard builds, MCP helper repos, and similar git-backed extensions — need a separate trust boundary.

## Goals

- Inventory external git-backed components in one central report.
- Fetch metadata before mutating installed code.
- Fail closed on dirty or diverged external repos.
- Run a conservative static audit gate before update.
- Make mutation opt-in; normal `hermes update` must not pull external code.
- Write machine-readable results for agents and operators.

## Proposed command surface

```bash
hermes update-sources check [--json]
hermes update-sources apply [--source NAME] [--yes]
```

`check` is the default-safe operation. It may run `git fetch` for source metadata but must not change working trees.

`apply` updates only sources that are trusted, clean, fast-forwardable, and audit-passing.

## Source discovery

Initial sources:

- `~/.hermes/plugins/*` when the entry is a git checkout.
- dashboard plugins when a plugin has `dashboard/manifest.json`.
- git-backed `HERMES_WEB_DIST`.
- project plugins only when an explicit `--project PATH` or config entry names the project; do not rely on arbitrary `Path.cwd()` from a global update.

Each source record should include:

- name, kind, path
- remote URL and tracking ref
- current HEAD and remote HEAD
- dirty status
- commit count available
- changed files
- audit findings
- final status

## Safety rules

Block apply when:

- no configured upstream/tracking ref
- working tree is dirty, including untracked files
- local and remote have diverged
- audit cannot inspect the diff
- audit finds critical patterns
- source is not explicitly trusted

Critical patterns:

- `curl`/`wget` piped to shell
- private key material
- likely literal API keys/tokens/passwords
- SUID/SGID chmod additions
- encoded script execution

Review-level patterns such as `eval`, broad `exec`, `subprocess`, and `shell=True` should be reported and block non-interactive apply unless explicitly approved.

## Report schema sketch

```json
{
  "schema_version": 1,
  "generated_at": "2026-05-15T00:00:00Z",
  "mode": "check",
  "sources": [
    {
      "name": "example",
      "kind": "plugin",
      "path": "/home/user/.hermes/plugins/example",
      "remote_url": "https://github.com/example/plugin.git",
      "tracking_ref": "origin/main",
      "head": "abc1234",
      "remote_head": "def5678",
      "dirty": false,
      "commits_available": 2,
      "changed_files": ["plugin.py"],
      "audit": {"passed": true, "findings": []},
      "status": "audit_passed"
    }
  ],
  "summary": {"total": 1, "updated": 0, "blocked": 0, "available": 1}
}
```

Write the latest report to:

`~/.hermes/update-sources/last-run.json`

## Test requirements

- discovers user plugins and external web dist
- project plugin discovery requires explicit project path/config
- `check` never pulls
- `apply` pulls only after audit passes
- dirty source blocks apply
- diverged source blocks apply
- no upstream blocks apply
- critical audit finding blocks apply
- report is written for success and failure
