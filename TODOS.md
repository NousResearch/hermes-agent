# TODOS

## DeepParser SDK + API Server

### P1 — Deferred from v1.0.0 plan

- **Add quickstart.md to deepparser/ subfolder**
  **Priority:** P1
  A condensed "5-minute quickstart" separate from the full README. Plan spec: deepparser/quickstart.md.
  Deferred from: claude/quirky-ramanujan-907688 (v1.0.0 launch)

- **Add CHANGELOG.md at v1.0.0**
  **Priority:** P1
  Repo-root CHANGELOG.md with `## [1.0.0.0] - YYYY-MM-DD` entry. RELEASE_v1.0.0.md covers GitHub release notes, but standard CHANGELOG.md is missing.
  Deferred from: claude/quirky-ramanujan-907688 (v1.0.0 launch)

- **Document semver policy in deepparser/README.md**
  **Priority:** P1
  Add a "Versioning" section: major=breaking API change, minor=new endpoint/field, patch=bug fix. Helps SDK consumers set version pins.
  Deferred from: claude/quirky-ramanujan-907688 (v1.0.0 launch)

- **Add "Next: Try With Your Own File" section in README after demo() example**
  **Priority:** P1
  DX plan called for a transition paragraph after the demo() code block pointing to basic_parse.py and the key registration endpoint.
  Deferred from: claude/quirky-ramanujan-907688 (v1.0.0 launch)

- **Add GitHub Issues support link in deepparser/README.md**
  **Priority:** P1
  Plan spec: "Support: GitHub Issues" with a clickable link. Currently missing from README.
  Deferred from: claude/quirky-ramanujan-907688 (v1.0.0 launch)

- **Daily SQLite backup via Fly.io scheduled machine**
  **Priority:** P1
  Engineering decision E10 noted backup as a follow-up. fly.toml has max_machines_running=1 for write safety but no automated backup. Add a Fly.io cron machine that copies /data/deepparser.db to S3/R2 daily.
  Deferred from: claude/quirky-ramanujan-907688 (v1.0.0 launch)

- **Test DWG hard gate before adding DWG claim to HN post**
  **Priority:** P1
  Parse one real .dwg or .dxf file against the live API and confirm the answer is meaningful before the HN headline includes the DWG/CAD claim. Command: `python deepparser/examples/dwg_query.py floor_plan.dwg "List all rooms"`.
  Deferred from: claude/quirky-ramanujan-907688 (v1.0.0 launch)

## Completed

<!-- Items completed in merged PRs move here -->
