# Production Hermes Agent on Angie

This runbook is the repo-owned implementation companion for making the production Hermes Agent distribution on `angie@10.10.70.17` usable. It is intentionally conservative: repo work can be completed locally, but production reads and all production mutations require the approval boundaries below.

## Target home and profile strategy

- Production Hermes home: `/home/angie/.hermes`.
- Initial migration target: the existing production home, not a new Hermes profile.
- No new profile is created unless Joe explicitly approves a profile migration plan.
- Keep production runtime state, secrets, backups, generated configs, logs, and evidence artifacts out of Git.

## Go-live modes

### Mode A — Slack production agent, minimum usable target

Required:

- Slack gateway works from the Angie host.
- The primary model/provider used for Slack responses has production-owned credentials and passes a non-sensitive smoke test.
- Required plugins are migrated or explicitly disabled with Joe approval.
- Required MCP servers are smoke-tested or explicitly disabled with Joe approval.
- Memory may be disabled/fail-open; Slack responses must still work.
- Dashboard is deferred.

### Mode B — Slack + Hindsight memory

Mode A plus:

- Hindsight DB/API are healthy on localhost only.
- Non-sensitive retain/recall smoke passes.
- Hindsight backup, restore, and fallback are documented and rehearsed.

### Mode C — Slack + Hindsight + local dashboard

Mode B plus:

- Dashboard is healthy and bound to localhost only.
- Public exposure, reverse proxying, or tunneling is out of scope unless separately approved.

## Hard blockers vs deferrable readiness

Hard blockers are not deferrable before production writes, restarts, service starts, or cutover:

- Explicit production mutation approval exists for the exact action and scope.
- No-secret artifact policy is active for Notion, GitHub, Slack, logs, diffs, doctor output, screenshots, terminal excerpts, and generated evidence.
- Secret-bearing backup exists outside Git/web/shared paths with owner-only permissions and expiry/delete date.
- Restore rehearsal completed into a temporary path, not over production.
- Rollback command sheet exists for config, `.env`, credential files, plugins, systemd, Hindsight, dashboard, gateway, and Angie repo commit.
- `angie doctor hermes --hermes-home /home/angie/.hermes --json` exists and runs.
- Production `.env` key-name inventory exists; values are never printed.
- Full sanitized Hermes config diff and Slack config classification exist.
- Plugin migration inventory exists.
- Hindsight fallback exists if memory breaks runtime.

Deferrable only with explicit Joe approval, owner, date, and risk:

- Hindsight memory, reducing go-live to Mode A.
- Dashboard, reducing go-live to Slack-only.
- Non-required MCPs.
- Non-required plugins.

## Production mutation approval boundary

The `angie` helpers are read-only by default. They must not write production files, restart services, start containers, change config, mutate Docker state, or query/call AWS/Greengrass directly without explicit same-thread approval of the exact action and scope.

Before any mutation, record:

1. action and target paths/services;
2. expected impact;
3. risk;
4. rollback or stop point;
5. why read-only diagnosis is insufficient.

## tbot / AWS / Greengrass out of scope

Hermes Agent should not hold AWS credentials, AWS SDK permissions, or AWS IAM permissions and should not directly query or call AWS IoT, Greengrass, ProofEvidence S3, or other AWS services. Production Teleport/SSH inspection, even read-only, requires explicit prior confirmation from Joe.

The production doctor includes negative verification for accidental AWS, Teleport, or tbot exposure by reporting key names and path presence only. It must not print credential values.

## No-secret artifact policy

Allowed in artifacts:

- path names;
- file existence;
- owner UID/GID;
- mode bits;
- key names;
- redacted config summaries;
- command exit codes;
- non-sensitive health state.

Forbidden in artifacts:

- `.env` values;
- Slack tokens (`xoxb-`, `xapp-`, etc.);
- bearer tokens;
- provider API keys;
- cookies;
- private keys;
- DB URLs/passwords;
- AWS access keys or session tokens;
- OAuth/Codex auth file contents.

## Read-only doctor

Run locally on the production host only after read-only production access is approved:

```bash
angie doctor hermes --hermes-home /home/angie/.hermes
angie doctor hermes --hermes-home /home/angie/.hermes --json
```

Exit codes:

- `0`: no hard blockers; warnings may remain.
- `1`: one or more hard blockers.
- `2`: invalid arguments or unreadable input path.

The JSON output shape is:

```json
{
  "status": "pass|warning|blocker",
  "mode_readiness": {
    "mode_a": "pass|blocker|warning",
    "mode_b": "pass|blocker|warning",
    "mode_c": "pass|blocker|warning"
  },
  "checks": [
    {
      "id": "string",
      "severity": "info|warning|blocker",
      "status": "pass|fail|skipped",
      "evidence": "redacted string",
      "owner": "optional",
      "rollback": "optional"
    }
  ],
  "redactions_applied": true
}
```

The doctor is read-only. It does not write diff artifacts, restart services, alter config, or start containers.

## Plugin migration contract

Phase 1 exposes a dry-run inventory command only:

```bash
angie hermes plugins sync \
  --hermes-home /home/angie/.hermes \
  --source plugins/hermes \
  --dry-run \
  --json
```

The inventory schema includes:

- plugin name;
- source path;
- target path under `/home/angie/.hermes/plugins/<plugin-name>`;
- decision (`sync_candidate` or `exclude`);
- reason;
- environment key names discovered from source;
- enabled state placeholder until production config audit.

`interactive-cli` is excluded from production migration. Applying plugin sync is intentionally not implemented in Phase 1; any production copy requires explicit approval after inventory review.

## Hindsight production template

The repo includes a production-shaped template at `deployments/hindsight/`:

- `compose.yaml` with DB and API services;
- localhost-only port binding;
- `.env.example` with placeholders only;
- README with start, health, smoke, stop, fallback, backup, and restore notes.

Before Mode B/C, verify the installed Hermes Hindsight provider contract from code/tests: config path, `memory.provider` value, `HINDSIGHT_API_URL`, health/version endpoint, and retain/recall smoke command. If not verified, Mode B/C remains blocked and Mode A must disable/fail-open memory.

## Backup / restore / rollback skeleton

Backups must be outside Git/web/shared paths, owner-only, and must not be attached to Notion/GitHub/Slack. The backup and restore layouts must match.

```bash
set -euo pipefail
TS=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP=/home/angie/hermes-backups/prod-hermes-$TS
umask 077
mkdir -p "$BACKUP/static" "$BACKUP/dynamic" "$BACKUP/meta"
chmod 700 "$BACKUP"
MANIFEST="$BACKUP/backup-manifest.tsv"
: > "$MANIFEST"

record() {
  status="$1"; original="$2"; backup="$3"; note="${4:-}"
  printf '%s\t%s\t%s\t%s\n' "$status" "$original" "$backup" "$note" >> "$MANIFEST"
}

copy_if_exists() {
  src="$1"; dest="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dest")"
    cp -a "$src" "$dest"
    record copied "$src" "$dest"
  else
    record missing "$src" "" "not present"
  fi
}

copy_if_exists /home/angie/.hermes/config.yaml "$BACKUP/static/config.yaml"
copy_if_exists /home/angie/.hermes/.env "$BACKUP/static/.env"
copy_if_exists /home/angie/.hermes/plugins "$BACKUP/static/plugins"
copy_if_exists /home/angie/.config/systemd/user/hermes-gateway.service "$BACKUP/static/hermes-gateway.service"

find "$BACKUP" -type f -exec chmod go-rwx {} +
find "$BACKUP" -type d -exec chmod go-rwx {} +
```

Restore rehearsal must copy the backup into a temp path first:

```bash
set -euo pipefail
RESTORE_TEST=/home/angie/hermes-restore-tests/restore-test-$TS
umask 077
mkdir -p "$RESTORE_TEST"
chmod 700 "$RESTORE_TEST"
cp -a "$BACKUP"/. "$RESTORE_TEST"/
python - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["RESTORE_TEST"])
required = [root / "backup-manifest.tsv", root / "static"]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"restore rehearsal missing: {missing}")
print("restore rehearsal layout ok")
PY
```

Rollback command sheet must include restoring config, `.env`, auth/Codex/provider credential files if touched, plugins, systemd units/drop-ins, Hindsight/db/api, dashboard, gateway, and the Angie repo commit.

## Smoke tests and observation

Minimum go-live smoke tests:

- controlled Slack test channel approved;
- mention and free-response behavior match config;
- channel prompt and skill binding load as expected;
- allowed-user/channel rules work;
- one response only, no bot loop;
- primary model/provider non-sensitive prompt succeeds;
- optional Codex/OAuth/tool features either pass or are explicitly disabled/deferred;
- migrated plugins are active or disabled with reason;
- Hindsight retain/recall passes for Mode B/C or memory is disabled for Mode A;
- required MCPs return expected read-only result;
- logs contain no secret values.

Observation commands/patterns:

```bash
journalctl --user -u hermes-gateway.service -f
journalctl --user -u hermes-gateway.service -b --no-pager -n 200
```

Watch for gateway reconnect loops, `msg_too_long`, MCP auth errors, plugin exceptions, Hindsight API errors, provider/Codex auth errors, secret-looking values in logs, CPU/RSS/disk regressions, and unexpected Docker disk growth.
