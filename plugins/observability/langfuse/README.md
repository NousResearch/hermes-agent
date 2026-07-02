# Langfuse Observability Plugin

This plugin ships bundled with Hermes but is **opt-in** — it only loads when
you explicitly enable it.

## Enable

Pick one:

```bash
# Interactive: walks you through credentials + SDK install + enable
hermes tools  # → Langfuse Observability

# Manual
pip install 'langfuse>=3.0'
hermes plugins enable observability/langfuse
```

**The `>=3.0` pin is mandatory, not a suggestion.** This plugin uses the
OTEL-based v3+ Python SDK (`start_observation`, `create_trace_id`). If an
unpinned `pip install langfuse` (or an environment rebuild) resolves to a
2.x SDK, the plugin's import fails silently — caught by a fail-open
try/except — and every hook becomes a permanent no-op. No error, no log
line, no trace, ever. See Troubleshooting below.

**If you self-host Langfuse, the server must also be v3+.** SDK v3+ ships
traces via OTLP (`/api/public/otel/v1/traces`), which does not exist on
Langfuse server 2.x — every export 404s regardless of how correctly the
SDK is configured. `docker-compose.selfhost-v3-example.yml` in this
directory is a known-working v3 stack (web + worker + Postgres +
ClickHouse + Redis + MinIO).

## Required credentials

Set these in `~/.hermes/.env` (or via `hermes tools`):

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK or credentials the hooks no-op silently — the plugin fails
open.

## Verify

```bash
hermes plugins list                 # observability/langfuse should show "enabled"
hermes chat -q "hello"              # then check Langfuse for a "Hermes turn" trace
```

## Optional tuning

```bash
HERMES_LANGFUSE_ENV=production       # environment tag
HERMES_LANGFUSE_RELEASE=v1.0.0       # release tag
HERMES_LANGFUSE_SAMPLE_RATE=0.5      # sample 50% of traces
HERMES_LANGFUSE_MAX_CHARS=12000      # max chars per field (default: 12000)
HERMES_LANGFUSE_DEBUG=true           # verbose plugin logging
```

## Disable

```bash
hermes plugins disable observability/langfuse
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| No traces ever appear, no errors anywhere | `langfuse` SDK is <3.0 (silent import failure, fail-open by design) | `pip install --upgrade 'langfuse>=3.0'`, restart the gateway |
| `agent.log` shows `ValueError: <Token ...> was created in a different Context` | Older versions of this plugin entered `start_as_current_observation()`'s context manager in one hook call and ended the span in another — Hermes hooks fire pre/post a turn on different worker threads, and OTEL's contextvars Token can't be popped cross-thread. Fixed in `_start_root_trace` by switching to the detached `client.start_observation()` (no "current span" contextvar involved). If you see this, your plugin copy predates that fix. | Pull the latest `plugins/observability/langfuse/__init__.py` |
| Self-hosted: every trace export gets `404` in the exporter logs | Langfuse *server* is still on major version 2; SDK v3+ only speaks OTLP, which 2.x servers don't implement | Upgrade the server to `langfuse/langfuse:3` — see `docker-compose.selfhost-v3-example.yml` |
| Self-hosted: `langfuse-web`/`langfuse-worker` crash-loop with `CLICKHOUSE_URL is not configured` | v3 requires ClickHouse (+ Redis + S3-compatible storage); v2's Postgres-only compose isn't enough | Add `clickhouse`, `redis`, `minio` services per the example compose file |
| Self-hosted: ClickHouse migration fails with `There is no Zookeeper configuration ... ReplicatedMergeTree` | The stock `clickhouse-server` image ships a demo `default` cluster; Langfuse's migrator tries `ON CLUSTER default` against it, which needs Zookeeper | Set `CLICKHOUSE_CLUSTER_ENABLED: "false"` explicitly on `langfuse-web`/`langfuse-worker` |
| Self-hosted: API calls return `Invalid authorization header. Confirm that you've configured the correct host.` after a v2→v3 upgrade | v3 added an organization layer above projects; API keys created under v2 have `organization_id = NULL` in Postgres and fail auth with this misleading message | Simplest fix: generate a fresh key pair for the project rather than repairing the old one (see script below) |

### Recreating an API key by hand (self-hosted, post v2→v3 migration)

If old keys are stuck with a null `organization_id`, generate a new pair
with correctly computed hashes and insert it directly rather than trying
to repair the old row (an in-place `UPDATE ... organization_id` backfill
was observed to be immediately followed by the row disappearing — likely
an app-side cleanup of org-less keys, not confirmed as reproducible):

```python
import uuid, hashlib, bcrypt

SALT = "<your SALT env var value>"
pk = f"pk-lf-{uuid.uuid4()}"
sk = f"sk-lf-{uuid.uuid4()}"
hashed_secret_key = bcrypt.hashpw(sk.encode(), bcrypt.gensalt(11)).decode()

def fast_hash(secret: str, salt: str) -> str:
    salt_hash = hashlib.sha256(salt.encode()).hexdigest()
    return hashlib.sha256((secret + salt_hash).encode()).hexdigest()

fast_hashed_secret_key = fast_hash(sk, SALT)
display_secret_key = sk[:6] + "..." + sk[-4:]
```

Then `INSERT INTO api_keys (id, note, public_key, hashed_secret_key,
display_secret_key, project_id, organization_id, fast_hashed_secret_key,
scope) VALUES (...)` — get `project_id`/`organization_id` via `SELECT id,
org_id FROM projects WHERE name = '<your project>'`.
