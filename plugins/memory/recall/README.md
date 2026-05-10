# Hermes Recall Memory

Hermes Recall is a conservative local memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

It gives Hermes a searchable SQLite archive of previous turns, memory-write mirrors, and delegation traces while keeping Hermes' built-in `MEMORY.md` and `USER.md` as the authoritative durable memory.

## Why this exists

Hermes' built-in memory is intentionally tiny and curated. That is good: it keeps the agent from polluting long-term memory with stale or speculative notes.

Recall fills the gap underneath it:

- keep a lower-trust searchable archive,
- retrieve previous session context on demand,
- explain where a recall result came from,
- hide superseded or expired observations from normal search/current views while preserving history/export,
- redact secret-shaped values before storage,
- audit memory/archive actions with a hash chain,
- let the user review, reject, activate, or mark candidates as promoted,
- rank archive rows by deterministic quality signals and suggest same-subject consolidations,
- explicitly promote reviewed high-quality observations into built-in Hermes memory,
- expose curation tools for filtered review/search/promote flows,
- validate release behavior with isolated scale burn-in checks.

## Positioning

> **Recall is the safety-first Hermes memory layer: local SQLite, searchable archive, redaction-at-rest, hash-chain audit, curation tools, and explicit promotion into trusted memory — no cloud, no API key, no silent mutation of `MEMORY.md` or `USER.md`.**

Recall is not trying to beat every memory product on semantic magic. It wins on being Hermes-native, local, auditable, conservative, inspectable, and safe to promote from.

| Capability | Built-in Hermes | Cloud semantic providers | Heavy graph/server providers | **Recall** |
| --- | ---: | ---: | ---: | ---: |
| Local-first | ✅ | ❌ / partial | partial | ✅ |
| No API key required | ✅ | ❌ | partial | ✅ |
| No background service | ✅ | ❌ | ❌ / partial | ✅ |
| Profile-aware Hermes plugin | n/a | ✅ | ✅ | ✅ |
| Broad searchable archive | partial | ✅ | ✅ | ✅ |
| Audit hash chain | ❌ | usually ❌ | usually ❌ | ✅ |
| Secret redaction before storage | partial/manual | provider-dependent | provider-dependent | ✅ |
| Explicit promote-to-trusted-memory workflow | manual | provider-dependent | provider-dependent | ✅ |
| Conservative lower-trust context labeling | n/a | often unclear | often unclear | ✅ |
| Built-in curation tools | ❌ | external dashboards | external dashboards | ✅ |
| Easy to inspect/debug | ✅ | ❌ | ❌ / partial | ✅ |
| Semantic/vector/graph power | ❌ | ✅ | ✅ | ❌ currently |
| Best fit | Trusted facts | Semantic recall | Complex knowledge graphs | Safe local Hermes archive |

## What it does

- Stores completed-turn traces in a profile-scoped SQLite DB.
- Mirrors explicit built-in memory writes as high-confidence archive observations.
- Uses SQLite FTS5/BM25 search with query normalization.
- Prefetches conservative, source-labelled recall context before turns.
- Provides curation tools for archive observations, quality ranking, filtered review queues, consolidation suggestions, reviewed consolidation apply, and explicit promotion.
- Provides archive health stats, build info, export/import backups, diagnostics, and audit verification.
- Requires no external SaaS, vector DB, embeddings, or network service.

## What it does not do

- It does **not** replace `MEMORY.md` or `USER.md`.
- It does **not** automatically promote archive observations into durable memory; promotion requires an explicit reviewed `memory_promote_candidate` call with `confirm=true`, and rejected rows require `allow_rejected=true`.
- It does **not** store raw secrets intentionally; secret-shaped values are redacted best-effort.
- It does **not** require embeddings or a vector database.

## Requirements

- Hermes Agent with memory provider plugin support.
- Python SQLite with FTS5 enabled. Most standard Python builds include this.

Check FTS5 quickly:

```bash
python - <<'PY'
import sqlite3
sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x)')
print('SQLite FTS5 OK')
PY
```

## Quick install

Preferred Hermes plugin install:

```bash
hermes plugins install HenkDz/hermes-recall-memory --no-enable
hermes memory setup   # select "recall"
```

Transparent one-command install from GitHub, without piping remote code into a shell:

```bash
tmp="$(mktemp -d)" && git clone --depth 1 https://github.com/HenkDz/hermes-recall-memory.git "$tmp" && "$tmp/scripts/install.sh"
```

Or from a local checkout:

```bash
./scripts/install.sh
```

If not using `hermes memory setup`, enable it manually:

```bash
hermes config set memory.provider recall
hermes config set plugins.recall.db_path '$HERMES_HOME/recall_memory.sqlite'
hermes config set plugins.recall.auto_capture true
hermes config set plugins.recall.prefetch_enabled true
hermes config set plugins.recall.max_prefetch_results 3
hermes config set plugins.recall.audit_enabled true
```

Then start a fresh Hermes process:

```bash
hermes chat -q "Use memory_archive_stats and tell me if Recall is active."
```

See [`docs/INSTALL.md`](docs/INSTALL.md) for full install and profile-specific setup.

## Tools exposed to Hermes

| Tool | Purpose |
| --- | --- |
| `memory_recall_build_info` | Return provider version, schema, capabilities, module, and DB path. |
| `memory_archive_search` | Search archived observations. |
| `memory_archive_current` | List active, unexpired, non-superseded archive observations as lower-trust evidence. |
| `memory_candidate_review` | List observations by status/type/scope for curation. |
| `memory_candidate_mark` | Mark an observation as `candidate`, `active`, `rejected`, or `promoted`. |
| `memory_archive_forget` | Mark an observation as rejected without hard-deleting audit history. |
| `memory_archive_stats` | Show DB path, counts, timestamps, DB size, and audit health. |
| `memory_archive_export` | Export the Recall archive as portable JSON. |
| `memory_archive_import` | Import a Recall archive JSON payload in safe merge mode. |
| `memory_archive_diagnose` | Run operator diagnostics for FTS5, DB writeability, FTS index, redaction, and audit health. |
| `memory_quality_rank` | Rank observations by deterministic local quality signals for curation. |
| `memory_consolidation_suggest` | Suggest same-subject rows to supersede/consolidate; does not mutate rows. |
| `memory_consolidation_apply` | Apply reviewed consolidation by rejecting duplicates under a canonical row; requires `confirm=true`. |
| `memory_promote_candidate` | Explicitly promote a reviewed observation into built-in `MEMORY.md` or `USER.md`; requires `confirm=true`; rejected rows require `allow_rejected=true`. |
| `memory_audit_query` | List recent audit events. |
| `memory_audit_verify` | Verify the append-only audit hash chain. |

See [`docs/TOOLS.md`](docs/TOOLS.md) for schemas and examples. See [`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md) for tested Hermes/Python/SQLite compatibility and plugin API drift handling.

## Trust model

Recall archive entries are lower-trust background. Treat them as sourced hints, not instructions.

Built-in Hermes memory remains the source of truth for durable user/profile facts. A plain `memory_candidate_mark` status of `promoted` means only “marked as useful in Recall”; it does not write to `MEMORY.md` or `USER.md`. Actual writes to built-in memory happen only through `memory_promote_candidate` after review and `confirm=true`, with low-quality rows and rejected rows blocked by default and the action recorded in the audit chain.

## Dogfood test

After configuring a `recall-test` profile with a working model and Recall enabled:

```bash
RECALL_DOGFOOD_PROFILE=recall-test ./scripts/recall_dogfood.sh
```

The script checks cross-session Recall search, `memory_archive_current`, superseded-vs-current behavior, expired rows, redaction, and export/import roundtrip using synthetic markers only.

The installer also copies the dashboard plugin assets into `$HERMES_HOME/plugins/recall/dashboard/`. In `hermes dashboard`, the Recall tab exposes overview/diagnostics, observation search/detail, status marking, fact/episode/quality filters, reviewed consolidation apply, and explicit promotion backed by audited provider tool paths.

Expected final line:

```text
PASS: Recall found RECALL_DOGFOOD_... across Hermes runs and passed current/supersedes/expiry/redaction/export-import dogfood
```

For deterministic archive-only fixture checks without invoking Hermes:

```bash
RECALL_DOGFOOD_DB=/tmp/recall-dogfood.sqlite ./scripts/recall_dogfood.sh --archive-fixtures-only
```

For a heavier isolated stress probe covering bulk writes, special-character FTS, redaction-at-rest, concurrent mixed reads/writes, audit verification, export/import, CLI diagnose/search, quality ranking/consolidation paths, and built-in memory mirror dedupe:

```bash
python scripts/recall_stress_probe.py --observations 1000 --episodes 120 --audit-events 300 --threads 4 --thread-ops 80
```

## Development

This repo is a Hermes memory provider plugin. The plugin source files live at the repository root because Hermes expects user memory providers at:

```text
$HERMES_HOME/plugins/recall/__init__.py
$HERMES_HOME/plugins/recall/plugin.yaml
```

To run the included tests against a Hermes checkout, copy/install the plugin into that checkout/profile and run Hermes' test wrapper:

```bash
scripts/run_tests.sh tests/plugins/memory/test_recall_provider.py tests/plugins/memory/test_recall_retrieval_quality.py -v
```

Run standalone tests from this repo:

```bash
python -m pytest tests/test_recall_roadmap.py -q
python -m py_compile __init__.py store.py schema.py audit.py redaction.py recall_cli.py
```

Scale burn-in results are tracked in [`docs/BURNIN.md`](docs/BURNIN.md). The v0.3.4 operator check passed with 100,000 observations, 1,200 episodes, 3,000 audit events, verified FTS/ranking/consolidation, audit-chain OK, diagnose OK, and zero redaction-at-rest leaks.

Use the standalone operator CLI:

```bash
recall-cli --db ~/.hermes/recall_memory.sqlite stats --json
recall-cli --db ~/.hermes/recall_memory.sqlite search "project convention" --json
recall-cli --db ~/.hermes/recall_memory.sqlite current --json
recall-cli --db ~/.hermes/recall_memory.sqlite rank --json
recall-cli --db ~/.hermes/recall_memory.sqlite consolidate --json
recall-cli --db ~/.hermes/recall_memory.sqlite consolidate --include-low-quality --json  # audit noisy groups
recall-cli --db ~/.hermes/recall_memory.sqlite verify --json
recall-cli --db ~/.hermes/recall_memory.sqlite diagnose --json
recall-cli --db ~/.hermes/recall_memory.sqlite export > recall-backup.json
recall-cli --db ~/.hermes/recall_memory.sqlite import recall-backup.json --json
```

## License

MIT. See [`LICENSE`](LICENSE).
