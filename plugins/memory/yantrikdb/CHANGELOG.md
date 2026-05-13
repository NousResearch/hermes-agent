# Changelog

All notable changes to the YantrikDB Hermes memory plugin.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project aims for semantic versioning once merged into Hermes.

## [0.4.2] — 2026-05-12 — First-class embedder loaders

v0.4.1 shipped the config surface for swapping embedders, but the only embedder-class path required users to write a thin wrapper around `model2vec` or `sentence-transformers`. That's friction the plugin should absorb — most users asking about multilingual want `potion-multilingual-128M` (a model2vec model) or one of the well-known HF sentence-transformers, both of which are one-liners to load.

v0.4.2 adds two first-class loaders so you can point at any Hugging Face model id directly, with no wrapper class to write and no `YANTRIKDB_EMBEDDING_DIM` to set (auto-probed).

### Added

- **`YANTRIKDB_EMBEDDER_MODEL2VEC`** — Hugging Face model id for the built-in `Model2VecEmbedder` loader (wraps `model2vec.StaticModel.from_pretrained`). Lightweight static-embedding family — no PyTorch dependency. Install with `pip install 'yantrikdb-hermes-plugin[model2vec]'`. Example: `YANTRIKDB_EMBEDDER_MODEL2VEC=minishlab/potion-multilingual-128M`.
- **`YANTRIKDB_EMBEDDER_HF`** — Hugging Face model id for the built-in `SentenceTransformerEmbedder` loader (wraps `sentence_transformers.SentenceTransformer`). Covers the broader HF embedder ecosystem; pulls in PyTorch. Install with `pip install 'yantrikdb-hermes-plugin[sentence-transformers]'`. Example: `YANTRIKDB_EMBEDDER_HF=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- **Auto-probed dim** for the two new loaders — the plugin calls `.encode("__yantrikdb_probe__")` once during init, uses `len()` of the result as the dim, and passes that into `YantrikDB(db_path, embedding_dim=N)`. No `YANTRIKDB_EMBEDDING_DIM` env var needed for these paths.
- **Two new pip extras**: `[model2vec]` and `[sentence-transformers]`. The default install stays slim; only users who pick one of the new paths pull the heavy dep.
- **10 new tests** in `tests/test_embedded.py` covering the new loaders, auto-probe, missing-dependency actionable errors, and the extended precedence rules (151 tests total, all green).

### Behavior changes

- Path precedence is now: **CLASS > MODEL2VEC > HF > EMBEDDER (bundled-named) > default**. More-specific user intent wins: `_CLASS` is the most specific (exact Python class), the built-in loaders pick an exact HF model, `_EMBEDDER` depends on which named variants the engine version ships, and default is the fallback.
- The `[model2vec]` and `[sentence-transformers]` extras can be installed together if you want to A/B different embedders without uninstalling.
- The error message when `model2vec` or `sentence-transformers` is missing is now actionable — it points at the right pip extra by name.

### Migration

None required. With no embedder env vars set, the plugin behaves identically to v0.4.1 / v0.3.x.

### Net install for multilingual

```bash
pip install 'yantrikdb-hermes-plugin[model2vec]'  # v0.4.2
yantrikdb-hermes install ~/hermes-agent

cat >> ~/.hermes/.env <<EOF
YANTRIKDB_EMBEDDER_MODEL2VEC=minishlab/potion-multilingual-128M
EOF
```

That's the whole integration — no Python wrapper to write, no dim to look up.

## [0.4.1] — 2026-05-12 — Unblock v0.4.0 publish (lint)

Patch release: v0.4.0's tagged commit failed the publish workflow at the `ruff` gate (F841 — unused `client = ...` locals in three `tests/test_embedded.py` cases that assert against the mock instead of the returned client). PyPI never received v0.4.0; this is the first PyPI release of the pluggable-embedder feature.

### Fixed
- Removed unused `client = ` assignments in `tests/test_embedded.py` so `ruff check` passes under CI's stricter config. Pure test-code cleanup; no behavior change in the plugin.

### Note
Functionally identical to v0.4.0. Use this if you want pluggable embedders on PyPI.

## [0.4.0] — 2026-05-12 — Pluggable embedders

Lands the configuration surface for swapping the bundled embedder — driven by the first user inquiry on the repo ([Issue #1](https://github.com/yantrikos/yantrikdb-hermes-plugin/issues/1): multilingual embedding support). Default behavior is unchanged for existing users; the new env vars only matter if you want a non-default embedder.

### Added

- **`YANTRIKDB_EMBEDDER`** — name of a bundled-download embedder (e.g. `potion-base-8M`, `potion-base-32M`). The plugin calls `db.set_embedder_named(name)` on engine construction. Works with whatever named embedders `yantrikdb >= 0.7.6` ships behind the `embedder-download` feature flag.
- **`YANTRIKDB_EMBEDDER_CLASS`** — dotted Python import path (e.g. `myapp.embedders.MultilingualEmbedder`) to a class that has a `.encode(text) -> list[float]` method. The plugin imports the class, instantiates with no args, and calls `db.set_embedder(instance)`. Lets users plug `sentence-transformers`, `model2vec-rs`, multilingual variants, or any custom embedder *without* waiting on upstream bundling.
- **`YANTRIKDB_EMBEDDING_DIM`** — required when either `_EMBEDDER` or `_EMBEDDER_CLASS` is set; matches the output dim of the chosen embedder (256 for potion-base-8M, 512 for potion-base-32M, 384 for `all-MiniLM-L6-v2`, etc.). The plugin passes this to `YantrikDB(db_path, embedding_dim=N)`.
- **13 new tests** in `tests/test_embedded.py` pinning the embedder-path semantics: default (with_default) path, bundled-named path, custom-class path, dim-required-when-custom invariant, class-must-have-encode invariant, malformed-class-path errors, and class-over-name precedence when both are set.

### Behavior changes

- The plugin's embedder selection logic is now three paths instead of one:
  - `YANTRIKDB_EMBEDDER_CLASS` set → import + instantiate + `set_embedder(instance)`.
  - else `YANTRIKDB_EMBEDDER` set → construct with `embedding_dim=N` + `set_embedder_named(name)`.
  - else → `YantrikDB.with_default(db_path)` (existing v0.3.x behavior, dim=64 potion-2M).
- Class path takes precedence over named path when both env vars are set — it's the more specific instruction and doesn't depend on upstream bundling state.
- All three paths use `set_embedder*` exactly once, immediately after construction, before the engine is shared (Arc::get_mut requirement per the engine's threading contract).

### Migration for v0.3.x users

None required. With no embedder env vars set, the plugin behaves identically to v0.3.1.

### Net install for non-default embedders

```bash
pip install yantrikdb-hermes-plugin                  # v0.4.0
yantrikdb-hermes install ~/hermes-agent

# Tier 2 bundled (downloaded on first use):
cat >> ~/.hermes/.env <<EOF
YANTRIKDB_EMBEDDER=potion-base-8M
YANTRIKDB_EMBEDDING_DIM=256
EOF

# OR — custom Python embedder (e.g. multilingual, sentence-transformers):
cat >> ~/.hermes/.env <<EOF
YANTRIKDB_EMBEDDER_CLASS=myapp.embedders.MultilingualEmbedder
YANTRIKDB_EMBEDDING_DIM=384
EOF
```

### Cross-stack note

Upstream `yantrikos/yantrikdb` may add `potion-multilingual-128M` (101 languages) as a fourth named-download variant in a future release. Once that lands, multilingual users can drop the `_EMBEDDER_CLASS` Python wrapper and just set `YANTRIKDB_EMBEDDER=potion-multilingual-128M` — the plugin code is already ready for it.

## [0.3.1] — 2026-05-09 — PyPI distribution

Tooling-only release. Plugin behavior unchanged from v0.3.0 — same 8 default tools, same 3 opt-in skill tools, same feature flag, same 128 tests.

### Added

- **PyPI distribution via `yantrikdb-hermes-plugin`.** `pip install yantrikdb-hermes-plugin` installs the source under the importable package `yantrikdb_hermes_plugin` (avoids the namespace collision with the existing `yantrikdb` engine package on PyPI).
- **`yantrikdb-hermes` CLI** — bridges the pip → filesystem gap. Hermes loads plugins from `$HERMES_ROOT/plugins/memory/<name>/`, which pip can't write to directly. Two subcommands:
  - `yantrikdb-hermes install <hermes_root>` — copy the plugin source into the Hermes checkout's `plugins/memory/yantrikdb/`. `--force` overwrites an existing install.
  - `yantrikdb-hermes path` — print the on-disk path of the installed package (for users who'd rather symlink: `ln -s "$(yantrikdb-hermes path)" ~/hermes-agent/plugins/memory/yantrikdb`).
- **`.github/workflows/publish.yml`** — automated PyPI publishing pipeline triggered by tag pushes matching `v*`. Builds wheel + sdist after running ruff + mypy + pytest as a gate. Uses PyPI Trusted Publisher (no API token in repo secrets); one-time config on PyPI's web UI.

### Net install flow (post v0.3.1 publish)

```bash
pip install yantrikdb-hermes-plugin           # the plugin source + CLI
yantrikdb-hermes install ~/hermes-agent       # copy into plugins/memory/
hermes config set memory.provider yantrikdb
echo "YANTRIKDB_MODE=embedded" >> ~/.hermes/.env
```

`yantrikdb` (the engine, ~10 MB with bundled embedder) is pulled automatically as a dependency.

### Internal

- `yantrikdb/__init__.py` now wraps `from agent.memory_provider import MemoryProvider` and `from tools.registry import tool_error` in try/except so the package imports successfully outside a Hermes runtime (e.g. when the CLI invokes `from yantrikdb_hermes_plugin.cli import main`). Stub `MemoryProvider` and `tool_error` are used in that path; they're never the ones Hermes sees because Hermes loads the plugin via fresh filesystem import from `plugins/memory/yantrikdb/`.

## [0.3.0] — 2026-05-09 — Skill substrate + feature flag

### Added

- **Three new tools (opt-in via feature flag)**: `yantrikdb_skill_search`, `yantrikdb_skill_define`, `yantrikdb_skill_outcome`. Bridges Hermes agents to YantrikDB's `skill_substrate` namespace where agent-authored procedural skills live alongside skills written by other consumers (Lane B SDK, server handlers, WisePick). Hermes-authored entries are tagged `metadata.source=hermes` so any consumer can filter Hermes-authored skills in or out cleanly.
- **`YANTRIKDB_SKILLS_ENABLED` feature flag** — defaults **off**. When unset, the three skill schemas are hidden from `get_tool_schemas()` and any direct skill-tool call short-circuits with a clear error pointing at the env var. Pattern: simple-stays-simple, advanced-reachable. Same shape as yantrikdb-server's bundled-embedder default-on engine feature.
- Client-side schema validation reproducing yantrikdb-server's wrapper checks (skill_id regex, body length, applies_to format, skill_type enum). Embedded mode ships full validation since there's no server in front; HTTP mode validates client-side too as defense-in-depth ahead of the server's own check.
- The load-bearing `applies_to` regex (`^[a-z][a-z0-9_]*$` — no hyphens, no dots) is regression-pinned in `tests/test_provider.py::TestSkillValidation::test_applies_to_REJECTS_HYPHEN` per yantrikdb-server's explicit flag. Anyone naturally writing "applies-to"-style hyphenated tags would corrupt the substrate convention; the test prevents that drift.
- 32 new tests: 9 skill dispatch tests, 3 feature-flag tests, 20 validation tests. Total: **128 tests passing** (was 96).

### Architecture

- **Skill substrate**: namespace `skill_substrate` for skill bodies, `outcome_substrate` for append-only outcome events. `metadata.source=hermes` tags all writes by this plugin. Single shared namespace + metadata filtering rather than sub-namespace, per yantrikdb-server's recommendation: sub-namespace would force every downstream consumer to UNION across N+1 namespaces if they wanted all skills, which is the wrong default for the agentic-loop story.
- **Outcomes are append-only**, never auto-rolled-up onto the parent skill. The "did this skill work?" computation is the agent's pedagogy decision, not the substrate's. Matches the WisePick pattern.
- **Embedded-mode TOCTOU on `on_conflict=reject`**: the uniqueness check is best-effort lookup-then-write rather than transactional (single-agent embedded use is non-racy in practice). HTTP mode preserves server-enforced 409. Documented as semantic difference between modes.
- **Engine surface used**: `db.recall_text(query, top_k, namespace=...)` for skill_search (requires yantrikdb >= 0.7.7; pre-0.7.7 falls back to `db.recall(query=..., namespace=...)`). `db.record_text(body, memory_type="procedural", namespace="skill_substrate", metadata={...})` for skill_define. `db.record_text(...)` to `outcome_substrate` for skill_outcome.

### Lifecycle distinction (worth knowing)

The Hermes plugin now lives alongside Hermes' own filesystem skills (`$HERMES_HOME/skills/*.md`) without overlap:

- **Filesystem skills**: human-authored, durable, version-controlled. Canonical for skills a human wrote and committed.
- **YantrikDB skills**: agent-authored, runtime-evolving, semantic-search-queryable. Canonical for patterns the agent distilled from observed success.

Different *kinds* of canonical, not competing authorities. The model resolves by lifecycle, not by competition.

### Configuration

| Env var | Default | Description |
|---|---|---|
| `YANTRIKDB_SKILLS_ENABLED` | `false` | Set `true` / `1` / `yes` to expose the three skill tools. |

When the flag is off, plugin behavior is identical to v0.2.1 (8 tools, same mode-aware backend selection).

## [0.2.1] — 2026-05-09 — Documentation polish for HN-tier scrutiny

Text-only release. No code changes; no behavioural changes. All findings from yantrikdb-core's post-publish review pass on v0.2.0.

### Changed

- README quality claims now cite the upstream evaluation script (`yantrikos/yantrikdb/scratch/eval_potion_2m.py`) so readers can reproduce the R@5 vs MiniLM-L6-v2 numbers. The "~89% / ~92% / ~95% of MiniLM" approximations are now scoped to that specific eval rather than presented as universal.
- Latency table extended with p99 tail numbers for both backends. Added the honest note that even embedded p99 beats HTTP p50 — and that long-running soak validation is in progress upstream, not concluded.
- New "About the embedder quality claims" section explains corpus-size dependence: at 3 records all vectors look similar (top score ~0.58); at 8+ with real diversity the score range opens up (~0.84). Readers running their own evals on toy corpora won't be surprised by the score collapse.
- New "Explainability is a side effect, not a bolt-on" section pulls a verbatim quote from the live DeepSeek Hermes session showing the model parsing `why_retrieved` reason codes naturally and reflecting them in its own reasoning. Frames the explainability surface as the recall response itself rather than a separate feature.

### Internal

- v0.2.0 commit + tag remain valid; v0.2.1 is the recommended pin for documentation-quality reasons but the on-disk plugin behaviour is identical.

## [0.2.0] — 2026-05-09 — Embedded by default

### Added

- **In-process backend** (`yantrikdb/embedded.py`) wrapping `yantrikdb._yantrikdb_rust.YantrikDB` to the same 8-method surface as the HTTP client. Users running a single Hermes instance no longer need a separate `yantrikdb-server`, Docker, token mint, or URL config. `pip install` and go.
- **Backend factory** (`make_backend()`) selects HTTP vs embedded based on `YANTRIKDB_MODE` env (default `embedded`). Provider's tool dispatch is unchanged — same 8 tools, same hooks, same namespace scoping, same circuit breaker policy.
- **New env config**: `YANTRIKDB_MODE` (`embedded` | `http`), `YANTRIKDB_DB_PATH` (defaults to `$HERMES_HOME/yantrikdb-memory.db`), `YANTRIKDB_EMBEDDER` (`""` for the bundled potion-base-2M, or `potion-base-8M` / `potion-base-32M` for tier-2/3 download paths).
- **Hermes-on-LXC verification for embedded mode** captured in `VERIFICATION.md` — real DeepSeek session, 3× `yantrikdb_remember` + `yantrikdb_recall` + `yantrikdb_stats` all sub-millisecond after one-time 80 ms engine warmup.
- 96 tests passing, all transport-agnostic — they exercise the provider contract, not the backend.

### Changed

- **Default backend is now embedded** (`YANTRIKDB_MODE=embedded`). Users pinning v0.1 behavior should set `YANTRIKDB_MODE=http` explicitly.
- `pip_dependencies` adds `yantrikdb>=0.7.6` (required for the bundled embedder via `YantrikDB.with_default()`). v0.7.6 ships only `uuid-utils` + `click` as hard deps; the install is ~10 MB total.
- `is_available()` now mode-aware: embedded mode is available iff `yantrikdb` is importable; HTTP mode requires a token (unchanged).
- `YantrikDBConfig` extended with `mode`, `db_path`, `embedder_name` fields; HTTP-only fields (`url`, `token`, `connect_timeout`, etc.) and embedded-only fields coexist on one dataclass.

### Performance (steady-state, post-warmup)

| Op | v0.1 HTTP (Apr 14, LXC vs LAN cluster) | v0.2 Embedded (today, in-process) |
|---|---|---|
| `record_text` p50 | ~13.8 ms | **0.60 ms** |
| `recall_text` p50 | ~24.0 ms | **2.58 ms** |
| Token mint at install | required | not needed |
| Server / Docker | required | not needed |
| Cold start (one-time) | n/a | 77 ms |

### Notes for HTTP-mode users

The HTTP backend (`YANTRIKDB_MODE=http`) is unchanged in v0.2 and still recommended for:

- HA cluster deployments where multiple Hermes instances share one yantrikdb-server.
- Multi-tenant scenarios needing the cluster's centralized control plane.
- Auditing setups requiring server-side request logs.

## [0.1.0] — 2026-04-14 — Initial

### Added

- `YantrikDBMemoryProvider` implementing Hermes' `MemoryProvider` ABC.
- Eight tool schemas: `yantrikdb_remember`, `yantrikdb_recall`, `yantrikdb_forget`, `yantrikdb_think`, `yantrikdb_conflicts`, `yantrikdb_resolve_conflict`, `yantrikdb_relate`, `yantrikdb_stats`.
- Explainable recall — the `why_retrieved` reason list from the server is surfaced per result.
- Structured `think()` response with consolidation counts, conflict counts, patterns, duration, and server-suggested triggers.
- Three optional hooks: `on_session_end` (auto-consolidation), `on_pre_compress` (preserves high-salience memories through Hermes context compression), `on_memory_write` (mirrors built-in MEMORY.md / USER.md additions).
- Typed error taxonomy: `YantrikDBAuthError`, `YantrikDBClientError`, `YantrikDBTransientError`, `YantrikDBServerError` on a `YantrikDBError` base.
- Circuit breaker: 5 consecutive transient/server/auth failures → 120 s cooldown. 4xx errors do not trip the breaker.
- Bounded HTTP retries on transient 5xx and connection blips (urllib3 Retry with exponential backoff).
- Per-request `req_id` + `latency_ms` at DEBUG for post-hoc log correlation.
- Client-side text truncation at `YANTRIKDB_MAX_TEXT_LEN` (default 25000) with a visible marker.
- Config resolution: env vars first, `$HERMES_HOME/yantrikdb.json` overlay, numeric coercion, empty-value skip.
- Configurable timeouts and retry count via env: `YANTRIKDB_READ_TIMEOUT`, `YANTRIKDB_CONNECT_TIMEOUT`, `YANTRIKDB_RETRY_TOTAL`.
- Namespace scoping: `{base}:{agent_workspace}:{agent_identity}` for per-identity isolation while allowing cross-session consolidation.
- `get_config_schema()` + `save_config()` so `hermes memory setup` can walk the user through token + URL configuration.
- 94 tests covering config loading, request formation, error taxonomy, tool dispatch, hook semantics, circuit breaker behavior, and text truncation. All tests run without network.

### Deliberate non-goals for this release

- No assistant-message extraction on `sync_turn` (hallucination amplification risk).
- No embedded / in-process YantrikDB — the plugin is a thin HTTP client. (Reversed in v0.2.0.)
- No local SQLite fallback — out of scope.
- No batch write queue — background threads already absorb latency; the added complexity is not justified for v1.
- No CLI subcommand (`hermes yantrikdb …`) — still tracked as future work.
