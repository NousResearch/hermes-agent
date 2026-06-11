# hexis_appraisal — observational metacognitive appraisal for hermes-agent

A `kind: standalone` hermes-agent plugin that runs one deadline-bounded, JSON-mode LLM
call per eligible turn (via the host-owned `ctx.llm.complete_structured`) and renders a
sentinel-prefixed `[hexis appraisal]` advisory block into the user message only. A
debounced reflection pass consolidates what each turn revealed into a small local SQLite
state store, so later appraisals carry context across sessions.

Zero new pip dependencies. Zero autonomy. Fail-open everywhere: no code path in any hook
may raise into the dispatcher or delay a turn past its configurable deadline — every
failure degrades to an empty injection plus a telemetry row.

## Non-goals (locked anti-features)

The plugin **never**:

- executes tools or asks for tool execution
- writes to memory providers (or mentions them)
- gates, blocks, retries, or delays a turn — appraisal is bounded by an executor
  deadline and every failure path returns `None`
- emits directive language — rendered lines are observational; imperative payload text
  survives only as quoted reported material (enforced by a directive-language test corpus)
- runs schedulers, timers, or background daemons — all work happens inside the four
  registered hooks

## Install — standalone

Copy **or** symlink this directory to `$HERMES_HOME/plugins/hexis_appraisal`, then
enable it:

```sh
# copy
cp -R hexis_appraisal "$HERMES_HOME/plugins/hexis_appraisal"
# …or symlink (development-friendly: the live install tracks your checkout)
ln -s /path/to/checkout/hexis_appraisal "$HERMES_HOME/plugins/hexis_appraisal"

hermes plugins enable hexis_appraisal
```

To verify the plugin loads, run a turn with plugin debug logging:

```sh
HERMES_PLUGINS_DEBUG=1 hermes -z "hello"
```

## Install — in-tree

When this directory is present under the hermes-agent repo as
`plugins/hexis_appraisal/`, the loader discovers it as a bundled plugin. Bundled
`kind: standalone` plugins are opt-in — the same enable command applies:

```sh
hermes plugins enable hexis_appraisal
```

The loader reads the `provides_hooks` manifest key from `plugin.yaml`; this plugin
declares exactly the four hooks it registers (`on_session_start`, `pre_llm_call`,
`post_llm_call`, `on_session_end`), which a manifest test asserts.

## Configuration

All keys live under `plugins.entries.hexis_appraisal` in the host config. Every key is
optional; absent or malformed config (or an unreadable host config) degrades to the
defaults below without error. Out-of-range values are clamped, wrong-typed values fall
back to the default.

```yaml
plugins:
  entries:
    hexis_appraisal:
      enabled: true                  # kill switch — false disables appraisal AND reflection
      confidence_threshold: 0.6      # float, clamped [0.0, 1.0] — signals below it are dropped
      deadline_seconds: 8.0          # float, clamped [0.5, 10.0] — appraisal wall-clock bound
      history_chars: 4000            # int, >= 0 — conversation-history budget in the prompt
      max_tokens: 700                # int, >= 1 — appraisal completion budget
      reflection_enabled: true       # reflection-only switch (appraisal unaffected)
      reflect_every_n_turns: 5       # int, clamped [1, 50] — debounce between reflections
      reflect_max_tokens: 700        # int, >= 1 — reflection completion budget
      reflect_deadline_seconds: 8.0  # float, clamped [0.5, 10.0] — reflection wall-clock bound
      llm:
        model: claude-haiku-4-5      # optional model override; default: none (host's active model)
```

Notes:

- `enabled: false` is the kill switch — the plugin records a `skipped:disabled`
  telemetry row and injects nothing.
- The `llm.model` override only *requests* a model; the host trust gate
  (`allow_model_override` / `allowed_models`) decides. If the gate denies the request,
  the plugin retries once with the host's active model (outcome `trust_fallback`) —
  zero other retries.
- A cheap/fast tier is recommended for the override (e.g. `claude-haiku-4-5`,
  `gpt-4o-mini`, `gemini-2.5-flash`) — documented only, never auto-applied.

## Observability

State lives in the plugin's own SQLite database at
`$HERMES_HOME/hexis_appraisal/state.db` (WAL mode). The `telemetry` table (capped at
2000 rows, oldest evicted) records one row per hook decision.

Outcome vocabulary:

| Outcome | Meaning |
|---------|---------|
| `ok` | appraisal succeeded |
| `trust_fallback` | appraisal succeeded after the trust gate denied the model override |
| `timeout` | appraisal exceeded `deadline_seconds`; turn proceeded without a block |
| `llm_error` | appraisal call failed |
| `parse_fail` | appraisal response did not match the JSON schema |
| `skipped:<reason>` | appraisal not attempted — `disabled`, `no_ctx`, `empty`, `social_close`, `duplicate` |
| `reflect_ok` | reflection pass applied deltas and advanced the watermark |
| `reflect_timeout` / `reflect_llm_error` / `reflect_parse_fail` | reflection failure; state unchanged |
| `reflect_skipped:<reason>` | reflection not attempted — `disabled`, `no_ctx`, `no_turns`, `debounce`, `db_locked` |

`store.telemetry_summary()` is the derived view: `total`, `by_outcome`,
`failure_count` (every outcome **except** `ok`, `trust_fallback`, `reflect_ok` and the
`skipped:*` / `reflect_skipped:*` prefixes — i.e. `timeout`, `llm_error`, `parse_fail`,
`reflect_timeout`, `reflect_llm_error`, `reflect_parse_fail`, plus any future unknown
outcome), `last_error` (error string of the newest failure row, same definition), and
`p50_wall_ms` (median wall time over appraisal `ok`/`trust_fallback` rows only —
`reflect_*` walls are excluded).

**Inspecting a live database — copy the WAL sidecars.** The store runs in WAL mode; a
read-only URI open against a bare `state.db` copy fails with `SQLITE_CANTOPEN` when the
`-wal`/`-shm` sidecars are missing. Copy all three files together:

```sh
cp "$HERMES_HOME/hexis_appraisal/state.db"     /tmp/hexis-inspect.db
cp "$HERMES_HOME/hexis_appraisal/state.db-wal" /tmp/hexis-inspect.db-wal
cp "$HERMES_HOME/hexis_appraisal/state.db-shm" /tmp/hexis-inspect.db-shm
sqlite3 "file:/tmp/hexis-inspect.db?mode=ro" \
  "SELECT outcome, COUNT(*) FROM telemetry GROUP BY outcome;"
```

For ad-hoc block inspection, `HEXIS_APPRAISAL_DEBUG_DUMP=/path/to/file` appends every
rendered `[hexis appraisal]` block to that file (off by default).

## Sub-session behavior

The host runs the full hook set for every sub-session, not just top-level turns. When a
turn fans out into parallel sub-sessions, their appraisals can contend with each other
into the fail-open timeout (observed live: `timeout` rows at 8005–8006 ms in parallel
sub-sessions; the turns proceeded intact every time). This is the designed degradation —
the cost is wasted appraisal calls, never a blocked turn.

## Limitations

- **One-turn lag is deliberate.** `pre_llm_call` fires before memory prefetch, so an
  appraisal sees the user message, conversation history, and the plugin's own SQLite
  state — not the current turn's memory injection. The reflection pass is the carrier of
  that context across the lag: what a turn revealed reaches future appraisals through
  reflected state, one turn later.
- **Deadline/latency trade-off.** The default 8.0 s deadline assumes a cheap-tier model;
  observed live p50 was ≈5.5 s with `claude-haiku-4-5`. Headroom is thin under load, and
  the live timeout fraction was nontrivial (3 timeouts vs 6 ok among real appraisal
  attempts at validation time). Every timeout fails open — the turn proceeds without a
  block.
- **Trust-gate fallback degrades on slow hosts.** The `trust_fallback` retry uses the
  host's active model; on hosts whose active model is slower than the deadline clamp
  (max 10.0 s), the fallback degrades to the designed fail-open timeout instead. On
  hosts with faster active models the fallback works as intended.

## State and privacy

Everything stays local under `$HERMES_HOME/hexis_appraisal/` — no network beyond the
host-mediated LLM calls. Tables are capped (`concerns` 20, `contradictions` 50,
`turn_log` 500, `trust_scores` 64, `telemetry` 2000) and weighted entries decay lazily
(half-life 7 idle days; entries below 0.1 effective weight are excluded from snapshots
and pruned during reflection). `turn_log` keeps short excerpts of user messages and
assistant responses; the plugin's own sentinel lines are stripped before storage so
rendered blocks never feed back into reflection.

The appraisal pre-phase never writes state — it reads a read-only snapshot. Belief-state
writes (affect, concerns, contradictions, trust scores, watermark) happen only inside
the reflection pass, through a single transactional funnel (`store.apply_deltas`). The
only other writes are bookkeeping: turn capture in `post_llm_call` and telemetry rows.

## Tests

From the development repo: `./scripts/test.sh` (stages pytest into a repo-local
`.devtools/` dir; the host venv is never modified). The suite is offline — no network,
no live database access.
