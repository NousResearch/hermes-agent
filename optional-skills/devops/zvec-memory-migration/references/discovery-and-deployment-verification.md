# Cross-Profile Deployment and the "Deployment Verification" Iron Rule

This document supplements SKILL.md, recording the root cause and reusable rules
from a 2026-09 incident in which the memory system silently stopped working for 5 weeks.

---

## 1. Memory provider discovery root is profile-scoped (the most easily misjudged architecture fact)

The discovery order in `plugins/memory/__init__.py` (verified against source; do not rely on memory):

```
① System tree bundled:  <repo>/plugins/memory/<name>/        ← priority, shared by all profiles
② User tree user:     $HERMES_HOME/plugins/<name>/         ← profile-scoped!
③ Project tree project:  ./.hermes/plugins/  (requires HERMES_ENABLE_PROJECT_PLUGINS)
④ pip entry point: group = hermes_agent.memory_providers  ← shared by all profiles
```

**Key point**: `$HERMES_HOME` is `~/.hermes/` for default and
`~/.hermes/profiles/<name>/` for sub-profiles. So a plugin placed at `~/.hermes/plugins/memory-zvec/`
is **visible only to default**; sub-profiles report `Plugin: NOT installed`.

**Consequence**: the sub-profile's config.yaml still says `provider: memory-zvec`, but external memory silently fails,
and `agent.log` contains not a single memory provider log line.

### Therefore: the correct placement for making the memory plugin a "system plugin"

| Option | Cross-profile | Survives updates | Cost |
|------|-----------|---------|------|
| A. Place in system tree `<repo>/plugins/memory/<name>/` | ✅ zero config | ⚠️ untracked file | git handling required (see below) |
| B. Install via pip entry point into the venv | ✅ | ✅ | lost on venv rebuild (exactly the silent failure to avoid) |
| C. Create a symlink per profile | ⚠️ per-profile work | ✅ | new profiles will surely be missed; a past incident was lost exactly this way |

**Recommended: A**, and use `.git/info/exclude` to make it invisible to git:

```bash
# Append to <repo>/.git/info/exclude (local file; never committed, unaffected by git pull)
plugins/memory/<name>/
```

Why this is safe: `hermes update` uses `git stash push --include-untracked` —
**ignored files are not stashed** (that is `--all` behavior); the Windows ZIP fallback path requires a clean worktree,
and ignored files do not count as dirty, so it will not trigger the "delete untracked files" tree replacement either.
Verification: `git status --short` should not show the plugin; `git check-ignore -v <path>` should match.

---

## 2. Deployment verification iron rule: you must run discovery; checking files/content alone is not enough

**Looking only at "is the file there" and "how many rows are in the store" leads to wrong conclusions.** The memory system has three layers, and each must be verified layer by layer:

```bash
# Layer 1: discovery (the easiest to skip, and the true cause of this incident)
for prof in default <profile> <profile> <profile> <profile>; do
  if [ "$prof" = "default" ]; then HH="$HOME/.hermes"; else HH="$HOME/.hermes/profiles/$prof"; fi
  HERMES_HOME="$HH" <venv>/python -c "
from plugins.memory import list_memory_provider_names, find_provider_dir
n = list_memory_provider_names()
print('$prof', 'memory-zvec' in n, find_provider_dir('memory-zvec'))"
done
```

```bash
# Layer 2: availability (provider self-report; False if the embedding model is missing)
hermes memory status            # default
hermes -p <profile> memory status
# Expect Status: available ✓ (not not available ✗)
```

```bash
# Layer 3: functionality (the write path must be tested for real; read-only checks do not count as verification)
<venv>/python scripts/verify-plugin-tools.py
```

**Note**: the `default` profile's `HERMES_HOME` is `~/.hermes/`, **not**
`~/.hermes/profiles/default/`. Any script that builds paths per profile must special-case default,
otherwise the verification script itself errors (an older `scripts/verify-plugin-tools.py` had this bug).

---

## 3. Top cause of silent failure: embedding model tag drift

`is_available()` is implemented as "call embedding once; return False on exception," and it swallows all exceptions
into `logger.debug`. So **when the tag changes on the Ollama side → the entire memory channel shuts down with almost no warning**.

```bash
# Mandatory health-check item: can the configured tag actually embed?
grep -h embedding_model ~/.hermes/config.yaml ~/.hermes/profiles/*/config.yaml
curl -s http://localhost:11434/api/tags | python3 -c "import sys,json;print([m['name'] for m in json.load(sys.stdin)['models']])"
# Test per profile in practice (replace <tag> with the configured value)
curl -s http://localhost:11434/api/embed -d '{"model":"<tag>","input":["t"]}' | head -c 120
```

Rule: **when changing a profile's embedding tag, change it for all profiles at the same time** — fixing only one profile
is the source of recurrence (first discovered in one profile; the other four kept running with a broken tag for another half month).
When the dimension is unchanged (bge-m3 = 1024 across the board), existing vectors need no migration.

---

## 4. Zvec version differences (the 0.5.x notes are outdated)

**Lock semantics (tested on 0.6.0; wording made precise):**

| Scenario | Result |
|------|------|
| Read-only + read-only (concurrent) | ✅ **read-only locks are shared**; multiple read-only opens can coexist |
| Read-only ↈ active writer process | ❌ `Can't lock read-only collection: .../LOCK` — **read-only and write locks are mutually exclusive** |
| Read-write + read-write | ❌ `Can't lock read-write collection: .../LOCK` (exclusive) |

**Implications**:
- With no writer process, "read-only + mmap" enables multiprocess concurrent inspection (the older note claimed
  read-only also needed an exclusive lock — outdated).
- But **with an active writer (rebuild/migration/gateway session), even read-only cannot open** — before diagnosing, first confirm
  there is no writer process, otherwise you may misdiagnose "corrupt store".
- The plugin's read-only fallback (`_open_read_only`) only helps after the lock **holder has been released by GC**.

`plugin.yaml`'s `dependencies: [zvec>=0.5.0]` has **no upper bound**, while `hermes update` refreshes
"active memory provider dependencies" → an unconstrained upgrade path exists. The repository policy requires
pre-1.0 to use `>=floor,<0.(minor+2)`. Before upgrading, cross-check the API against `references/zvec-api-reference.md`.

`plugin.yaml`'s `dependencies: [zvec>=0.5.0]` has **no upper bound**, while `hermes update` refreshes
"active memory provider dependencies" → an unconstrained upgrade path exists. The repository policy requires
pre-1.0 to use `>=floor,<0.(minor+2)`. Before upgrading, cross-check the API against `references/zvec-api-reference.md`.

---

## 4b. Score semantics: the vector branch is "distance", FTS/hybrid are "similarity" (‼️ a pitfall that reverses ranking)

**Measured (3-document controlled set, COSINE):**

| Branch | `score` meaning | Measured | Higher = more relevant? |
|------|-------------|------|-------------|
| Vector `Query(field_name="vector", ...)` | **distance** = `1 - cosine_similarity` | same vector 0.000000; near neighbor 0.006116; orthogonal 1.000000 | ❌ lower = more relevant |
| FTS `Query(field_name="content", fts=...)` | **similarity** | 3-word hit 0.724 > 1-word hit 0.453 | ✅ |
| Hybrid `MultiQuery + RrfReRanker` | RRF score (~0.03 magnitude) | — | ✅ |

**Any code path that does `sort(key=score, reverse=True)` is wrong for the vector branch**: it ranks the worst first;
combined with `if score < min_score: continue`, it keeps only the **least similar**.

**Diagnostic method**: query the **self-vector** of a document already in the store and check the top1 score —
≈ 0.0 means distance semantics.

**Fix**: normalize the vector branch to similarity so all three branches share one semantics:
```python
similarity = 1.0 - float(getattr(r, "score", 1.0))
if similarity < min_score:
    continue
... _row_to_dict(r, score=similarity, source="vector", ...)
```

**Affected downstream**: `mode=vector` searches, `_do_hybrid_search_fallback` (the vector branch's rankings feed RRF),
and **`prefetch()` (injects top-3 into the system prompt)** — before the fix, this was effectively stuffing the least relevant memories into the prompt.

**Legacy caveat**: the three branches' scores are on **different scales** (vector ~0.6-0.7, FTS ~15-24, RRF ~0.03),
so `min_score` means very different things across modes; check the actual score range of a mode before using `min_score` with it.

---

## 5. HNSW completeness: the main write path must amortize optimization

**Symptom**: `index_completeness` stays < 1 for a long time (measured 0.71), while `enable_hnsw_optimize: true`
in the config looks effective but actually only applies to `on_session_end`.

**Root cause**: `sync_turn` (the main per-turn real-time write path) only does `insert + flush`, never `optimize()`.

**Fix** (module-level constant + write counter):

```python
_OPTIMIZE_EVERY_N_WRITES = 64   # amortize: optimize once every N writes

# inside _insert(), after flush
self._writes_since_optimize += 1
if (self._writes_since_optimize >= _OPTIMIZE_EVERY_N_WRITES
        and self._config.get("enable_hnsw_optimize", True)):
    try:
        self._coll.optimize()
        self._writes_since_optimize = 0
    except Exception as opt_err:
        logger.debug("amortized optimize failed: %s", opt_err)
```

Unit-test method (no Ollama/real store needed): mock a `_coll` that counts insert/flush/optimize calls;
200 writes should yield `optimize == 200 // 64 == 3`; 0 when the switch is off; an optimize exception must not interrupt writes.

---

## 6. A "stale user-tree mirror" can shadow the system tree (a troubleshooting trap)

If `~/.hermes/plugins/` contains an **old copy of entire category directories** (browser/ model-providers/
platforms/ memory/ …), two classes of problems arise:

1. **Generic plugins**: `PluginManager` is later-wins (project > user > bundled) → the user copy wins,
   and system-tree updates are completely shadowed; also `gate_manifest`'s bundled auto-load branch stops working,
   so backend/platform plugins that were auto-enabled become `not enabled`.
2. **Memory providers**: the opposite — bundled has **priority**. So the user copy is inert for loading, but **it still appears in
   `list_memory_provider_names()`**, and it also makes the `memory/` category directory's own `__init__.py`
   be treated as a provider named `memory` (log: `Memory provider 'memory' loaded but no
   provider instance found`).

**Judgment and disposal**: do not judge new/old by mtime/version number (the two versions may be complementary); do a **content-level diff** first;
after confirming no unique content, **archive and move it away** (`cp -a` outside ~/.hermes, verify the file counts match, then `rm -rf`),
then re-run the layer 1/2 verification.

### 6.1 Reusable procedure for cleaning stale mirrors (measured on 54 items, zero false deletions)

**Step 1 — three-way reconciliation (read-only)**: for each `~/.hermes/plugins/<d>`:
```
No same name in system tree            → KEEP (user-only, e.g. example-dashboard / strike-freedom-cockpit)
Files in user tree absent from system tree  → eyeball it manually (may hold unique content)
Identical file sets / version-only diffs → shadowing source, queue for cleanup
```
Compare per-file md5 sets (excluding `__pycache__`/`*.pyc`); do not rely on directory size or version numbers.

**Step 2 — the decisive test for "contains local customization"**:
Compute the mtime distribution of all files in the user tree. **All one timestamp + no recent changes** ⇒ a bulk-copied mirror,
containing no hand edits.
> ⚠️ Reverse caveat: `cp -a` **preserves** mtime, so matching timestamps alone cannot prove the files were never edited;
> fully identical timestamps (e.g. all in the same second) look more like a Windows-side copy/extraction (which normalizes timestamps)
> than `cp -a`. To fully rule out customization, run a **behavioral equivalence test** on the most-different files
> (load both versions, call the same method with real arguments, compare return values). Measured on the deepseek provider:
> the two versions' code structures differ completely (the old version inlined `_model_supports_thinking`; the new version uses the core module
> `agent.reasoning_effort`), but the output for `deepseek-v4-flash` was **identical field by field** ⇒ refactor-style upgrade, safe to replace.

**Step 3 — archive and move away in batches** (each batch: archive → verify file counts match → move away → re-run discovery verification):

| Batch | Content | Benefit |
|------|------|------|
| 1 | backend / platform directories | Recover silently disabled auto-load capabilities (most visible) |
| 2 | standalone / others | Return to system versions |
| 3 | model-providers | Do last (model routing depends on it; run the behavioral equivalence test first) |

**Step 4 — closing verification** (complete only when all four pass):
1. Shadow count back to zero (for each `_plugins` entry, compare whether `manifest.path` is still in the user tree while the system tree has a same-named one)
2. All providers the user actually uses resolve to BUNDLED (`providers.get_provider_profile(n)` + `inspect.getfile`)
3. User-only plugins still present
4. Memory system functional verification re-run (55/55) — incidentally confirms the batch operations did not break anything else

**Incidental finding (do not confuse cause and effect)**: platform-type plugins are **lazy-loaded** (`defer`); after cleanup,
their showing as "not found" in `_plugins` is **normal**; check `PluginManager._plugin_platform_names` to confirm registration is complete.

---

## 7. Backup coverage: memory stores must enter the "absolute checklist"

**Lesson**: the default backup script's `KEY_FILES` only checks `config.yaml / SOUL.md / cron/jobs.json`,
**not the memory stores**. And once a memory store disappears at the source, ordinary "compare size against the last backup" logic cannot notice it —
the backup faithfully copied the "absent at the source too" state, with consecutive all-green rounds and zero alerts.

**Rule**: the backup script must maintain an `EXPECTED_MEMORY_DBS` absolute list and verify each item:

```
① Existence at the source   → alert if missing (the only defense against silent loss)
② Existence on the backup side → present at source but absent in backup = backup pipeline failure
③ Per-DB size comparison → shrunk ≥30% or vanished = block rotation; keep all old backups for manual review
```

Also check `ROTATION_KEEP`: retained copies × execution frequency = the actual lookback window.
`KEEP=3` + 3 runs per week ≈ only a 1-week window — problems noticed as "fine last week, broken this week" cannot be traced.

---

## 8. Rebuild the memory store from state.db (the recovery path after loss)

When the memory store is lost but `state.db` is intact, the content can be rebuilt (what is lost is vectors, not conversation).

**Pairing granularity**: take it by "turn" — each user message pairs with that turn's **last assistant reply with substantive content**.
Naively pairing with "the next assistant message" pairs an empty stub that only launched tool calls, dropping the entire turn's real answer
(measured: naive pairing kept 392 items; turn-level pairing kept 1302).

**Must filter**:
- System-injected pseudo user messages (`[IMPORTANT: Background process ... completed`,
  `[CONTEXT COMPACTION`, `{"output"`, etc.)
- Turns whose assistant content is too short (tool-call stubs)

**Must sanitize first**: historical conversations often contain plaintext credentials (measured: hit a Feishu appSecret). The correct approach is
**extracting real secret values from `.env` for exact replacement** (zero false positives) + key-name pattern replacement, and **removing**
"generic long-string replacement" style over-sanitization (it destroys commit SHAs, model names, and other legitimate content).
A residual scan must be run before delivery, with a requirement of 0 hits.

**Idempotency**: dedupe by `(session_id, created_at)`; safe to re-run after failure.
