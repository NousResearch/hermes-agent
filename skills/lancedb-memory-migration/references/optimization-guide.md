# LanceDB Memory Optimization — Post-Migration Content Quality

**When to use:** After running a migration script, evaluate and optimize the quality of stored vectors.
**Current version:** v2 (`optimize_lance_memory_v2.py`) — Pipeline + Twig + Incremental + Pattern Discovery.

## Why Naive Migration Fails

The migration template stores raw concatenated `[user]/[assistant]` content directly from `state.db`. This causes a critical problem:

```
Session A (raw): [IMPORTANT: skill frontmatter 4700 chars][assistant] REAL REPORT 500 chars
Session B (raw): [IMPORTANT: skill frontmatter 4700 chars][assistant] REAL REPORT 480 chars
                    ↑─── ~4700 chars identical across all sessions ───↑
Vector A ≈ Vector B  (cos ≈ 1.0)
```

**Result:** All session vectors are nearly identical. Semantic search becomes useless — every query returns the same results regardless of intent.

### Root Cause

Skill invocation blocks (`[IMPORTANT: The user has invoked the "skill-name" skill...]` + YAML frontmatter `---...---`) are **hardcoded into every session** and account for 60-80% of total content length. These duplicate blocks dominate the embedding vector, drowning out the actual signal.

## The v2 Optimization Pipeline

The v2 script (`optimize_lance_memory_v2.py`) applies these steps:

```
state.db → Strip Pipeline → Twig Split → Twig Dedup → LanceDB
              ↑ Pattern Registry (configurable)
```

### Step 1 — Strip Pipeline (Layered Pattern Registry)

Unlike v1's single `extract_real_content()` function, v2 uses a **pluggable pipeline** of independent Stages. Each Stage handles one type of duplicate:

| Stage | Type | Action | Examples |
|-------|------|--------|----------|
| `prefix` | regex | strip_before | `[IMPORTANT: skill...]---<yaml>---` frontmatter blocks |
| `assistant_frontmatter` | regex | strip_after_match | `[assistant]---## Overview\n## When to Use\n...` — skill frontmatter 节标题嵌在 assistant 响应中 |
| `trailing` | regex | strip_after | trailing markers, fixed footer text |
| `embedded_meta` | regex | replace | UUID session IDs, embedded timestamps |
| `quality_gate` | quality | filter | short greetings, pure punctuation, residual template fragments |

**Actions defined in `_run_regex_stage`:**
- `strip_before` — `text = text[m.end():]` (remove match + everything before it)
- `strip_after_match` — `text = text[:m.start()] + text[m.end():]` (remove match only, keep prefix; essential for `assistant_frontmatter` to preserve `[assistant]\n` while removing what follows)
- `strip_after` — removes everything from last match to end of text
- `replace` — substitute with string literal

**Trailing `---` pattern pitfall (CRITICAL — destroys report body content):**
The pattern `r'^\s*---\s*$.*'` uses `.*` which greedily consumes EVERYTHING after the first `---` in the document — including report body separators and real content. **Always use lookahead** `(?=[\n\s]*$)` to ensure only **trailing** `---` is stripped:

```python
# WRONG — deletes report body section dividers and all following content
r'^\s*---\s*$.*'

# CORRECT — only matches --- at document end (followed only by whitespace)
r'^\s*---\s*$(?=[\n\s]*$)'
r'\n{3,}---\s*$(?=[\n\s]*$)'
```

**Common trailing patterns (all use lookahead `(?=[\n\s]*$)`):**
```python
trailing_patterns = [
    r'将搜索结果整理成舆情报告格式.*',
    r'## Skill Execution Summary.*',
    r'^\s*---\s*$(?=[\n\s]*$)',        # trailing separator only
    r'\n{3,}---\s*$(?=[\n\s]*$)',        # 3+blank+--- at end
    r'\n{5,}$',                            # 5+ trailing newlines
    r'\n\*\*(?:提示|注意|Warning).*?$',   # trailing notes
    r'\*本报告基于公开信息整理.*$',     # fixed report disclaimer
]
```

**Key benefit:** Adding a new pattern type only requires editing the `DEFAULT_STRIP_STAGES` config — no code logic changes needed. Different profiles can have different Stage configurations.

```python
# Stage config example (in DEFAULT_STRIP_STAGES):
{
    "id": "prefix",
    "name": "模板前缀裁剪",
    "enabled": True,
    "type": "regex",
    "action": "strip_before",
    "patterns": [
        # Priority 1: complete frontmatter block
        r'\[IMPORTANT:[^\]]*\]\n\n---\n[\s\S]{50,5000}\n---\n+',
        # Priority 2: frontmatter without [IMPORTANT:] wrapper
        r'\n---\n[\s\S]{50,5000}\n---\n+',
    ],
    "flags": re.DOTALL | re.IGNORECASE,
    "stop_on_first": True,
}
```

**Actual results on zunhunfan profile:** 67.2% content stripped (from 214,362 to 70,255 chars).

### Step 2 — Twig Split (Sub-Session Retrieval)

Instead of storing entire sessions as single records, v2 splits sessions into **Twigs** — the smallest independently retrievable unit.

| Strategy | Trigger | Use Case |
|----------|---------|----------|
| `pair_uai` | 2+ `[user]/[assistant]` pairs | cron jobs with multiple independent reports |
| `section_header` | 2+ `##` headings, clean_len > 3000 | long reports divided by section (e.g. skill frontmatter templates split into many thin twigs) |
| `last_pair_only` | single pair, clean content extracted | short sessions with one query — `len_clean` = actual body length, not full part |
| `no_split` | content < 3000 chars | small sessions kept whole |

**Known issue with `len_clean`:** `split_into_twigs` sets `len_clean = len(part)` for all strategies, which does NOT account for frontmatter stripped by the pipeline. The `extract_meta` function compensates by calling `_estimate_clean_len(content)` to detect residual frontmatter and correct `clean_len` / `strip_ratio`. Do not rely on `twig.len_clean` for accuracy — use `meta["strip_ratio"]` instead.

Each Twig carries metadata linking it back to its parent session: `twig_id`, `twig_index`, `twig_count`, `twig_strategy`.

**Benefits:**
- Searching "哲弗智能负面" now returns only the matching Twig, not the entire session containing 8 different company reports
- Twig-level dedup (cos > 0.98) is more precise than session-level dedup

### Step 3 — Twig-Level Deduplication

Cosine similarity > 0.98 on Twig content (not full session). Older Twigs deleted, newer ones kept.

### Step 4 — Incremental Optimization

The `--incremental` flag skips already-processed sessions by reading existing `session_id`s from LanceDB. State is persisted in `~/.hermes/profiles/<name>/.optimization_state.json`.

```bash
# Full optimization
python optimize_lance_memory_v2.py --profile zunhunfan --apply

# Incremental (only new sessions)
python optimize_lance_memory_v2.py --profile zunhunfan --apply --incremental
```

### Step 5 — Pattern Auto-Discovery

The `--discover-patterns` flag scans new sessions for unrecognised template blocks (matches against `PATTERN_DISCOVERY_PATTERNS`). Discovered samples are stored in the optimization state file for manual review and addition to the Pattern Registry.

```bash
python optimize_lance_memory_v2.py --profile zunhunfan --discover-patterns --dry-run
```

### Step 6 — Write with Correct Session IDs

**Critical:** Always read `session_id` from `state.db`, never from LanceDB metadata.

```python
# Read session_id directly from state.db
conn = sqlite3.connect(str(STATE_DB))
cur = conn.cursor()
cur.execute("""
    SELECT id, message_count, started_at,
           datetime(started_at, 'unixepoch') as start_dt
    FROM sessions WHERE message_count > 0 ORDER BY started_at
""")
db_sessions = cur.fetchall()
conn.close()
```

### Step 7 — LanceDB Write with Overwrite

```python
db.create_table("memories", schema=schema, mode="overwrite")
# NOT: db.create_table("memories", schema=schema)  # fails if table exists
```

## Running the Optimization

```bash
# List current Strip Pipeline patterns
python optimize_lance_memory_v2.py --list-patterns --profile zunhunfan

# Export patterns to JSON for editing
python optimize_lance_memory_v2.py --export-patterns --profile zunhunfan

# Dry-run first
python optimize_lance_memory_v2.py --profile zunhunfan --dry-run

# Apply full optimization
python optimize_lance_memory_v2.py --profile zunhunfan --apply

# Incremental (only new sessions)
python optimize_lance_memory_v2.py --profile zunhunfan --apply --incremental

# Auto-discover new patterns
python optimize_lance_memory_v2.py --profile zunhunfan --discover-patterns --dry-run
```

Expected output after optimization:
- Content length reduced 60-77%
- **Twigs as minimum retrieval unit** (not full sessions)
- Topic/sentiment/source labels + Twig metadata on every record
- `recency_weight` field for re-ranking by date
- Cosine distance now shows genuine differentiation (dist 0.46-0.99, not cos ≈ 1.0)

## Verification

```python
import lancedb, json
from pathlib import Path

LANCE_DIR = Path.home() / ".hermes/profiles/<profile>/lance_memory"
db = lancedb.connect(str(LANCE_DIR))
table = db.open_table("memories")
rows = table.search([0.0]*1024).limit(10000).to_list()

# Check: bad session_id = "_optimized" strings
bad = [r for r in rows if "_optimized" in r["session_id"]]
print(f"Bad session_ids: {len(bad)}")  # Should be 0

# Check: Twig metadata present
m = json.loads(rows[0]["metadata"])
print(f"Has twig_id: {'twig_id' in m}")  # Should be True
print(f"Has twig_index: {'twig_index' in m}")  # Should be True
print(f"Has twig_count: {'twig_count' in m}")  # Should be True

# Check: average length (should be 500-3000 chars, not 4000+)
lens = [len(r["content"]) for r in rows]
print(f"Avg len: {sum(lens)/len(lens):.0f}")  # Should be much lower than raw

# Check: metadata labels
print(f"Has topics: {'topics' in m}")  # Should be True
print(f"Has recency_weight: {'recency_weight' in m}")  # Should be True
```

## When to Run Optimization

- **Immediately after first migration** — naive migration always produces suboptimal vectors
- **When new sessions are added** — run with `--incremental` to only process new sessions
- **Periodically** — to re-chunk long sessions that have grown
- **After adding new skills** — run `--discover-patterns` to detect new template blocks

For ongoing use: modify the migration script to apply content stripping + metadata enrichment at initial migration time, rather than running a separate optimization pass.

## CLI Flags Reference

| Flag | Description |
|------|-------------|
| `--profile` | Hermes profile name (default: zunhunfan) |
| `--dry-run` | Analyze without writing |
| `--apply` | Execute optimization and write to LanceDB |
| `--incremental` | Skip already-processed sessions |
| `--discover-patterns` | Scan for new template patterns |
| `--list-patterns` | List current Strip Pipeline config |
| `--export-patterns` | Export Pattern Registry to JSON |
| `--min-asst-chars` | Skip sessions with less assistant content |
| `--similarity-threshold` | Twig dedup cosine threshold (default: 0.98) |
| `--min-twig-len` | Minimum Twig length (default: 200 chars) |

## Why Naive Migration Fails

The migration template stores raw concatenated `[user]/[assistant]` content directly from `state.db`. This causes a critical problem:

```
Session A (raw): [IMPORTANT: skill frontmatter 4700 chars][assistant] REAL REPORT 500 chars
Session B (raw): [IMPORTANT: skill frontmatter 4700 chars][assistant] REAL REPORT 480 chars
                    ↑─── ~4700 chars identical across all sessions ───↑
Vector A ≈ Vector B  (cos ≈ 1.0)
```

**Result:** All session vectors are nearly identical. Semantic search becomes useless — every query returns the same results regardless of intent.

### Root Cause

Skill invocation blocks (`[IMPORTANT: The user has invoked the "skill-name" skill...]`) are **hardcoded into every session** and account for 60-80% of total content length. These duplicate blocks dominate the embedding vector, drowning out the actual signal.

## The Optimization Pipeline

The full optimization script (`optimize_<profile>_memory.py`) applies these steps:

### Step 1 — Strip: Extract Real Content

For each session, extract only the meaningful content:
- Take the **last** `[assistant]` block (the final actual response, not the skill instruction echo)
- Strip `[IMPORTANT: skill...]` blocks that appear mid-content
- Remove trailing artifact paragraphs (e.g., "完成后将完整报告发送给彤彤。")

```python
def extract_real_content(content: str, session_id: str) -> str:
    last_asst = content.rfind("[assistant]")
    if last_asst == -1:
        return content.strip()
    
    body = content[last_asst + 11:]  # skip "[assistant]\n"
    
    # Strip following IMPORTANT block
    next_imp = body.find("[IMPORTANT:")
    if next_imp > 50:
        body = body[:next_imp].strip()
    
    # Trim trailing artifacts
    body = _trim_trailing_artifacts(body)
    
    if len(body.strip()) < 100:
        return "[EMPTY: no valid report]"  # filter out later
    
    return body.strip()
```

**Expected savings:** 59-77% of raw content stripped for cron/skill-heavy sessions.

### Step 2 — Classify: Add Metadata Tags

From the clean content, extract:

```python
meta = {
    "company": "哲弗智能",           # named entities
    "topics": ["litigation_complaints", "finance_ipo", ...],  # multi-label
    "sentiment": "negative",         # negative / positive / neutral
    "source": "cron",               # cron / feishu / cli
    "days_ago": 5,                  # days since session
    "recency_weight": 0.88 ** (5/7),  # exponential decay: 7d≈0.5, 14d≈0.25
    "strip_ratio": 0.33,            # clean/original ratio (quality indicator)
}
```

### Step 3 — Deduplicate

Compare vectors of clean content (not raw). Use cosine threshold **0.98** (stricter than search threshold):

```python
SIMILARITY_THRESHOLD = 0.98  # cos > 0.98 = approximate duplicate

# Keep newer, mark older for removal
to_remove = set()
for i, rec_newer in enumerate(records_sorted_by_date):
    for rec_older in records_sorted[i+1:]:
        if cosine_sim(rec_newer.emb, rec_older.emb) > SIMILARITY_THRESHOLD:
            to_remove.add(rec_older.session_id)
```

**Key insight:** After stripping, different sessions have meaningful differentiation (cos 0.47-0.99), so dedup finds only truly identical content — not all sessions at once.

### Step 4 — Chunk (Long Sessions Only)

Only split sessions with `clean_len > 4000` chars. Split on `\n\n(?=## )` or `\n\n(?=\[)` boundaries. This prevents very long content from diluting the relevance of shorter, more targeted queries.

### Step 5 — Write with Correct Session IDs

**Critical:** Always read `session_id` from `state.db`, never from LanceDB metadata (which may have been written incorrectly in earlier runs).

```python
# Read session_id directly from state.db
conn = sqlite3.connect(str(STATE_DB))
cur = conn.cursor()
cur.execute("""
    SELECT id, message_count, started_at,
           datetime(started_at, 'unixepoch') as start_dt
    FROM sessions WHERE message_count > 0 ORDER BY started_at
""")
db_sessions = cur.fetchall()
conn.close()
```

### Step 6 — LanceDB Write with Overwrite

```python
db.create_table("memories", schema=schema, mode="overwrite")
# NOT: db.create_table("memories", schema=schema)  # fails if table exists
```

## Running the Optimization

```bash
# Dry-run first
~/.hermes/venv/bin/python3 ~/.hermes/scripts/optimize_<profile>_memory.py --dry-run

# Apply
~/.hermes/venv/bin/python3 ~/.hermes/scripts/optimize_<profile>_memory.py --apply
```

Expected output after optimization:
- Content length reduced 60-77%
- Topic/sentiment/source labels on every record
- `recency_weight` field for re-ranking by date
- Cosine similarity between sessions showing genuine differentiation

## Verification

```python
import lancedb, json
from pathlib import Path

LANCE_DIR = Path.home() / ".hermes/profiles/<profile>/lance_memory"
db = lancedb.connect(str(LANCE_DIR))
table = db.open_table("memories")
rows = table.search([0.0]*1024).limit(10000).to_list()

# Check: bad session_id = "_optimized" strings
bad = [r for r in rows if "_optimized" in r["session_id"]]
print(f"Bad session_ids: {len(bad)}")  # Should be 0

# Check: average length (should be 500-3000 chars, not 4000+)
lens = [len(r["content"]) for r in rows]
print(f"Avg len: {sum(lens)/len(lens):.0f}")  # Should be much lower than raw

# Check: metadata labels
meta_sample = json.loads(rows[0]["metadata"])
print(f"Has topics: {'topics' in meta_sample}")  # Should be True
print(f"Has recency_weight: {'recency_weight' in meta_sample}")  # Should be True
```

## When to Run Optimization

- **Immediately after first migration** — naive migration always produces suboptimal vectors
- **When new sessions are added** — run incremental optimization on new sessions only
- **Periodically** — to re-chunk long sessions that have grown

For ongoing use: modify the migration script to apply content stripping + metadata enrichment at initial migration time, rather than running a separate optimization pass.
