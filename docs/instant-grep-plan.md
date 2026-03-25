# Instant Grep: Persistent Search Index for Hermes

## Status

Implemented (session-scoped):
- ④ Result cache in `tools/search_cache.py` — caches ripgrep output, invalidates on write/patch
- ⑤ In-memory trigram index in `tools/search_cache.py` — builds lazily on first search, narrows candidates for subsequent searches

Planned (persistent, cross-session):
- Persistent disk-backed trigram/sparse-ngram index
- Git-anchored index lifecycle
- Incremental updates on file changes

---

## Architecture Overview

```
Session-scoped (done)              Persistent (planned)
┌─────────────────────┐           ┌──────────────────────────┐
│  ResultCache        │           │  DiskIndex               │
│  (exact match,      │           │  (trigram/sparse-ngram,   │
│   120s TTL,         │           │   mmapped lookup table,   │
│   invalidate on     │           │   git-commit anchored,    │
│   write/patch)      │           │   survives across         │
│                     │           │   sessions)               │
├─────────────────────┤           ├──────────────────────────┤
│  TrigramIndex       │           │  DirtyLayer              │
│  (in-memory,        │──evolve──▶│  (uncommitted changes,    │
│   dies with session,│           │   agent writes tracked,   │
│   500+ files to     │           │   merged at query time)   │
│   activate)         │           │                          │
└─────────────────────┘           └──────────────────────────┘
         ↑                                  ↑
    search_tool()                      search_tool()
    in file_tools.py                   in file_tools.py
```

---

## 1. Index Lifecycle

### 1.1 When to build

Build triggers (in priority order):

1. **First search in a git repo with 500+ tracked files.**
   Detect via `git rev-parse --show-toplevel` + `git ls-files | wc -l`.
   If the count exceeds threshold, check for an existing index.

2. **Stale index detected.** The stored git commit doesn't match
   `git rev-parse HEAD`. Rebuild or incrementally update.

3. **Manual trigger.** Future: `hermes index` CLI command for explicit
   rebuild (useful for enterprise users who want it pre-warmed).

### 1.2 Build process

```
① git rev-parse HEAD                    → current_commit
② git rev-parse --show-toplevel         → repo_root
③ hash(repo_root)                       → repo_id (for index path)
④ Check ~/.hermes/indexes/{repo_id}/    → existing index?
⑤ If exists and commit matches          → load from disk, done
⑥ If exists but commit differs          → incremental update (§1.3)
⑦ If not exists                         → full build (§2)
```

### 1.3 Incremental updates (git-anchored)

When the stored commit doesn't match HEAD:

```
① git diff --name-only {stored_commit}..HEAD  → changed_files
② For each changed file:
   - Remove old trigrams from index
   - Re-extract trigrams from current content
   - Update posting lists
③ Store new commit hash
④ Flush updated index to disk
```

If `stored_commit` is unreachable (force push, rebase), fall back to full rebuild.

For uncommitted changes (dirty working tree):

```
① git diff --name-only HEAD               → dirty_files
② git ls-files --others --exclude-standard → untracked_files
③ Merge into a DirtyLayer that overlays the committed index
④ DirtyLayer is session-scoped (not persisted)
```

### 1.4 Agent writes

When Hermes writes or patches a file:
- Add the file path to the DirtyLayer
- On next search, include all DirtyLayer files in candidates
- Don't update the on-disk index (it represents the git state)

### 1.5 Staleness strategy

The index anchors to a git commit. Three tiers of freshness:

| Layer | Freshness | Scope |
|-------|-----------|-------|
| On-disk index | Last git commit at build time | Persistent |
| DirtyLayer | Current working tree diff | Session |
| ResultCache | Exact search results, 120s TTL | Session |

This means the on-disk index can be slightly behind (by a few commits),
but the DirtyLayer compensates. The worst case is a false positive
(trigram index says a file might match when it doesn't) — never a
false negative (missing a real match), because dirty files are always
included.

---

## 2. Index Storage

### 2.1 Location

```
~/.hermes/indexes/
  {repo_id}/
    meta.json          # repo_root, commit, build_time, file_count, version
    lookup.bin         # sorted (hash, offset) pairs — mmapped at query time
    postings.bin       # posting lists (file IDs) — read on demand
    files.json         # file_id → file_path mapping
```

`repo_id` = first 16 chars of SHA-256 of the canonicalized repo root path.

### 2.2 On-disk format

**meta.json:**
```json
{
  "version": 1,
  "repo_root": "/path/to/repo",
  "commit": "abc123...",
  "build_time": "2026-03-24T22:00:00Z",
  "file_count": 12345,
  "trigram_count": 89000,
  "algorithm": "trigram"  // or "sparse-ngram" in v2
}
```

**lookup.bin:**
Fixed-width records, sorted by hash for binary search.
```
┌─────────┬────────┐
│ hash    │ offset │  (4 bytes + 4 bytes = 8 bytes per entry)
│ (u32)   │ (u32)  │
├─────────┼────────┤
│ 0x00012 │ 0      │
│ 0x00020 │ 14     │
│ ...     │ ...    │
└─────────┴────────┘
```

Hash: lower 32 bits of the trigram's hash (FNV-1a or CRC32).
Offset: byte position in postings.bin.

**postings.bin:**
Variable-length posting lists, one after another.
```
┌────────┬───────┬───────┬───────┐
│ count  │ id_0  │ id_1  │ ...   │  (2 bytes count + 2 bytes per file_id)
│ (u16)  │ (u16) │ (u16) │       │
└────────┴───────┴───────┴───────┘
```

File IDs are 16-bit, supporting up to 65,535 files per index.
For repos larger than that, use 32-bit IDs (version flag in meta.json).

**files.json:**
```json
{
  "0": "src/main.py",
  "1": "src/utils.py",
  ...
}
```

### 2.3 Memory footprint

For a 10,000-file repo with ~50,000 unique trigrams:
- lookup.bin: 50,000 × 8 bytes = 400 KB (mmapped, not resident until accessed)
- postings.bin: varies, typically 2-5 MB (read on demand, not mmapped)
- files.json: ~200 KB
- Total resident memory: < 1 MB typically

### 2.4 Query flow

```
① Decompose regex into required trigrams
② For each trigram:
   a. Hash the trigram → u32
   b. Binary search lookup.bin for the hash → offset
   c. Read posting list at offset from postings.bin → set of file_ids
③ Intersect all posting lists → candidate file_ids
④ Map file_ids → file paths via files.json
⑤ Merge with DirtyLayer files
⑥ Pass candidate file list to rg (via --files-from or positional args)
⑦ Return matches
```

### 2.5 Cross-platform considerations (WSL)

- Windows paths (C:\Users\...) and WSL paths (/mnt/c/Users/...) for the
  same repo produce different repo_ids. This is intentional — the file
  system behavior differs between them.
- File I/O across the WSL boundary (/mnt/c/...) is slow. Building the
  index will be slower on WSL for Windows repos, but queries will be
  fast because lookup.bin is mmapped and postings reads are sequential.
- `mmap` works on WSL2 (it's a real Linux kernel).

---

## 3. Implementation Plan

### Phase 1: Disk-backed trigram index (MVP)

**Goal:** Persist the current in-memory TrigramIndex to disk so it
survives across sessions. No incremental updates yet — full rebuild
when commit changes.

Files to create/modify:
- `tools/search_index.py` — new module for DiskIndex
- `tools/search_cache.py` — add DiskIndex integration to SearchAccelerator
- `tools/file_tools.py` — use DiskIndex when available

Steps:
1. Add git commit detection to SearchAccelerator
2. Implement DiskIndex with lookup.bin + postings.bin + files.json
3. Build index in background thread on first search
4. Load existing index from disk if commit matches
5. Full rebuild if commit doesn't match
6. Wire into search_tool() as a third acceleration layer

Estimated effort: 2-3 days

### Phase 2: Incremental updates

**Goal:** When HEAD has changed by a few commits, update the index
incrementally instead of rebuilding from scratch.

Steps:
1. Use `git diff --name-only {old_commit}..{new_commit}` to find changed files
2. Remove old entries and re-index changed files
3. Flush updated index to disk
4. Handle edge cases: force pushes, rebases, unreachable commits

Estimated effort: 1-2 days

### Phase 3: DirtyLayer for working tree changes

**Goal:** Track uncommitted changes and agent writes as an overlay
on the committed index.

Steps:
1. On session start, run `git diff --name-only HEAD` + `git ls-files --others`
2. Build in-memory DirtyLayer from those files
3. Merge DirtyLayer candidates with DiskIndex candidates at query time
4. Agent writes already tracked (from ④/⑤ work)

Estimated effort: 1 day

### Phase 4: Sparse n-grams (v2)

**Goal:** Replace trigram index with sparse n-gram index for better
selectivity and smaller posting lists.

This is the approach from the Cursor blog — using frequency-weighted
hash functions to generate variable-length n-grams that are more
selective than fixed trigrams.

Steps:
1. Build a character-pair frequency table from a large code corpus
   (or ship a pre-computed one)
2. Implement sparse n-gram extraction (build_all for indexing,
   build_covering for querying)
3. Update DiskIndex format to handle variable-length keys
4. Benchmark against trigram index on real repos

Estimated effort: 3-5 days

### Phase 5: CLI integration

**Goal:** `hermes index` command for manual index management.

Commands:
- `hermes index build` — force rebuild
- `hermes index status` — show index info
- `hermes index clear` — delete index
- `hermes index stats` — trigram distribution, posting list sizes

Estimated effort: 1 day

---

## 4. Performance Targets

| Metric | Current (rg) | With index |
|--------|-------------|------------|
| First search (cold) | rg full scan | rg full scan + background index build |
| Second search (same pattern) | rg full scan | Cache hit (0ms) |
| Second search (diff pattern) | rg full scan | Trigram lookup + rg on candidates |
| Search on 10k-file repo | 1-3s | < 200ms (after index built) |
| Search on 50k-file repo | 5-15s | < 500ms (after index built) |
| Index build time (10k files) | N/A | 5-15s (background, one-time) |
| Index disk size (10k files) | N/A | 2-5 MB |
| Memory overhead | 0 | < 1 MB (mmapped lookup table) |

---

## 5. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| False negatives (index misses a file) | Impossible by design — trigrams are necessary conditions, not sufficient. Dirty files always included. |
| Stale index (wrong commit) | Git-anchored lifecycle. DirtyLayer for uncommitted. ResultCache TTL for short-term. |
| Large index build time blocks agent | Background thread. First search goes through rg normally. Index kicks in from second search onward. |
| Memory pressure from mmapped files | Only lookup.bin is mmapped (~400KB for 10k files). Postings read on demand. |
| Cross-platform path issues | Separate repo_ids per platform. Canonical path normalization. |
| Regex decomposition too conservative | Start with literals only. Expand to character classes, alternations in later iterations. Better to have more candidates than miss any. |
