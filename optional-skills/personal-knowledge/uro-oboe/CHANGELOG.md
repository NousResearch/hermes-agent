# Changelog

All notable changes to this project will be documented in this format.

## [0.2.1] - 2024-08-26

### Breaking Changes
- **Skill renamed**: `episodic-memory` → `uro-oboe` (うろ覚え)
- **Tool namespace**: All tools prefixed with `episodic_*` to avoid collisions
  - `memory_store` → `episodic_store`
  - `memory_recall_fuzzy` → `episodic_recall_fuzzy`
  - `memory_fetch` → `episodic_fetch`
  - `memory_search_fts` → `episodic_search_fts`
  - `memory_delete` → `episodic_delete`
  - `memory_stats` → `episodic_stats`
- **Installation path**: `skills/personal-knowledge/episodic-memory/` → `skills/personal-knowledge/uro-oboe/`
- **GitHub repo**: `corekobe/episodic-memory-skill` → `corekobe/uro-oboe`

### Updated
- All documentation (README, guides, SKILL.md) reflects new names
- Distribution package renamed to `uro-oboe.zip`

## [0.2.0] - 2024-08-25

### Added
- **memory_search_fts**: Full-text search using SQLite FTS5 (exact phrases, prefixes, AND/OR/NOT operators)
- **memory_delete**: Soft deletion with logical `is_deleted` flag (preserves FAISS index integrity, excluded from all searches)
- **memory_stats**: System statistics (active/deleted counts, index size, embedding configuration)
- **Auto-tagging**: `memory_store(auto_tag=True)` automatically assigns domain tags from content keywords (11 domains: novel, php, coding, creative, infra, config, cooking, llm, debug, idea, reference)
- **Startup consistency verification**: Automatic rebuild of FAISS index from SQLite (source of truth) on mismatch detection
- **Tag filtering optimization**: Pre-fetched valid ID set for O(1) membership checks during candidate iteration
- **NOVEL_WRITING_GUIDE.md**: Deep-dive guide for novelists (creative DNA accumulation, character voice preservation, noise-driven serendipity)
- **MUSIC_PRODUCTION_GUIDE.md**: Guide for music producers (chord progressions, sound design, arrangement patterns, cross-genre noise discovery)
- **Distribution package**: Ready-to-publish zip with all source, tests, docs, and guides

### Changed
- **Core architecture**: Two-stage retrieval (binary HNSW wide recall → float32/FTS5 verification) now fully implemented
- **Tool interface**: All tools use keyword arguments (not dict) for consistency
- **Persistence**: SQLite schema includes `is_deleted`, `updated_at`, vector BLOB storage
- **Index format**: FAISS IndexBinaryHNSW with M=16, efConstruction=200, efSearch=100
- **Memory count**: Tested with 200+ memories, ~14ms search latency

### Fixed
- FTS5 search test collisions with existing data (now uses unique UUID-prefixed strings)
- Persistence test index rebuild on fresh session
- Tag filtering N+1 query problem (single pre-fetch query)

## [0.1.0] - 2024-08-24

### Added
- Core episodic memory system with 1-bit quantized FAISS binary HNSW
- Three tools: `memory_store`, `memory_recall_fuzzy`, `memory_fetch`
- SQLite with FTS5 virtual table for full-text search
- 1-bit quantization: 384-dim float32 → 384-bit binary via median threshold
- Noise injection parameter (`noise_ratio`) for creative serendipity
- Tag-based filtering at SQL level
- Cross-session persistence (SQLite + FAISS index + mapping)
- 7 verification tests (all passing)
- Basic documentation (README.md, requirements.txt, LICENSE)