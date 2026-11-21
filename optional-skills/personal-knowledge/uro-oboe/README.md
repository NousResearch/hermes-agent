# Uro-oboe Skill

**Uro-oboe (うろ覚え)** — Personal episodic memory with 1-bit quantized FAISS binary HNSW for fuzzy recall.

## Features

- **Embedding**: `all-MiniLM-L6-v2` (384-dim) → 1-bit quantization → Binary HNSW
- **Retrieval**: Two-stage — wide binary recall (Hamming) → precise float32/FTS5 verification
- **Creative noise**: `noise_ratio` parameter (0.0-0.5) intentionally mixes low-score results for serendipity
- **Storage**: SQLite with FTS5 full-text search, tags, metadata, float32 vectors
- **Soft deletion**: Logical `is_deleted` flag preserves index integrity
- **Auto-tagging**: Keyword-based domain tagging (11 domains)
- **Local-first**: No external API calls, all data stays in your SQLite DB

## Tools

| Tool | Purpose |
|------|---------|
| `episodic_store` | Store text with tags, metadata, auto-tagging |
| `episodic_recall_fuzzy` | Fuzzy recall via 1-bit binary HNSW with noise injection |
| `episodic_fetch` | Fetch full records by IDs (vectors, metadata, text) |
| `episodic_search_fts` | Exact full-text search via SQLite FTS5 |
| `episodic_delete` | Soft-delete by IDs (excluded from all searches) |
| `episodic_stats` | System statistics |

## Quick Start

```bash
# Install dependencies
pip install faiss-cpu sentence-transformers numpy

# In Hermes (after placing skill in skills/personal-knowledge/uro-oboe/)
skill_view(name='uro-oboe')

# Store
episodic_store(text="User prefers Japanese responses", tags=["preference", "language"])
episodic_store(text="PHP async processing uses queues", auto_tag=True)  # → tags: ["php", "coding"]

# Fuzzy recall with creative noise
results = episodic_recall_fuzzy(query="testing preferences", k=5, noise_ratio=0.2)

# Fetch full records
memories = episodic_fetch(ids=[r["id"] for r in results["results"]])

# Exact keyword search
fts_results = episodic_search_fts(query="Ollama", k=3)

# Soft delete
episodic_delete(ids=[1, 2, 3])

# Statistics
stats = episodic_stats()
```

## Noise Ratio: The Creative Differentiator

```
noise_ratio=0.0  → Pure relevance (top-k only)
noise_ratio=0.3  → 30% low-score results mixed in (recommended for creative work)
noise_ratio=0.5  → Maximum serendipity
```

**Why it works**: Binary HNSW recall is intentionally coarse. Mixing low-score candidates surfaces:
- Forgotten connections across domains
- Your own past patterns you didn't realize were relevant
- Cross-pollination between code, writing, cooking, etc.

## Tag Filtering

```python
# Filter by tags at SQL level (fast)
episodic_recall_fuzzy(query="debugging", k=10, tags=["coding", "debug"])
episodic_recall_fuzzy(query="plot twist", k=10, tags=["novel", "structure"])
```

## Auto-Tagging Domains

Enable with `auto_tag=True`. Keywords map to:

| Domain | Keywords |
|--------|----------|
| `novel` | chapter, character, plot, scene, protagonist, narrative |
| `php` | php, laravel, symfony, composer, artisan |
| `coding` | function, class, variable, loop, algorithm, refactor |
| `creative` | design, illustration, storyboard, concept, moodboard |
| `infra` | docker, kubernetes, terraform, ansible, ci/cd |
| `config` | yaml, json, toml, ini, config, settings |
| `cooking` | recipe, ingredient, sauce, bake, simmer, season |
| `llm` | prompt, embedding, fine-tune, quantization, context |
| `debug` | error, exception, stack trace, breakpoint, log |
| `idea` | concept, hypothesis, brainstorm, sketch, prototype |
| `reference` | documentation, manual, spec, api, guide, tutorial |

## Persistence

Data lives in `skills/personal-knowledge/episodic-memory/data/`:
- `episodic_memory.db` — SQLite (FTS5, tags, vectors, metadata)
- `episodic_memory_binary.hnsw` — FAISS IndexBinaryHNSW
- `episodic_memory_meta.pkl` — ID ↔ FAISS index mapping

Survives restarts. On startup: auto-verifies consistency (SQLite count = FAISS size = mapping count), rebuilds if needed.

## Use Cases

### 🎭 Novel Writing (Deep Dive)
→ See [NOVEL_WRITING_GUIDE.md](NOVEL_WRITING_GUIDE.md)
- Long-term "creative DNA" accumulation across books/years
- Character voice preservation via vector clustering
- Plot serendipity via `noise_ratio=0.3-0.4`
- Rejected ideas resurfacing as solutions

### 🎵 Music Production
→ See [MUSIC_PRODUCTION_GUIDE.md](MUSIC_PRODUCTION_GUIDE.md)
- Chord progressions, melody motifs, arrangement patterns
- Sound design recipes, mix settings
- Cross-genre noise discovery

### 💻 Code Snippets & Technical Notes
- Language-agnostic, tag-based organization
- Fuzzy recall for "how did I do that thing?"

### 📚 Learning & Study Notes
- Noise injection = spaced repetition + cross-domain connections

### 💼 Sales / Business Memos
- Quick capture, fuzzy recall by context

### 🗂️ General Personal Knowledge Base
- Anything you want to remember and rediscover

## Distribution

Ready-to-publish package: `uro-oboe.zip` (excludes runtime `data/`)

```bash
# Users install by extracting to:
~/.hermes/skills/personal-knowledge/uro-oboe/
# or
~/AppData/Local/hermes/skills/personal-knowledge/uro-oboe/  (Windows)

# Then:
skill_view(name='uro-oboe')
```

## Installation

### From GitHub Releases (recommended)
1. Download `uro-oboe.zip` from [Releases](https://github.com/corekobe/uro-oboe/releases)
2. Extract to your Hermes skills directory:
   - **Windows**: `%LOCALAPPDATA%\hermes\skills\personal-knowledge\uro-oboe\`
   - **macOS/Linux**: `~/.hermes/skills/personal-knowledge/uro-oboe/`
3. In Hermes: `skill_view(name='uro-oboe')`

### From source
```bash
git clone https://github.com/corekobe/uro-oboe.git
# Move to skills directory as above
```

## Requirements

- Python 3.8+
- `faiss-cpu>=1.15.0`
- `sentence-transformers>=6.0.0`
- `numpy>=1.24.0`
- `sqlite3` (stdlib)

## License

MIT — see [LICENSE](LICENSE)

## Changelog

See [CHANGELOG.md](CHANGELOG.md)