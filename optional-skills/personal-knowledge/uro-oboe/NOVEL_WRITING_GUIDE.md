# Novel Writing Guide

How to use **Uro-oboe** for long-form fiction writing.

## Why This Works for Novelists

Your prose style, character voices, thematic preoccupations, and plot instincts become searchable vectors. Unlike keyword search, **semantic + stylistic similarity** finds what *feels* related.

The `noise_ratio` parameter is the key differentiator: it intentionally mixes low-relevance results, creating **controlled serendipity** — the computational equivalent of "staring at a wall until connections form."

## Recommended Tag Schema

```python
# Work / Series identity (required)
"work:memory_shop"           # Unique work ID
"series:memory_shop"         # Series ID (if multi-book)

# Structural location
"chapter:3"
"scene:7"
"beat:choice"                # hook / inciting / choice / reversal / climax / resolution

# Content classification
"pov:雨宮透"                 # POV character
"type:dialogue"              # dialogue / description / action / internal / exposition
"theme:喪失"                 # Themes: 喪失 / 身分 / 記憶 / 信頼 / 裏切り / etc.

# Character voices (build these up over time)
"character:雨宮透"
"character:七瀬"
"voice"                      # Mark voice samples

# Quality / Status
"status:final"               # final / draft / rejected / cut / outline
"quality:keeper"             # Particularly strong prose — use as reference
```

## Workflows

### Phase 1: Seed Collection (Anytime)
Dump everything without organization:
```python
episodic_store("夢：時計塔の針が逆回転して、過去の自分とすれ違う", ["seed", "image"])
episodic_store("電車で見かけた老夫婦『忘れて正解だったこともある』", ["seed", "dialogue"])
episodic_store("テーマ案：AIが書いた小説を人間が『編集』する職業", ["seed", "concept"])
```

### Phase 2: Plot Building (Focused Sessions)
```python
# Broad search with high noise for unexpected connections
results = episodic_recall_fuzzy(
    query="新作 核心 テーマ 対立",
    k=20,
    noise_ratio=0.35,
    tags=["seed"]
)
```
Review results. Look for: rejected ideas that fit now, thematic echoes across seeds, structural patterns you've used before.

### Phase 3: Drafting (Daily)
```python
# Character voice reference
episodic_recall_fuzzy(
    query="雨宮透 怒り 短い 突き放す",
    k=8,
    noise_ratio=0.2,
    tags=["character:雨宮透", "voice"]
)

# Prose rhythm reference
episodic_recall_fuzzy(
    query="冒頭 フック 感覚 描写",
    k=8,
    noise_ratio=0.25,
    tags=["quality:keeper", "type:description"]
)
```

### Phase 4: Revision (Per Chapter)
```python
# Find your own best openings/endings
episodic_recall_fuzzy(
    query="章 終わり 余韻",
    k=10,
    noise_ratio=0.2,
    tags=["quality:keeper", "beat:resolution"]
)
```

## Long-Term Value (6+ Months)

| Time | What Emerges |
|------|-------------|
| 1 month | Searchable scene database, voice samples |
| 3 months | Structural patterns, recurring motifs visible |
| 6 months | Your "creative fingerprint" — thematic DNA searchable |
| 1 year+ | Cross-series connections, rejected ideas become solutions |

## Pro Tips

1. **Store rejected scenes** with `status:rejected` and `meta:why_rejected` — they often solve future problems
2. **Store "meta" notes** about *why* a choice worked: `episodic_store("第3章の選択シーン：雨の音で始めると読者の呼吸が合う", ["meta", "technique"])`
3. **Full text > summaries** — put actual prose in. 200-800 chars per entry (scene-level chunking)
4. **Weekly noise browse**: `episodic_recall_fuzzy("今の作品 核心", k=15, noise_ratio=0.3, tags=["work:current"])`

## Example: Character Voice Building

```python
# Build voice profile over time (10-20 entries per character)
episodic_store("「記憶なんて、売らなきゃよかったって後悔するもんじゃない。買わなきゃよかったって後悔するもんだ」", ["character:雨宮透", "voice", "philosophy"])
episodic_store("「──黙れ。俺は依頼人の記憶を預かる。お前の説教じゃない」", ["character:雨宮透", "voice", "short", "cold"])
episodic_store("「忘れることが癒やしだと思ってる。違うんだ。忘れることこそが、傷なんだ」", ["character:雨宮透", "voice", "internal"])

# Later: "write like 雨宮透"
results = episodic_recall_fuzzy({"query": "雨宮透 声 口調", "k": 10, "tags": ["character:雨宮透", "voice"]})
```