# Music Production Guide

How to use **Uro-oboe** for music production, composition, and sound design.

## Why This Works for Music Makers

Musical ideas are fragmentary, non-linear, and often expressed in vague sensory terms ("warm," "driving," "floating"). Vector search excels at **semantic + sensory similarity** — finding ideas that *feel* related even when keywords differ.

`noise_ratio` enables **cross-genre serendipity**: a "melancholic drop" query might surface a cooking note about "slow-simmered broth" or a novel note about "delayed payoff" — sparking genuinely new arrangement approaches.

## Recommended Tag Schema

```python
# Project / Track identity
"project:album_2025"
"track:03"
"status:demo"                # sketch / demo / wip / mix / master / released

# Musical attributes
"key:F#m"
"bpm:128"
"time:4/4"
"genre:melodic-house"

# Structural location
"part:intro"                 # intro / verse / pre-chorus / chorus / drop / bridge / outro
"section:A"                  # For A-B-A form labeling

# Element / Role
"element:drums"              # drums / bass / lead / pad / vocal / fx / atmosphere
"role:hook"                  # hook / groove / texture / transition / fill

# Content type
"type:chords"                # chords / melody / lyrics / rhythm / structure / sound-design / mixing
"lyrics:chorus"              # For vocal parts

# Quality / Reference
"quality:keeper"             # Successful technique/result to reuse
"quality:reference"          # Reference track analysis
"quality:discard"            # Failed experiment (with meta:why)
```

## Workflows

### Phase 1: Sketch & Ideation
```python
# Capture fragments immediately
episodic_store("3拍子ハウス、キックだけ4つ打ち、ハイハットは3連符", ["project:new", "idea", "rhythm", "genre:house"])
episodic_store("歌詞：君の影を追いかけて / 光のない部屋で / 足音だけが答え", ["project:new", "lyrics", "verse", "theme:喪失"])
episodic_store("コード進行：F#m - D - A - E / Bm - F#m - D - C#", ["project:new", "chords", "progression"])
```

### Phase 2: Arrangement & Structure
```python
# Stuck on bridge? High noise for cross-genre ideas
results = episodic_recall_fuzzy(
    query="ブリッジ 展開 意外性 感情",
    k=10,
    noise_ratio=0.35,
    tags=["structure", "arrangement"]
)
# May surface: novel plot twist technique, cooking "resting time" concept, etc.
```

### Phase 3: Sound Design & Mixing
```python
# "Bass gets buried" → past successes + noise
results = episodic_recall_fuzzy(
    query="ベース 埋もれない EQ コンプ サイドチェイン",
    k=8,
    noise_ratio=0.2,
    tags=["mixing", "bass", "quality:keeper"]
)

# Synth patch recreation
episodic_store("Saw波 4オシレーター + FM比率2:1 + フィルター24dB オートメーション", ["sound-design", "synth", "lead", "quality:keeper"])
```

### Phase 4: Reference Analysis
```python
# Analyze reference tracks
episodic_store("Flume - Never Be Like You ドロップ: サブベースのみ4小節→フルミックスイン、ハイパスオートメーション", ["reference", "artist:Flume", "genre:future-bass", "structure:drop"])
episodic_store("シティポップ コーラス: 3度上下ハモリ + シンセパッドで厚み", ["reference", "genre:city-pop", "arrangement", "vocal"])

# Later: "Flume風ドロップどう作ったっけ"
episodic_recall_fuzzy({"query": "Flume ドロップ 構成", "k": 5, "tags": ["reference", "artist:Flume"]})
```

### Phase 5: Project Archive (Post-Release)
```python
# Complete track documentation
episodic_store(full_track_notes, [
    "project:album_2025", "track:03", "status:master",
    "key:F#m", "bpm:128", "genre:melodic-house",
    "quality:keeper"
])
```

## Long-Term Value

| Time | Musical DNA Accumulated |
|------|------------------------|
| 1 month | Personal chord vocabulary, go-to progressions |
| 3 months | Signature sound design techniques, mixing templates |
| 6 months | Cross-genre structural instincts, arrangement patterns |
| 1 year+ | Evolving artistic fingerprint — your "sound" becomes queryable |

## Pro Tips

1. **Store failed mixes** with `quality:discard` and `meta:why` — negative examples are valuable
2. **Chunk by musical idea**, not time: one entry per chord progression, melody motif, or mix decision
3. **Weekly noise session**: `episodic_recall_fuzzy("今の曲 核心 雰囲気", k=15, noise_ratio=0.3, tags=["project:current"])`
4. **Separate "reference" from "own work"**: `tags: ["reference"]` vs `tags: ["project:X"]` — enables both "how did I do it" and "how did they do it"

## Example: Building a Personal "Sound"

```python
# Over months, accumulate your signature techniques
episodic_store("マスターバス: SSL G-Bus コンプ 1.5:1 アタック30ms リリズオート + リミッター -0.3dB", ["mixing", "master-bus", "quality:keeper"])
episodic_store("ボーカル: U87 近接5cm + dbx 160A 4:1 3dB GR + Pultec EQP-1A 低域+2dB 高域+3dB", ["recording", "vocal-chain", "quality:keeper"])
episodic_store("シンセベース: Monologue ノコギリ波 オクターブ下 + ディケイ短め + ドライブ少し", ["sound-design", "bass", "analog", "quality:keeper"])

# Later: "自分の音でベース作る"
results = episodic_recall_fuzzy({"query": "ベース 自分の音 厚み", "k": 8, "tags": ["sound-design", "bass", "quality:keeper"]})
```