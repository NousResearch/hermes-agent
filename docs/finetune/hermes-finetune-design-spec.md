# Design: Personal Fine-Tuning Pipeline (`hermes-finetune`)

**Status:** Proposal
**Author:** Ivy Darling
**Target:** Hermes Agent v0.3.x
**Category:** `optional-skills/mlops/finetune`

---

## Motivation

Hermes already has a closed learning loop at the prompt layer — skills, memory, Honcho user modeling, and DSPy/GEPA self-evolution all improve what goes into the context window. This proposal adds the missing layer: improving the model weights themselves using the user's own session history.

The pipeline reads from Hermes's existing session DB, scores conversation quality automatically, discovers emergent usage domains via unsupervised clustering, trains QLoRA adapters per domain, and routes incoming prompts to the best adapter at inference time. No manual taxonomy, no external data capture, no fork of core — it ships as an optional skill with minimal core touch points.

### Relationship to Existing Learning Systems

| System | Layer | What it optimizes |
|---|---|---|
| Skills (procedural memory) | Context window | Task-specific procedures via markdown injection |
| Memory (MEMORY.md, USER.md) | Context window | Persistent facts and user preferences |
| Honcho (dialectic modeling) | Context window | Deepening user model across sessions |
| Self-evolution (DSPy/GEPA) | Prompts | Skill files, tool descriptions, system prompt sections |
| Trajectory export / Atropos | Training data | SFT data generation, online RL with environment rewards |
| **This proposal** | **Model weights** | **Offline QLoRA from scored session data with domain-specific adapters** |

These layers stack. A fine-tuned model with good skills and memory outperforms either approach alone.

---

## Architecture Overview

```
~/.hermes/state.db ──► Extract ──► Score ──► Cluster ──► Train ──► Evaluate
        │                                                             │
   (read-only)                                                  ┌─────┴─────┐
                                                                │  promote  │  reject
                                                                ▼           ▼
                                                         Adapter Registry   /dev/null
                                                        (versioned, rollback)
                                                                │
                                                    Inference-time routing
                                                    (provider pre-request hook)
```

Each stage is independently runnable via the `/finetune` skill. A failed or regressed adapter never reaches production — the evaluation gate enforces this.

---

## 1. Data Source: Hermes Session DB

The pipeline does not build its own capture layer. `SessionDB` in `hermes_state.py` already persists every conversation turn in `~/.hermes/state.db` with messages, metadata, tool calls, token counts, and lineage across compression splits. The extractor reads from it directly.

### 1.1 Extraction

The extractor queries `state.db` for sessions matching configurable filters (minimum turn count, source exclusions, date range) and outputs normalized JSONL. It mirrors the core reader's semantics so training data reflects what the user actually saw:

- **Active messages only** (`active = 1`): rewound/retracted turns and pre-compaction originals are soft-deleted rows and never extracted.
- **Insertion order** (`ORDER BY id`, not timestamp): matches core, which orders by AUTOINCREMENT id because wall-clock regressions can reorder timestamps.
- **Multipart content is decoded** from the core's sentinel-prefixed JSON encoding back into a parts list, so downstream flattening drops non-text parts (base64 image blobs never reach training data).
- **Archived sessions are excluded** (`archived = 1`).
- **Delegate subagent sessions are excluded by default** (the `_delegate_from` marker in `model_config`): they are agent-to-agent traffic, not user conversations. Set `finetune.extract.include_delegates: true` to keep them. `/branch` children and compression continuations are real user conversations and are extracted normally.

**Each session is extracted standalone — lineages are never concatenated.** An earlier revision merged compression-split children into their root session; that behavior is **removed**. It was wrong twice over: it followed *every* `parent_session_id` edge (conflating compression continuations with `/branch` children and delegate runs), and a compression child begins with a re-flushed summary plus verbatim copies of retained parent turns, so concatenation duplicated content. Each session row is already a coherent conversation on its own (a compression child's leading summary is valid context). Lineage is retained only as a `root_session_id` field — computed by walking `parent_session_id` edges to the highest reachable ancestor — which the formatter uses to keep a session and its continuations on the same side of the train/eval split (§1.2).

Normalized session format:

```json
{
  "session_id": "uuid",
  "root_session_id": "uuid (== session_id when the session has no ancestors)",
  "started_at": "ISO-8601",
  "turns": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]}
  ],
  "metadata": {
    "source": "cli | telegram | discord | ...",
    "model": "qwen3-8b",
    "parent_session_id": "uuid | null",
    "tool_call_count": 12,
    "total_tokens": 4820
  }
}
```

### 1.2 Formatting

**The atomic unit of training is the assistant turn, not the session.**

This is a deliberate departure from a naive "one record per session" pipeline. Hermes sessions vary by four orders of magnitude in length (observed range: 39 to 414,432 tokens) and almost always contain a mix of high- and low-quality turns. Treating a session as the unit forces a single label across that mix and produces training records that don't fit any reasonable `sequence_len`. The pipeline instead emits **one training record per qualifying assistant turn**, with a sliding window of preceding context.

Each training record is a (context, target) pair:

- **target** = a single assistant turn (the thing the model should learn to produce)
- **context** = the system prompt + up to N preceding turns (default: 8) leading up to the target

A 12-turn session that has 6 assistant responses produces up to 6 training records. A 414K-token session produces dozens of bounded ~1-3K-token records instead of one untrainable monster.

**Selection criteria**: only assistant turns whose **effective per-turn score** meets `finetune.training.min_turn_score` (default: 0.7) are emitted. The effective score is, in priority order:

1. Per-turn retro label (set via `/finetune retro good <id> 2,4`) — explicit ground truth
2. Per-turn automated score (from §2.2 turn-level signals) — heuristic
3. Session-level composite score (from §2.5) — fallback when no per-turn data exists

Bad and neutral turns are skipped entirely, not down-weighted. The trainer only sees turns the user (or the heuristic) considers worth learning from.

**Output format**: ShareGPT-style records compatible with Axolotl's `chat_template` dataset type. Each record's conversation list ends on a `gpt` turn so the trainer has a clear loss target. The shared system prompt and tool-call block formatting reuse Hermes's existing `agent/trajectory.py` normalization, plus reasoning-tag conversion (`<REASONING_SCRATCHPAD>` → `<think>`).

**Train/eval split**: deterministic by `root_session_id` hash, falling back to `session_id` for records without lineage data such as external imports (10–15% held out). All turns from a session — and from its compression/branch continuations, which share verbatim content with it — land in the same split. This prevents context leakage where earlier turns in a lineage would otherwise act as a near-duplicate prompt for held-out turns. Within a split, individual turns from the same session can appear in different orders due to packing, but they never cross the split boundary.

**Why this matters**: with the old session-based approach, a user with 184 conversations would produce maybe 7 training examples that fit a 1024-token window. With the turn-based approach, the same data produces 250–800 training examples depending on the score threshold — a roughly 30–100× increase in usable training signal from identical raw data.

**Configuration knobs** (under `finetune.training` in `config.yaml`):

```yaml
finetune:
  training:
    context_window_turns: 8    # max preceding turns per training example
    min_turn_score: 0.7        # only emit turns scoring at least this
```

Lowering `min_turn_score` produces more examples at lower average quality. Raising it produces fewer but stricter. Lowering `context_window_turns` produces shorter records (faster training, less context-dependent learning); raising it teaches the model to use longer histories at the cost of more VRAM per example.

### 1.3 External Imports (Optional)

Sessions from external sources (e.g., Claude JSON exports, other OpenAI-format conversation logs) can be imported into `~/.hermes/finetune/data/imported/` and normalized to the same intermediate format. These go through the same scoring and clustering pipeline as native Hermes sessions.

---

## 2. Automated Quality Scoring

The goal is to assign a quality signal to each assistant turn without requiring manual labels. The scorer produces a composite score from multiple heuristics.

### 2.1 Conversation-Level Signals

| Signal | Detection | Interpretation |
|---|---|---|
| Abrupt termination | Session ends 1–2 turns after an assistant response with no resolution marker | Likely unhelpful response |
| Retry / rephrase | User's next turn is semantically similar to their previous turn (cosine similarity > threshold) | Model misunderstood or gave inadequate answer |
| Explicit correction | User turn contains correction patterns ("no, I meant...", "that's wrong", "actually...") | Preceding assistant turn is low quality |
| Productive conclusion | Session ends with user acknowledgment, code output, or decision | Preceding turns are likely good |
| Session length vs. complexity | Turn count relative to task complexity (estimated from token count and tool usage) | Very short on complex tasks suggests failure; very long suggests thrashing |

### 2.2 Turn-Level Signals

| Signal | Detection | Interpretation |
|---|---|---|
| Direct affirmation | User responds with "exactly", "perfect", "that's right", or builds directly on the response | High-quality assistant turn |
| Contradiction | User immediately negates or overrides assistant output | Low-quality assistant turn |
| Follow-up depth | User asks a deeper follow-up question (not a correction) | Response was useful enough to build on |
| Artifact adoption | User references or modifies code/text the assistant produced in a later turn | Strong positive signal |

### 2.3 Sentiment Modifier

A lightweight lexicon-based sentiment classifier runs on user turns following each assistant response. Frustration penalizes the preceding turn; satisfaction boosts it. Sentiment is a modifier (±0.1–0.2 on a 0–1 scale), not a standalone filter.

### 2.4 Model-as-Judge (Optional, Bootstrap Phase)

During bootstrap when labeled data is sparse, a stronger model (or the current best local adapter via Hermes's auxiliary model routing) evaluates candidate responses on correctness, relevance, and style match. This can be phased out as heuristic-scored data accumulates.

### 2.5 Composite Score

```
score = w1 * conversation_signal + w2 * turn_signal + w3 * sentiment + w4 * judge_score
```

Default weights: `w1=0.3, w2=0.4, w3=0.1, w4=0.2`. Configurable in `~/.hermes/config.yaml` under `finetune.scoring.weights`.

This composite (and `finetune.scoring.weights`) applies only to the **legacy** scoring mode (`finetune.scoring.mode: legacy`). The default `positive_signals` mode reads its own weight dict, `finetune.scoring.weights_positive` — the two dicts share some key names, so they are kept strictly separate to prevent legacy values from bleeding into the positive-signals math. See the positive-signals spec §"Updated Default Weights" for the default values.

### 2.6 Bucketing

| Bucket | Score | Training action |
|---|---|---|
| **Good** | ≥ 0.7 | Include in training set |
| **Neutral** | 0.4 – 0.7 | Include with reduced sampling weight, or hold for validation |
| **Bad** | < 0.4 | Exclude. Optionally retain as rejected examples for DPO/ORPO |

### 2.7 Manual Overrides (Feedback)

Users can override automated scores via two mechanisms:

**CLI:** Keybindings (`Ctrl+Y` thumbs up, `Ctrl+N` thumbs down) on the last assistant response. Implemented as a post-response callback in `cli.py`, gated by `finetune.feedback.cli_keybindings` config flag.

**Gateway:** Not implemented (removed). No platform adapter emits reaction events into the gateway hook system, and session-level labels from arbitrary chat members are unsafe training signal (a one-emoji poisoning primitive). If platform reaction feedback returns, it needs per-turn attribution and reactor authorship checks first.

Manual flags override the composite score to 1.0 (good) or 0.0 (bad) and are stored in `~/.hermes/finetune/feedback.jsonl`. They also serve as calibration anchors — periodic comparison of automated scores against manual flags tunes the heuristic weights.

---

## 3. Emergent Domain Discovery

### 3.1 Design Principle

Domains are not predefined. The pipeline discovers them from the data. As session volume grows, natural clusters emerge — the system detects them, labels them, and decides when a cluster is dense enough to justify its own adapter. The default path requires zero manual taxonomy work.

### 3.2 Embedding

Every scored session is projected into an embedding space using a small, fast model (e.g., `all-MiniLM-L6-v2` or `nomic-embed-text-v1.5` for fully local inference). Only user turns are embedded to capture intent rather than model behavior.

### 3.3 Clustering

HDBSCAN over the embedding space. HDBSCAN is preferred over k-means because it does not require a predefined cluster count, it identifies noise points rather than forcing bad fits, and it handles varying density.

```python
hdbscan.HDBSCAN(
    min_cluster_size=30,
    min_samples=5,
    metric='euclidean',       # on normalized embeddings, equivalent to cosine
    cluster_selection_method='eom',
    prediction_data=True,
)
```

`min_cluster_size` is the primary granularity lever. Higher produces fewer, broader adapters. Lower produces more specialized ones. Configurable in `finetune.clustering.min_cluster_size`.

### 3.4 Noise Handling

Sessions classified as noise are routed to a `_general` adapter trained on all unaffiliated data.

### 3.5 Cluster Identity

Cluster IDs are content-addressed: a short hash of the centroid embedding (`c-a7f3e2`). If re-clustering produces a cluster whose centroid has cosine similarity > 0.9 to a previous cluster, it inherits the previous ID and adapter lineage. Otherwise it's treated as new.

### 3.6 Auto-Labeling

After clustering, a human-readable label is generated from the top TF-IDF terms across the cluster's sessions. Optionally, the local model can be asked to suggest a short label from the top terms. Labels are for human consumption only — the pipeline uses `cluster_id` internally.

### 3.7 Cluster Maturity & Adapter Lifecycle

| Maturity | Condition | Training behavior |
|---|---|---|
| **Embryonic** | < 50 good-bucket turns | No dedicated adapter. Data folds into `_general`. |
| **Nascent** | 50–150 good-bucket turns | Train with aggressive regularization: `lora_dropout: 0.1`, `learning_rate: 5e-5`, `num_epochs: 2`. |
| **Established** | 150–500 good-bucket turns | Train with standard config. |
| **Mature** | > 500 good-bucket turns | Train normally. Monitor silhouette score — if it drops below 0.3, the cluster may be ready to split. |

Transitions trigger automatic retraining. Clusters can regress if data is deleted or re-scored below threshold.

### 3.8 Session Segmentation (Obsolete — see §1.2)

This section originally proposed splitting high-variance sessions at embedding discontinuities. The turn-based formatting in §1.2 supersedes this entirely: each assistant turn becomes its own training record with its own context window, so within-session topic shifts are handled implicitly. Clustering still operates at the session level for routing purposes, but training granularity is per-turn.

### 3.9 User Overrides (Power User)

All overrides are optional. The default pipeline operates fully autonomously.

| Override | Effect |
|---|---|
| Pin | Prevents cluster dissolution during re-clustering. Sessions excluded from HDBSCAN, routed directly. |
| Merge | Combines two clusters' session pools, retrains a single adapter. |
| Split | Runs HDBSCAN on the cluster's sessions with a lower `min_cluster_size`. |
| Rename | Changes the human-readable label. No effect on routing or training. |
| Force-assign | Moves a session between clusters. |
| Set `min_cluster_size` | Global or per-run granularity override. |

Overrides are persisted in the registry and respected by subsequent re-clustering runs.

### 3.10 Re-clustering Cadence

Re-clustering runs on the same triggers as retraining (§6): 20% data growth or scheduled cadence. Clusters are not assumed stable — they can split, merge, or dissolve as usage patterns evolve.

---

## 4. Training

### 4.1 Base Configuration (Axolotl + QLoRA)

```yaml
base_model: Qwen/Qwen3-8B
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer
chat_template: chatml

load_in_4bit: true
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

sequence_len: 2048
sample_packing: true
pad_to_sequence_len: true

num_epochs: 3
learning_rate: 0.0001
warmup_ratio: 0.05
gradient_accumulation_steps: 4
micro_batch_size: 2

weight_decay: 0.01
max_grad_norm: 1.0

eval_steps: 50
save_steps: 50
save_strategy: steps
logging_steps: 10
early_stopping_patience: 3
metric_for_best_model: eval_loss
greater_is_better: false

bf16: auto
tf32: true
```

The config template lives at `~/.hermes/skills/mlops/finetune/templates/base_qlora.yaml`. A config generator produces per-cluster configs by applying maturity-stage overrides and resolving data/output paths.

The `base_model`, `chat_template`, and all hyperparameters are user-configurable in `~/.hermes/config.yaml` under `finetune.training`. The template above is the default.

### 4.2 Training Execution

Training runs via `accelerate launch -m axolotl.cli.train`, executed through Hermes's terminal backend. This means training can happen locally, over SSH to a GPU box, on Modal, on Daytona — wherever the user's terminal environment is configured. No separate training infrastructure is required beyond what Hermes already supports.

```bash
accelerate launch -m axolotl.cli.train \
    ~/.hermes/finetune/adapters/{cluster_id}/{version}/config.yml
```

### 4.3 Small-Data Considerations

Personal session datasets are typically hundreds to low thousands of examples. Adjustments: learning rate starts at `1e-4` (not `2e-4`), early stopping is enabled by default, and the maturity-stage overrides (§3.7) apply more aggressive regularization to nascent clusters. Data augmentation is not used — the model should learn the user's actual patterns.

### 4.4 Merge & Quantize

After a promoted adapter passes evaluation, it is merged into the base model and re-quantized:

```bash
python -m axolotl.cli.merge_lora config.yml \
    --lora_model_dir ~/.hermes/finetune/adapters/{cluster_id}/{version}

llama-quantize merged_model/ output.gguf Q5_K_M
```

The merged GGUF is tested against the unmerged LoRA-on-base model. If quantization degrades quality beyond tolerance, a higher quantization level (Q6_K, Q8_0) is used, or the adapter is kept unmerged and loaded at runtime via llama.cpp's LoRA loading.

---

## 5. Evaluation & Promotion

### 5.1 Evaluation Harness

Every new adapter must pass evaluation before promotion. The harness runs automatically after training.

**Held-out test set:** Per cluster, 10–15% of *sessions* are reserved at format time (deterministic by `session_id` hash, see §1.2). All turns extracted from those sessions go into the eval split together — turns from the same session never cross the boundary, which prevents context leakage. When a new cluster forms, its held-out set is established automatically.

**Metrics:**

| Metric | Method | Pass threshold |
|---|---|---|
| Perplexity | Eval loss on held-out set | Must not regress > 5% vs. previous adapter |
| Format compliance | Parse assistant outputs for correct ChatML structure | ≥ 95% valid |
| Task completion | Canned prompts checked for expected output patterns | ≥ previous adapter's score |
| A/B preference (optional) | Model-as-judge compares new vs. previous on 20–50 prompts | New preferred ≥ 50% |

### 5.2 Promotion Flow

```
Training complete
    → Run evaluation harness
    → All metrics pass?
        YES → Promote: update symlink, archive previous as rollback target
        NO  → Log failure reason, keep previous adapter active, flag for review
```

### 5.3 Canary Test

A fixed "canary" test set that never changes across adapter versions detects catastrophic forgetting. If canary performance degrades beyond threshold, promotion is blocked regardless of other metrics.

---

## 6. Adapter Registry & Rollback

### 6.1 File Layout

All pipeline state lives under `~/.hermes/finetune/`, following Hermes's convention of keeping user data under `~/.hermes/`.

```
~/.hermes/finetune/
├── config.yaml                   # Pipeline-specific config overrides
├── feedback.jsonl                # Manual thumbs up/down signals
├── data/
│   ├── extracted/                # Normalized JSONL from state.db
│   ├── scored/                   # Quality-annotated sessions
│   └── clusters/                 # Per-cluster training splits
│       ├── _general/
│       │   ├── train.jsonl
│       │   └── eval.jsonl
│       └── {cluster_id}/
│           ├── train.jsonl
│           └── eval.jsonl
├── adapters/
│   ├── registry.json             # Adapter manifest
│   ├── cluster_state.json        # HDBSCAN state, centroids, labels
│   ├── _general/
│   │   ├── v1/
│   │   │   ├── adapter_model/
│   │   │   ├── config.yml
│   │   │   ├── dataset_manifest.json
│   │   │   ├── eval_results.json
│   │   │   └── merged.gguf
│   │   └── active -> v1/
│   └── {cluster_id}/
│       ├── cluster_meta.json
│       ├── v1/
│       ├── v2/
│       └── active -> v2/
├── models/
│   └── merged/                   # Merged + quantized GGUFs
└── logs/                         # Training and eval logs
```

### 6.2 Registry Manifest

```json
{
  "adapters": [
    {
      "cluster_id": "c-a7f3e2",
      "cluster_label": "auto:protocol-specification",
      "version": "v2",
      "status": "active",
      "maturity": "established",
      "base_model_hash": "sha256:...",
      "dataset_version": "2026-04-06",
      "dataset_size": 347,
      "training_config_hash": "sha256:...",
      "eval_results": {
        "perplexity": 2.31,
        "format_compliance": 0.98,
        "task_completion": 0.85
      },
      "promoted_at": "2026-04-06T14:30:00Z",
      "rollback_target": "v1"
    }
  ],
  "clustering": {
    "algorithm": "hdbscan",
    "min_cluster_size": 30,
    "embedding_model": "all-MiniLM-L6-v2",
    "last_run": "2026-04-06T12:00:00Z",
    "total_sessions": 412,
    "clusters_active": 3,
    "noise_sessions": 58
  }
}
```

### 6.3 Rollback

Rollback is a symlink swap. All adapter versions are retained until explicit garbage collection. A minimum of 2 previous versions are always kept.

```bash
cd ~/.hermes/finetune/adapters/{cluster_id}/
rm active && ln -s v1 active
# Restart inference server or reload LoRA
```

Rollback is also exposed via `/finetune rollback {cluster_id}`.

---

## 7. Inference-Time Adapter Routing

### 7.1 Routing Flow

When a prompt arrives at a local model provider:

1. The prompt is embedded using the same model as clustering.
2. The embedding is compared against all active (non-embryonic) cluster centroids via cosine similarity.
3. The highest-similarity cluster's active adapter is loaded, if similarity exceeds the confidence threshold (default: 0.6, configurable via `finetune.clustering.confidence_threshold`).
4. If no cluster exceeds threshold, fall back to `_general` or base model.
5. The routing decision and confidence are logged for re-clustering analysis.

### 7.2 Hermes Integration Point

Routing ships as a standard hermes plugin (`finetune-routing`) registering `llm_request` middleware — no core changes. It only activates when (a) `finetune.routing.enabled` is true, (b) `manage.py redeploy` has written a serving manifest (`<hermes-home>/finetune/serving.json`) describing the llama-server it launched and the adapters that server preloaded, and (c) the request's base_url host matches the manifest server's host (loopback aliases are equivalent; parsed hosts, never substring matching). For cloud providers, routing is a no-op.

### 7.3 llama.cpp Per-Request Adapter Scaling

llama.cpp cannot load an arbitrary adapter path per request. Its per-request API is scale control over adapters **preloaded** via `--lora` at server startup: `"lora": [{"id": N, "scale": s}]`, an extension field accepted through the OpenAI-compatible endpoint, where `id` is the positional index of the `--lora` flag. The middleware therefore sets scale 1.0 on the served adapter whose cluster matches the routed cluster, and scale 0.0 on all served adapters for off-domain prompts — off-domain traffic deliberately falls back to the base model. The serving manifest's adapter list (single entry today) is the extension point for multi-adapter serving.

---

## 8. Skill Interface

The pipeline is exposed as a Hermes optional skill at `optional-skills/mlops/finetune/`. Users interact via slash commands from any platform (CLI, Telegram, Discord, etc.).

### 8.1 Commands

| Command | Action |
|---|---|
| `/finetune status` | Show adapter registry, cluster state, data volume, last training run |
| `/finetune extract` | Extract new sessions from state.db since last extraction |
| `/finetune score` | Run quality scoring on extracted sessions |
| `/finetune cluster` | Run/update domain discovery |
| `/finetune train [cluster]` | Train adapter for a cluster (or all eligible) |
| `/finetune eval [cluster] [version]` | Run evaluation harness |
| `/finetune promote [cluster] [version]` | Promote adapter to active |
| `/finetune rollback [cluster]` | Roll back to previous version |
| `/finetune route "prompt"` | Test which adapter would route for a given prompt |
| `/finetune run` | Full pipeline: extract → score → cluster → train → eval → promote |

### 8.2 SKILL.md

```yaml
---
name: finetune
description: >
  Personal model fine-tuning pipeline. Extract sessions from the Hermes DB,
  score quality, discover usage domains, train QLoRA adapters, and manage
  versioned adapters with rollback.
version: 0.1.0
platforms: [linux]
metadata:
  hermes:
    tags: [mlops, fine-tuning, qlora, training]
    category: mlops
    requires_toolsets: [terminal]
required_environment_variables:
  - name: FINETUNE_BASE_MODEL
    prompt: "Path or HuggingFace ID of the base model for training"
    help: "e.g., Qwen/Qwen3-8B or /path/to/model"
    required_for: training
---
```

### 8.3 Skill Directory Structure

```
optional-skills/mlops/finetune/
├── SKILL.md
├── references/
│   ├── design-spec.md            # This document
│   └── scoring-signals.md        # Detailed signal reference
├── scripts/
│   ├── extract.py
│   ├── score.py
│   ├── cluster.py
│   ├── train.py
│   ├── eval.py
│   └── manage.py
└── templates/
    └── base_qlora.yaml
```

---

## 9. Retraining & Continuous Improvement

### 9.1 Triggers

| Trigger | Action |
|---|---|
| Data growth > 20% of current training set | Queue retraining for affected cluster(s) |
| Scheduled (configurable, default: weekly) | Extract → score → retrain if threshold met |
| Manual | `/finetune run` or `/finetune train` |

### 9.2 Cron Integration

Hermes's built-in cron system handles scheduled retraining as a first-class agent task. The user sets it up conversationally or via `/finetune cron weekly`. The cron job extracts new sessions, scores them, checks data growth per cluster, triggers retraining where warranted, runs evaluation, and reports results via the user's configured delivery platform.

### 9.3 Scorer Calibration

When manual feedback exists, periodic comparison of automated scores against thumbs up/down flags calibrates the heuristic weights. If the scorer consistently disagrees with manual flags, weights are adjusted or new signals added.

---

## 10. Configuration

All pipeline settings live under the `finetune` key in `~/.hermes/config.yaml`:

```yaml
finetune:
  enabled: true

  extract:
    min_turns: 2
    exclude_sources: []            # e.g., ["cron"] to skip automated sessions

  scoring:
    weights:
      conversation_signal: 0.3
      turn_signal: 0.4
      sentiment_modifier: 0.1
      judge_score: 0.2
    thresholds:
      good: 0.7
      neutral: 0.4

  clustering:
    embedding_model: "all-MiniLM-L6-v2"
    min_cluster_size: 30
    confidence_threshold: 0.6

  training:
    base_model: "Qwen/Qwen3-8B"
    chat_template: "chatml"
    quantization: "Q5_K_M"
    terminal_backend: "local"      # or "modal", "ssh", "daytona"

  routing:
    enabled: true

  retraining:
    data_growth_trigger: 0.2
    schedule: "weekly"

  feedback:
    cli_keybindings: true
```

---

## 11. Core PR Surface

The plugin minimizes core changes. For full integration, these touch points need PRs:

| Change | File(s) | Impact |
|---|---|---|
| Config schema: add `finetune` section | `hermes_cli/config.py` | Additive, no breaking changes. `finetune` key is optional and ignored if absent. |
| CLI keybinding: thumbs up/down | `cli.py` | Optional, gated by `finetune.feedback.cli_keybindings`. No-op if finetune is not installed. |
| Provider hook: adapter routing | plugin middleware (`llm_request`) | Ships inside the skill; only activates when `finetune.routing.enabled` is true AND a serving manifest exists (see routing section). |
| Skill bundling | `optional-skills/mlops/finetune/` | Ships as official optional skill. |

Everything else lives in `~/.hermes/finetune/` and the skill directory, requiring zero core modifications for basic functionality. The skill works without any core PRs — the core changes only enable the deeper integrations (feedback capture, inference routing).

---

## 12. Dependencies

Installed by the user when the skill is loaded (not required by Hermes core):

```
sentence-transformers>=2.2    # session embedding
hdbscan>=0.8.33               # clustering
scikit-learn>=1.3             # TF-IDF labeling
axolotl                       # training (on GPU machine only)
accelerate                    # training (on GPU machine only)
```

The clustering and scoring stages can run on CPU. Only training requires GPU access, which can be a remote machine via Hermes's SSH/Modal/Daytona terminal backends.

---

## 13. Open Questions & Future Work

**DPO / ORPO:** The "bad" bucket could serve as rejected examples for preference optimization. This requires paired good/bad responses to the same prompt, which may require generating alternatives.

**Cross-cluster transfer:** Some capabilities (e.g., clear technical writing) span clusters. Investigate adapter stacking — a shared base adapter plus cluster-specific adapters loaded simultaneously.

**Hierarchical clustering:** HDBSCAN produces a natural hierarchy. Future work could allow nested adapters (broad parent + specialized children) as stacked LoRAs.

**Cluster stability:** Early in the pipeline's life, clusters will be volatile. A minimum session age (e.g., ≥ 7 days) before data participates in clustering could reduce thrashing.

**Forgetting:** The canary test set (§5.3) detects catastrophic forgetting, but long-term drift across many fine-tune cycles needs monitoring.

**Privacy:** Session logs may contain sensitive information. Support redaction rules applied during extraction, and allow users to exclude specific sessions or date ranges.

**Cluster visualization:** Expose a UMAP 2D projection of the embedding space with cluster boundaries, so users can visually inspect what the pipeline has learned and spot mis-clustered sessions.

**Atropos convergence:** The quality scorer and Atropos's reward signals serve similar purposes. Investigate unifying them so batch-generated trajectories and interactive sessions share a single quality pipeline.

**Batch runner integration:** `batch_runner.py` output can be fed through the quality scorer directly, creating a unified pipeline where both interactive and synthetic trajectories contribute to adapter training.
