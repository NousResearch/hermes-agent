# Implementation Tasks: `hermes-finetune`

**Branch:** `feat/finetune`
**Specs:** [design-spec](hermes-finetune-design-spec.md) | [bench-spec](hermes-finetune-bench-spec.md)

---

## Phase 1: Skill Skeleton & Data Pipeline

### 1.1 Create optional skill directory structure

- Create `optional-skills/mlops/finetune/` with:
  - `SKILL.md` — frontmatter from design spec (name, description, version, platforms, required_environment_variables, metadata with tags/requires_toolsets)
  - `scripts/` — empty Python modules for each pipeline stage
  - `templates/base_qlora.yaml` — default Axolotl QLoRA config from design spec
  - `references/` — symlink or copy the design spec and bench spec
- Follows the structure documented in CONTRIBUTING.md under "Adding a Skill" and observed in existing `optional-skills/` entries

### 1.2 Session extractor (`scripts/extract.py`)

- Read from `~/.hermes/state.db` using the schema in `hermes_state.py` (tables: `sessions`, `messages`)
- Query sessions matching configurable filters: minimum turn count, source exclusions, date range
- Reconstruct compression-split conversations via `parent_session_id` lineage
- Output normalized JSONL to `~/.hermes/finetune/data/extracted/` matching the format in design spec (session_id, started_at, turns, metadata)
- Support incremental extraction (track last extraction timestamp to avoid re-processing)
- Support external imports from `~/.hermes/finetune/data/imported/`

### 1.3 Training data formatter (`scripts/format.py`)

- Convert extracted JSONL to Axolotl `chat_template` dataset format
- **Per-turn extraction**: emit one training record per qualifying assistant turn, not one per session. Each record is a (context, target) pair where context = system prompt + sliding window of preceding turns and target = a single assistant turn the model should learn to produce. See design spec §1.2 for the rationale.
- Filter by per-turn effective score against `min_turn_score` (default 0.7). The effective score consults: (1) per-turn retro labels first, (2) automated per-turn scores second, (3) session-level composite as a fallback.
- Reuse reasoning markup normalization from `agent/trajectory.py` (`convert_scratchpad_to_think`)
- Ensure correct ChatML tokenization (`<|im_start|>` / `<|im_end|>`) and tool-call block formatting
- Canonicalize system prompts (strip ephemeral injections, normalize skill sections)
- Deterministic train/eval split by session ID hash (10-15% held out). All turns from a single session land in the same split — no context leakage across the boundary.
- Each training example must end on a `gpt` turn so axolotl has a clear loss target

### 1.4 Create `~/.hermes/finetune/` directory layout

- On first run, create the full directory tree from design spec: `data/{extracted,scored,clusters}`, `adapters/`, `models/merged/`, `logs/`, `bench/`
- Initialize `adapters/registry.json` with empty manifest
- Initialize `adapters/cluster_state.json`

---

## Phase 2: Quality Scoring

### 2.1 Conversation-level signal detection (`scripts/score.py`)

- Abrupt termination detection (session ends 1-2 turns after assistant with no resolution)
- Retry/rephrase detection (cosine similarity between consecutive user turns > threshold)
- Explicit correction pattern matching ("no, I meant...", "that's wrong", "actually...")
- Productive conclusion detection (user acknowledgment, code output, decision)
- Session length vs. complexity heuristic (turn count relative to token count and tool usage)

### 2.2 Turn-level signal detection

- Direct affirmation matching ("exactly", "perfect", "that's right", or user building on response)
- Contradiction detection (user negates or overrides assistant output)
- Follow-up depth classification (deeper follow-up vs. correction)
- Artifact adoption detection (user references or modifies assistant-produced code/text later)

### 2.3 Sentiment modifier

- Lightweight lexicon-based sentiment classifier on user turns following assistant responses
- Modifier range: +/-0.1-0.2 on the 0-1 scale

### 2.4 Composite scoring and bucketing

- Weighted composite: `w1*conversation + w2*turn + w3*sentiment + w4*judge`
- Default weights: 0.3, 0.4, 0.1, 0.2
- Bucket thresholds: good >= 0.7, neutral 0.4-0.7, bad < 0.4
- Output scored sessions to `~/.hermes/finetune/data/scored/`

### 2.5 Model-as-judge (optional bootstrap)

- Use auxiliary model (via `agent/auxiliary_client.py` pattern) to evaluate candidate responses
- Score on correctness, relevance, and style match
- Weight configurable, can be phased out as heuristic data accumulates

---

## Phase 3: Domain Discovery

### 3.1 Session embedding (`scripts/cluster.py`)

- Embed user turns using `sentence-transformers` (default: `all-MiniLM-L6-v2`)
- Only embed user turns to capture intent, not model behavior
- Cache embeddings to avoid re-computation

### 3.2 HDBSCAN clustering

- Run HDBSCAN with configurable `min_cluster_size` (default: 30)
- Route noise points to `_general` adapter pool
- Content-addressed cluster IDs from centroid embedding hash (`c-XXXXXX`)
- Centroid similarity matching (cosine > 0.9) to inherit previous cluster ID and adapter lineage on re-clustering

### 3.3 Cluster labeling and metadata

- Auto-label clusters from top TF-IDF terms (scikit-learn)
- Optional: local model suggestion for human-readable labels
- Persist cluster state to `adapters/cluster_state.json`
- Track cluster maturity (embryonic / nascent / established / mature) based on good-bucket turn counts

### 3.4 Per-cluster data splitting

- Write per-cluster train/eval JSONL to `~/.hermes/finetune/data/clusters/{cluster_id}/`
- Include `_general` cluster for noise sessions

### 3.5 User overrides

- Pin, merge, split, rename, force-assign operations
- Persist overrides in registry, respect during re-clustering

---

## Phase 4: Training

### 4.1 Config generator (`scripts/train.py`)

- Generate per-cluster Axolotl config from `templates/base_qlora.yaml`
- Apply maturity-stage overrides (nascent: higher dropout, lower lr, fewer epochs)
- Resolve data paths, output paths, base model
- Write config to `adapters/{cluster_id}/{version}/config.yml`

### 4.2 Training execution

- Launch training via `accelerate launch -m axolotl.cli.train` through Hermes terminal tool
- Support local, SSH, Modal, Daytona backends (whatever the user's terminal environment is configured for)
- Log training output to `~/.hermes/finetune/logs/`

### 4.3 Merge and quantize

- Merge LoRA adapter into base model via `axolotl.cli.merge_lora`
- Quantize with `llama-quantize` (default: Q5_K_M, configurable)
- Fall back to higher quant level or unmerged LoRA if quantization degrades quality
- Output merged GGUF to `adapters/{cluster_id}/{version}/merged.gguf`

---

## Phase 5: Evaluation Benchmark

### 5.1 Benchmark environment (`environments/benchmarks/finetune_bench/`)

- Create `finetune_bench_env.py` subclassing `HermesAgentBaseEnv` from `environments/hermes_base_env.py`
- Implement `setup()`, `get_next_item()`, `format_prompt()`, `compute_reward()`, `evaluate()`
- `CaseResult` and `FinetuneBenchConfig` dataclasses per bench spec

### 5.2 Prompt bank

- Create `prompt_bank.yaml` with ~220 test cases across three tiers:
  - Tier 1: Tool selection (~100 cases)
  - Tier 2: Tool execution quality (~80 cases)
  - Tier 3: End-to-end task completion (~40 cases)
- Designate 20-30 cases as canary set (frozen across versions)
- Support custom cases from `~/.hermes/finetune/bench/custom/`

### 5.3 Scoring implementation

- Tier 1: Exact match on tool name, partial credit for wrong args
- Tier 2: `ToolContext` sandbox verification of actual outcomes
- Tier 3: Functional test via test commands in sandbox, optional judge model for subjective quality
- Format compliance checks (valid ChatML, parseable tool call JSON)
- Hallucination detection (non-existent tool calls)

### 5.4 Comparison report

- Aggregate metrics: tool selection accuracy, execution success, task completion rate, format compliance, no-tool accuracy, hallucination rate, mean turns, mean errors, canary pass rate
- Baseline vs. candidate comparison with delta and pass/fail per metric
- Regression thresholds from bench spec (3% tool selection, 5% execution/completion, 95% format, 0% hallucination)
- Formatted table output (ASCII box drawing as shown in bench spec)

### 5.5 Promotion gate (`scripts/eval.py`)

- Run benchmark against candidate adapter
- Compare to baseline (or run baseline first if none exists)
- Pass/fail verdict drives promotion decision
- Store results at `~/.hermes/finetune/bench/results/`

---

## Phase 6: Adapter Registry & Rollback

### 6.1 Registry management (`scripts/manage.py`)

- CRUD operations on `adapters/registry.json`
- Version tracking per cluster (v1, v2, ...)
- Active symlink management (`active -> vN/`)
- Record eval results, dataset version, training config hash, base model hash

### 6.2 Promotion flow

- Training complete -> run eval -> all metrics pass -> update symlink, archive previous
- On failure: log reason, keep previous active, flag for review

### 6.3 Rollback

- Symlink swap to previous version
- Retain minimum 2 previous versions before garbage collection
- Expose via `/finetune rollback {cluster_id}`

### 6.4 Status reporting

- `/finetune status`: show adapter registry, cluster state, data volume, last training, last eval scores

---

## Phase 7: Inference-Time Routing

### 7.1 Prompt router

- Embed incoming prompt using same model as clustering
- Compare against active cluster centroids (cosine similarity)
- Select highest-similarity cluster's adapter if above confidence threshold (default: 0.6)
- Fall back to `_general` or base model below threshold
- Log routing decisions for re-clustering analysis

### 7.2 Routing plugin (llm_request middleware)

- Ship a `finetune-routing` plugin inside the skill (`plugin/finetune-routing/`); `/finetune route enable` installs it into `<hermes-home>/plugins/` where standard plugin discovery loads it
- Register `llm_request` middleware that rewrites the request payload per call (request-local — no env vars or process-global state)
- Only activate when provider is local (llama.cpp, custom endpoint)
- No-op for cloud providers (OpenRouter, Nous Portal, OpenAI)
- Pass selected adapter path to llama.cpp server

### 7.3 llama.cpp LoRA hot-swap

- Use llama.cpp runtime LoRA loading (no base model reload)
- If hot-swap unavailable, fall back to serving merged GGUF for most-used cluster
- Detect llama.cpp server capabilities at startup

---

## Phase 8: Core Integration PRs

These are optional and additive. The skill works without them.

### 8.1 Config schema extension

- **File:** `hermes_cli/config.py` — add `finetune` key to `DEFAULT_CONFIG` (line ~201+)
- All sub-keys from design spec: `enabled`, `extract`, `scoring`, `clustering`, `training`, `routing`, `retraining`, `feedback`
- Optional key, ignored if absent, no breaking changes

### 8.2 CLI feedback keybindings

- **File:** `cli.py` — extend `_register_extra_tui_keybindings()` (line ~6826)
- `Ctrl+Y` thumbs up, `Ctrl+N` thumbs down on last assistant response
- Gated by `finetune.feedback.cli_keybindings` config flag
- Write feedback to `~/.hermes/finetune/feedback.jsonl`

### 8.3 Gateway reaction hook

- **File:** `gateway/hooks.py` — register reaction handler for emoji feedback
- Map `thumbs_up`/`thumbs_down` reactions to feedback signals
- Gated by `finetune.feedback.gateway_reactions` config flag
- Create hook in `~/.hermes/hooks/finetune-feedback/` with `HOOK.yaml` + `handler.py`

### 8.4 Slash command registration

- **File:** `hermes_cli/commands.py` — add `CommandDef` to `COMMAND_REGISTRY` (line ~27)
- Name: `finetune`, category: `Tools`, subcommands: `(status, extract, score, cluster, train, eval, promote, rollback, route, run, cron)`
- Not `cli_only` or `gateway_only` — available on all platforms

---

## Phase 9: Cron & Continuous Improvement

### 9.1 Scheduled retraining

- Integrate with Hermes cron system (`tools/cronjob_tools.py`)
- `/finetune cron weekly` sets up recurring: extract -> score -> check data growth -> retrain if threshold met -> eval -> report
- Data growth trigger: retrain when new data exceeds 20% of current training set

### 9.2 Scorer calibration

- Compare automated scores against manual feedback anchors
- Adjust heuristic weights when scorer consistently disagrees with manual flags

---

## Phase 10: Testing

### 10.1 Unit tests

- `tests/test_finetune_extract.py` — extraction from mock state.db, lineage reconstruction, incremental extraction
- `tests/test_finetune_score.py` — each signal detector, composite scoring, bucketing
- `tests/test_finetune_cluster.py` — clustering with synthetic embeddings, cluster ID stability, maturity transitions
- `tests/test_finetune_format.py` — ChatML formatting, trajectory normalization, train/eval splits
- `tests/test_finetune_manage.py` — registry CRUD, symlink management, rollback
- `tests/test_finetune_route.py` — routing logic, confidence thresholds, provider gating
- `tests/test_finetune_bench.py` — scoring per tier, comparison report, promotion verdict

### 10.2 Integration tests

- End-to-end pipeline with small synthetic dataset: extract -> score -> cluster -> train config generation -> eval
- Benchmark environment runs against mock agent responses

### 10.3 Manual testing

- `hermes --toolsets skills -q "Use the finetune skill to show status"` — verify skill loads and responds
- Test each `/finetune` subcommand against real session data
- Test on Linux (primary target per `platforms: [linux]` in SKILL.md)

---

## Dependency Summary

| Dependency | Required by | Install context |
|---|---|---|
| `sentence-transformers>=2.2` | Embedding (Phase 3) | Skill load time |
| `hdbscan>=0.8.33` | Clustering (Phase 3) | Skill load time |
| `scikit-learn>=1.3` | TF-IDF labeling (Phase 3) | Skill load time |
| `axolotl` | Training (Phase 4) | GPU machine only |
| `accelerate` | Training (Phase 4) | GPU machine only |

None of these are Hermes core dependencies. They install when the skill is activated.

---

## Suggested Implementation Order

Phases 1-2 are independent groundwork. Phase 3 depends on 2. Phase 4 depends on 1 and 3. Phase 5 can start in parallel with Phase 3. Phase 6 depends on 4 and 5. Phase 7 depends on 6. Phase 8 can be done at any point. Phase 9 depends on the full pipeline. Phase 10 should run continuously alongside development.

Critical path: **1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7**
