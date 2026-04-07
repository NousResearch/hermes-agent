---
name: finetune
description: >
  Personal model fine-tuning pipeline. Extract sessions from the Hermes DB,
  score quality, discover usage domains, train QLoRA adapters, and manage
  versioned adapters with rollback.
version: 0.1.0
author: Ivy Darling
license: MIT
platforms: [linux]
required_environment_variables:
  - name: FINETUNE_BASE_MODEL
    prompt: "Path or HuggingFace ID of the base model for training"
    help: "Defaults to ~/programs/carnice/Carnice-9b-Q8_0.gguf (Qwen-based). Override to use a different base."
    required_for: training
prerequisites:
  commands: [accelerate, python]
metadata:
  hermes:
    tags: [mlops, fine-tuning, qlora, training, adapters, personalization]
    category: mlops
    requires_toolsets: [terminal]
---

# Fine-Tune: Personal Model Training Pipeline

Train QLoRA adapters from your own Hermes session history. The pipeline extracts conversations, scores quality automatically, discovers usage domains via clustering, trains per-domain adapters, and routes inference to the best adapter.

## When to Use

- User asks to fine-tune, train, or personalize their model
- User wants to improve model performance on their specific tasks
- User asks about adapter management, training status, or rollback
- User wants to score or review session quality
- Trigger phrases: "fine-tune", "train adapter", "finetune status", "model training"

## Quick Reference

| Command | Action |
|---|---|
| `/finetune status` | Show adapter registry, cluster state, data volume |
| `/finetune extract` | Extract new sessions from state.db |
| `/finetune score` | Run quality scoring on extracted sessions |
| `/finetune cluster` | Run/update domain discovery |
| `/finetune train [cluster]` | Train adapter for a cluster (or all eligible) |
| `/finetune eval [cluster] [version]` | Lightweight perplexity gate from training metrics |
| `/finetune bench` | **Run the full 243-case benchmark against the active model** |
| `/finetune retro list` | **Show priority queue of unlabeled sessions** |
| `/finetune retro show <id>` | **Inspect a session's full conversation** |
| `/finetune retro good <id> [turns]` | **Label session or specific turns as good** |
| `/finetune retro bad <id> [turns]` | **Label session or specific turns as bad** |
| `/finetune retro skip <id>` | **Drop a session out of the queue** |
| `/finetune retro stats` | **Show labeling progress** |
| `/finetune promote [cluster] [version]` | Promote adapter to active |
| `/finetune rollback [cluster]` | Roll back to previous version |
| `/finetune route "prompt"` | Test which adapter would route for a prompt |
| `/finetune run` | **Full pipeline, no bench gate** — fast, auto-promotes |
| `/finetune run --with-bench` | **Full pipeline + bench gate** — auto-rollback on regression |
| `/finetune cron` | Schedule recurring retraining |

## Two Workflows

The pipeline supports two retraining modes. Pick the one that matches your appetite for speed vs. safety.

### Workflow A — Fast pipeline (no bench gate)

```
/finetune run
```

**What it does, in order:**
1. **Extract** new sessions from `~/.hermes/state.db` since the last run
2. **Score** every session with the heuristic quality scorer
3. **Cluster** sessions into usage domains via HDBSCAN
4. **Train** a QLoRA adapter for each eligible (non-embryonic) cluster
5. **Register** each new adapter version in `adapters/registry.json`
6. **Promote** each new adapter to active immediately (no verification)

**Trade-offs:**
- Fast — no benchmark to wait for
- Lightweight perplexity check from training metrics is the only gate
- A bad adapter goes live without warning. If quality regresses, you have to notice and run `/finetune rollback <cluster>` manually
- Best for early iteration where you're tuning hyperparameters and re-running often

**After running, it's your responsibility to verify quality:**
```
/finetune bench           # run the 243-case benchmark
/finetune rollback c-id   # if it regressed
```

### Workflow B — Safe pipeline (with bench gate)

```
/finetune run --with-bench
```

**What it does, in order:**
1. Steps 1–5 from Workflow A (extract, score, cluster, train, register)
2. **Promote** each new adapter to active
3. **Run the benchmark** automatically (243 prompt cases against the now-active model)
4. **Compare** the new bench result against the most recent prior bench result
5. If regressed beyond the thresholds → **automatically rollback** every adapter promoted in step 2 and surface the regression report

**Trade-offs:**
- Safe — a regressing adapter never stays active for long
- Slow — the bench takes 20–35 min on top of the training time
- Requires a prior bench result to compare against. If none exists, the new result is recorded as the new baseline and the gate is a no-op
- Best for scheduled retraining (cron) and any retrain you don't intend to babysit

**Regression thresholds** (defined in the bench env config):
- `tool_selection_accuracy`: must not drop more than 3%
- `tool_execution_success`: must not drop more than 5%
- `task_completion_rate`: must not drop more than 5%
- `format_compliance`: must remain ≥ 95%
- `hallucination_rate`: must remain 0%
- `canary_pass_rate`: must not drop more than 5%

If **any** of these fails, all adapters promoted in this run are rolled back automatically.

## Choosing a workflow

| Situation | Recommended |
|---|---|
| First time setting up the pipeline | A — get a baseline first |
| Iterating on hyperparameters | A — fast feedback loops |
| Weekly cron retraining | **B — never wake up to a broken model** |
| Production deployment | **B — gate everything** |
| Debugging why a specific adapter is bad | A — promote, inspect, rollback by hand |

## Benchmark Standalone

Run the benchmark against whatever's currently active (without retraining):

```
/finetune bench
```

This is useful for:
- Establishing a baseline before your first retraining run
- Spot-checking the active model after manual changes
- Investigating a regression flagged by Workflow B

Results land in `~/.hermes/finetune/bench/results/bench_<timestamp>.json` and the most recent prior result is used as the comparison baseline automatically.

## Retroactive Labeling

The automated quality scorer is conservative — early in the pipeline you'll likely see most sessions land in the **neutral** bucket (composite 0.4–0.7), which means they don't contribute to the **good** bucket the trainer needs. The retro flow lets you go back and label historical sessions and turns by hand, seeding real positive/negative signal.

### When to use retro

- After your first `/finetune extract` and `/finetune score` run, when the heuristic scorer hasn't found enough "good" sessions to train on
- Periodically as new sessions accumulate, to maintain fresh ground-truth signal
- After a regression, to label the bad cases that the scorer missed

### The flow

Retro is **stateless and command-driven** — each command does one thing and returns. State lives in `~/.hermes/finetune/feedback.jsonl`. A typical session looks like:

```
/finetune retro list             ← see the top-priority queue
/finetune retro show abc12       ← inspect a session
/finetune retro good abc12 2,4   ← label specific turns good
/finetune retro list             ← queue refreshes, abc12 is gone
/finetune retro show def34
/finetune retro bad def34 1      ← label one turn bad
/finetune retro skip ghi56       ← drop a session you don't want to review
/finetune retro stats            ← see your progress
```

You can come back to retro at any time. Labels persist until you delete `feedback.jsonl`.

### Priority ranking

The queue is **not chronological**. Sessions are ranked by how much a human label would improve training data quality. The top-of-queue session is the one your label would help the most. The ranking factors:

| Factor | Weight | Why |
|---|---|---|
| **Uncertainty** | 50% | Sessions in the neutral zone (composite ≈ 0.5) benefit most from a human verdict. The scorer already knows what to do with 0.95 or 0.05. |
| **Tool density** | 25% | Sessions with more tool calls matter more — tool-calling behavior is what the fine-tune most directly affects. |
| **Recency** | 15% | Recent sessions rank higher (14-day half-life) because you'll remember the context better. |
| **Turn count** | 10% | Longer sessions yield more labeled turns per minute of review. |

### Turn-level labels

Most sessions contain a mix of quality. A 12-turn session might have 8 great turns and 4 mediocre ones. Training on all 12 dilutes the signal. Use the turn syntax to target specific turns:

| Syntax | Meaning |
|---|---|
| `/finetune retro good abc12` | Label all assistant turns in this session as good |
| `/finetune retro good abc12 3` | Label only assistant turn 3 as good |
| `/finetune retro good abc12 2,4` | Label turns 2 and 4 as good |
| `/finetune retro good abc12 1-5` | Label turns 1 through 5 (inclusive) as good |
| `/finetune retro good abc12 1,3-5,8` | Mixed: turn 1, turns 3-5, turn 8 |
| `/finetune retro bad abc12 7` | Label turn 7 as bad — other turns keep automated scores |

Turn numbers are 1-based and count only assistant turns (system, user, and tool messages don't count). Run `/finetune retro show <id>` to see the numbered turns before labeling.

### How retro labels affect training

After labeling, the next `/finetune score` run honors your labels. The scorer applies them in priority order:

1. **Per-turn retro labels** — highest priority. Override the automated turn score directly.
2. **Session-level retro labels** — apply to every assistant turn that doesn't have a turn-level label.
3. **In-the-moment feedback** — `Ctrl+Y`/`Ctrl+N` from the CLI, gateway emoji reactions.
4. **Automated heuristic** — falls through if nothing above applies.

You can use `/finetune retro` independently of the rest of the pipeline. The labels just sit in `feedback.jsonl` until the next `/finetune score` consumes them.

### Tip: prefix matching

Session IDs are long. Every retro command accepts a prefix as long as it's unambiguous:

```
/finetune retro show abc       ← finds 'abc12345_xyz...' if it's the only match
/finetune retro show ab        ← errors with the list of matches if 'abc...' and 'abe...' both exist
```

## Individual Stages

Every step from Workflow A can be invoked on its own. This is mainly for debugging — for normal use, prefer `/finetune run` or `/finetune run --with-bench`.

| Subcommand | What it does |
|---|---|
| `/finetune extract` | Pull new sessions from `state.db` |
| `/finetune score` | Run quality scoring on previously-extracted sessions |
| `/finetune cluster` | Discover domains via HDBSCAN |
| `/finetune train` | Train adapters for all eligible clusters |
| `/finetune train --cluster c-id` | Train one specific cluster |
| `/finetune eval --cluster c-id --version v2` | Lightweight perplexity check from training metrics |
| `/finetune promote --cluster c-id --version v2` | Mark an adapter active |
| `/finetune rollback --cluster c-id` | Revert to the previous version |
| `/finetune route "prompt text"` | Show which adapter would handle this prompt |
| `/finetune retro <subcommand>` | Retroactively label sessions and turns (see "Retroactive Labeling" above) |
| `/finetune status` | Display pipeline state, active adapters, cluster maturity |
| `/finetune cron weekly` | Schedule `/finetune run --with-bench` to run on a cron |
| `/finetune gc --keep 2` | Garbage collect old adapter versions |

## Dependencies

Install when activating this skill (not required by Hermes core):

```bash
pip install sentence-transformers>=2.2 hdbscan>=0.8.33 scikit-learn>=1.3
```

For training (GPU machine only):

```bash
pip install axolotl accelerate
```

## Pitfalls

- **Small datasets**: With fewer than ~100 good sessions, the `_general` adapter is all you get. Domain-specific adapters need density — don't force clusters.
- **Chat template mismatch**: Training data MUST use the same chat template as inference. Verify `chat_template` in config matches your model.
- **Quantization degradation**: If a merged GGUF performs worse than LoRA-on-base, try a higher quant level (Q6_K, Q8_0) or use unmerged LoRA loading.
- **Catastrophic forgetting**: The canary test set catches this. If canary scores drop, the adapter is blocked from promotion regardless of other metrics.

## Verification

After running the pipeline:

1. Check `/finetune status` — adapters should show as "active" with eval scores
2. Run `/finetune route "test prompt"` — should route to a cluster with confidence > 0.6
3. Start a conversation and verify the model uses the fine-tuned adapter (check logs)
