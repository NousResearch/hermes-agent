---
name: finetune
description: >
  Personal model fine-tuning pipeline. Extract sessions from the Hermes DB,
  score quality at the assistant-turn level, discover usage domains, train
  QLoRA adapters on individual high-quality turns, and manage versioned
  adapters with rollback.
version: 0.1.0
author: Ivy Darling
license: MIT
platforms: [linux]
required_environment_variables:
  - name: FINETUNE_BASE_MODEL
    prompt: "Path or HuggingFace ID of the base model for training"
    help: "Defaults to kai-os/Carnice-9b (HF repo ID). Must be HF safetensors, not GGUF. Takes precedence over finetune.training.base_model in config.yaml."
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

Train QLoRA adapters from your own Hermes session history. The pipeline extracts conversations from the Hermes session DB, scores each assistant turn for quality, discovers usage domains via clustering, trains per-domain adapters on individual high-quality turns (not whole sessions), and routes inference to the best adapter.

## When to Use

- User asks to fine-tune, train, or personalize their model
- User wants to improve model performance on their specific tasks
- User asks about adapter management, training status, or rollback
- User wants to score or review the quality of their conversation history
- User wants to label specific turns as good or bad for training
- Trigger phrases: "fine-tune", "train adapter", "finetune status", "model training", "label turn"

## Enabling the skill

`/finetune` is gated behind a master switch, off by default. Nothing runs — no extraction, no feedback capture — until you opt in:

```yaml
# ~/.hermes/config.yaml
finetune:
  enabled: true
```

Every `/finetune` subcommand refuses with a pointer to this setting while disabled. The `Ctrl+Y`/`Ctrl+N` feedback keybindings additionally require `finetune.feedback.cli_keybindings: true`.

## Quick Reference

| Command | Action |
|---|---|
| `/finetune status` | Show adapter registry, cluster state, data volume |
| `/finetune extract` | Extract new sessions from state.db |
| `/finetune score` | Run quality scoring on extracted sessions |
| `/finetune cluster` | Run/update domain discovery |
| `/finetune train [cluster]` | Train adapter for a cluster (or all eligible) |
| `/finetune eval --cluster c-id --version v2` | Lightweight perplexity gate from training metrics |
| `/finetune bench` | **Run the full 243-case benchmark against the active model** |
| `/finetune retro list` | **Show priority queue of unlabeled sessions** |
| `/finetune retro show <id>` | **Inspect a session's full conversation** |
| `/finetune retro good <id> [turns]` | **Label session or specific turns as good** |
| `/finetune retro bad <id> [turns]` | **Label session or specific turns as bad** |
| `/finetune retro skip <id>` | **Drop a session out of the queue** |
| `/finetune retro stats` | **Show labeling progress** |
| `/finetune promote [cluster] [version]` | Promote adapter to active |
| `/finetune rollback [cluster]` | Roll back to previous version |
| `/finetune redeploy` | **Convert active adapter to GGUF and restart llama-server with it loaded** |
| `/finetune route "prompt"` | Test which adapter would route for a prompt |
| `/finetune route enable` | **Install the inference-time adapter-routing plugin** |
| `/finetune route disable` | Remove the adapter-routing plugin |
| `/finetune run` | **Full pipeline, no bench gate** — fast, auto-promotes |
| `/finetune run --with-bench` | **Full pipeline + bench gate** — auto-rollback on regression |
| `/finetune gc --keep 2` | Delete old adapter versions (active + rollback target always kept) |
| `/finetune cron` | Schedule recurring retraining |

`/finetune` is CLI-only — it is not available from gateway platforms (Telegram/Discord/Slack).

Long-running subcommands (`train`, `bench`, `run`) can be cancelled with `Ctrl+C` — first press stops the training/bench subprocess cleanly, a quick second press force-kills it. Mutating operations (`run`, `promote`, `rollback`, `redeploy`, `gc`, training) are serialized behind a coarse lock (`<hermes-home>/finetune/finetune.lock`); a second concurrent invocation exits with "another finetune operation is running" instead of corrupting the registry.

## How training data is built

**The pipeline trains on individual assistant turns, not whole sessions.** This is the most important thing to understand about how the skill thinks about your data.

A Hermes session might be 50 turns long and contain a mix of great answers and mediocre ones. Treating it as a single training example would force one quality label across the whole thing and produce records that are far too long for typical training context windows. Instead, the pipeline walks through each session and emits **one training example per assistant turn that meets the quality bar**, with a sliding window of preceding context (default: 8 turns).

### What this means in practice

- A 12-turn conversation with 6 assistant responses produces **up to 6 training examples**, not one
- A 50-turn conversation with 25 assistant responses produces **up to 25 examples**
- Each example is bounded to its own context window (default ~8 turns), so it fits comfortably in any reasonable `sequence_len`
- Only assistant turns whose **per-turn score** meets `min_turn_score` (default 0.7) are emitted — bad and neutral turns are skipped, not down-weighted

### Where the per-turn scores come from

In priority order:

1. **Retro labels** you set with `/finetune retro good <id> 2,4` — explicit ground truth, override everything else
2. **Automated per-turn scores** from `/finetune score` — heuristic signals (affirmation, correction, follow-up depth, etc.)
3. **Session-level composite score** as a fallback when no per-turn data exists

### Tuning the granularity

Two knobs in `~/.hermes/config.yaml` control this:

```yaml
finetune:
  training:
    context_window_turns: 8      # max preceding turns per training example
    min_turn_score: 0.7          # only emit turns scoring at least this
```

Lowering `min_turn_score` to 0.45 produces dramatically more examples at lower average quality — useful for a first end-to-end run when you don't have many retro labels yet. Raising it produces fewer but stricter examples.

For 184 typical Hermes sessions, the default `min_turn_score: 0.7` typically produces ~250 training examples. Lowering to 0.45 produces ~750. The same dataset under the old session-based approach produced ~7.

---

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
- Lightweight perplexity check from training metrics is the only gate (when a training run recorded no eval metrics — e.g. datasets under 50 records train without an eval split — the gate loudly skips and passes)
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

**Requires `serving.auto_redeploy: true` and a configured `server_command`** (see Auto-Redeploy below). The gate refuses to start without them, because a bench that isn't measuring the newly deployed adapter would be measuring the *previous* model and calling it a verdict on the new one.

**What it does, in order:**
1. Steps 1–5 from Workflow A (extract, score, cluster, train, register)
2. **Promote** each new adapter to active
3. **Redeploy** llama-server with the new adapter loaded (aborts the gate if this fails — it never falls back to benching the old model)
4. **Run the benchmark** (243 prompt cases against the now-served adapter)
5. **Compare** against the accepted baseline (`bench/baseline.json`)
6. If regressed beyond the thresholds → **automatically rollback** (or, for a cluster's first-ever adapter, **deactivate**) the measured adapter and redeploy the previous adapter — or the bare base model if there is none. A regressed adapter is never left serving.
7. If passed → the new result becomes the accepted baseline

Note: with more than one cluster promoted in a run, only the **last-deployed** adapter is being served, so the bench measures only that one — the pipeline prints a warning and, on regression, rolls back only the served adapter.

**The baseline is a pointer, not "the most recent file."** Only runs that passed the gate (or the first-ever run) update `bench/baseline.json`. A regressed, rolled-back run can never become the next run's baseline — quality can't silently ratchet downward.

**Trade-offs:**
- Safe — a regressing adapter never stays active for long
- Slow — the bench takes 20–35 min on top of the training time
- Requires an accepted baseline to compare against. If none exists, the run passes by definition and is recorded as the baseline
- Best for scheduled retraining (cron) and any retrain you don't intend to babysit

**Regression thresholds** (hardcoded in `scripts/eval.py` and the bench env):
- `tool_selection_accuracy`: must not drop more than 3%
- `tool_execution_success`: must not drop more than 5%
- `task_completion_rate`: must not drop more than 5%
- `format_compliance`: must remain ≥ 95%
- `hallucination_rate`: must stay ≤ 1% (small tolerance so a single flaky case can't auto-reject an adapter)
- `canary_pass_rate`: must not drop more than 5%

Gates apply only to metrics present in both the candidate and the baseline; a missing gate metric is loudly skipped, never silently passed or failed. `hallucination_rate` counts calls to tool names that don't exist; unparseable tool-call JSON is tracked separately as `tool_call_parse_failure_rate`. Cases that fail on infrastructure errors (endpoint timeouts, connection resets) are excluded from all quality denominators and reported as `infra_errors`; a run with more than 5% infra errors is invalidated (exit code 3) instead of producing a verdict. Bench requests are seeded (`seed: 1234` in `bench/default.yaml`) for run-to-run determinism.

If **any** gate fails, the measured adapter is rolled back automatically (see the note above about multi-cluster runs).

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

Results land in `~/.hermes/finetune/bench/results/bench_<timestamp>.json` and are compared against the accepted baseline (`bench/baseline.json`); a passing standalone bench updates that baseline. The bench requires a running Docker daemon — it sandboxes every case and refuses to run agent-generated commands on the host unless you explicitly override (`FINETUNE_BENCH_ALLOW_UNSANDBOXED=1`).

## Auto-Redeploy (llama.cpp integration)

By default, training a new adapter doesn't change what your llama-server is actually serving. The adapter lands in the registry and gets marked active, but llama.cpp keeps serving whatever GGUF it was started with. To actually use the trained adapter, you'd normally have to convert it to GGUF, stop llama-server, restart with `--lora`, and verify it came up. The pipeline can do all of that for you.

### What it does

When `auto_redeploy: true` is set, after `/finetune run` finishes promoting a new adapter, the pipeline runs an extra step:

1. Looks up the active adapter from the registry
2. Auto-detects the HuggingFace snapshot directory for the base model (from `~/.cache/huggingface/hub/`)
3. Calls llama.cpp's `convert_lora_to_gguf.py` to produce a GGUF copy of the adapter
4. Stops the managed llama-server (strictly via its PID file — servers started outside the skill are never touched)
5. Restarts llama-server with `--lora <path-to-gguf>` appended to the configured launch command
6. Polls `/v1/models` until the new server is responsive (default 30s timeout) — and verifies the process it launched is the one answering, so a pre-existing server squatting on the port can't fake a healthy deploy
7. Writes a serving manifest (`<hermes-home>/finetune/serving.json`) recording the server PID and which adapters it preloaded — this is what inference-time routing reads
8. If the rest of the pipeline runs `/finetune run --with-bench`, the bench then measures the adapter that's *actually being served*

GGUF conversion is atomic: a killed or timed-out conversion can never leave a truncated `adapter.gguf` behind for a later deploy to trust. The `%LORA%` path is shell-quoted at substitution time, so adapter paths containing spaces are safe.

### Configuration

Add the following section to `~/.hermes/config.yaml`. **All of it is required if you want auto-redeploy on** — the defaults assume nothing about your specific setup:

```yaml
finetune:
  serving:
    auto_redeploy: true

    # Path to llama.cpp's convert_lora_to_gguf.py script.
    # No default — redeploy aborts with an error until you set this.
    converter: ~/programs/llama.cpp/convert_lora_to_gguf.py

    # Where to find the HF safetensors snapshot of the base model.
    # "auto" detects from ~/.cache/huggingface/hub/ based on the
    # finetune.training.base_model HF repo ID. You can also pass an
    # explicit path here.
    base_model_snapshot: auto

    # The exact command used to launch llama-server. Multi-line is fine.
    # Use %LORA% as a placeholder for the LoRA path — it will be replaced
    # at deploy time. If you don't include %LORA%, the pipeline appends
    # --lora <path> automatically.
    server_command: |
      ~/programs/llama.cpp/build/bin/llama-server
      -m ~/programs/carnice/Carnice-9b-Q8_0.gguf
      -ngl 999 -c 32768 --host 0.0.0.0 --port 8008
      --lora %LORA%

    # Where the pipeline writes the launched server's PID and log. Used to
    # stop the previous server cleanly between deploys. Leave empty to use
    # the defaults under <hermes-home>/finetune/llama-server.{pid,log}.
    server_pid_file: ""
    server_log_path: ""

    # Health check after restart
    health_check_url: http://localhost:8008/v1/models
    health_check_timeout: 30
```

### Manual redeploy

Even with `auto_redeploy: false`, you can run the redeploy step on demand:

```
/finetune redeploy
```

This deploys whatever's currently active in the registry. Useful when you want to push a manually-promoted adapter live, or when you've edited the LoRA outside the pipeline — manual `redeploy` always reconverts the adapter to GGUF (pipeline auto-redeploys reuse the cached conversion).

You can also redeploy a specific cluster/version:

```
/finetune redeploy --cluster _general --version v3
```

### Prerequisites

- **llama.cpp checkout with `convert_lora_to_gguf.py`** at the configured path. This script is in recent versions of llama.cpp.
- **The HF safetensors must be in the local cache.** Axolotl downloads them automatically the first time you run `/finetune run`, so this is usually a no-op. If you want to pre-warm the cache, use `huggingface-cli download <repo>`.
- **No other process holding the GPU** when llama-server restarts. The pipeline doesn't manage other GPU consumers.

### What can go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| `redeploy: could not auto-detect HF snapshot` | The base model isn't in the HF cache, or `base_model` is a local path instead of an HF repo ID | Run `huggingface-cli download <repo>` to populate the cache, or set `base_model_snapshot` explicitly |
| `redeploy: GGUF conversion failed` | llama.cpp's converter doesn't support this base architecture, or the adapter is malformed | Check the converter's stderr in the error message; fall back to manual conversion to debug |
| `redeploy: server did not respond within 30s` | The new llama-server failed to start (bad command, missing model, GPU OOM) | Check `<hermes-home>/finetune/llama-server.log` for the actual failure |
| llama-server starts but the new adapter isn't applied | The `server_command` template doesn't include `--lora %LORA%`, OR llama.cpp version doesn't support runtime LoRA loading | Add `--lora %LORA%` to the template and verify your llama.cpp build supports `--lora` |

### Interaction with the bench gate

`/finetune run --with-bench` **requires** `auto_redeploy: true` (plus a `server_command`) and refuses to start without it. The flow:

1. Train the adapter
2. Promote it
3. **Redeploy llama-server with the new adapter loaded** (gate aborts if this fails)
4. Run the bench (now actually measuring the new adapter)
5. Compare to the accepted baseline; if regressed → rollback (or deactivate a first-ever adapter) AND redeploy the previous adapter — or the bare base model — so the served model always matches the registry; if passed → the result becomes the new accepted baseline

This is the closed-loop "retrain and validate without human intervention" workflow. The cron-scheduled retraining mode (`/finetune cron weekly`) becomes truly hands-off when combined with `auto_redeploy: true` — `/finetune cron` schedules the gated pipeline automatically when serving is configured, and the ungated fast pipeline (with honest wording about the missing gate) when it isn't.

---

## Inference-Time Adapter Routing

Once adapters are trained, promoted, **and redeployed**, hermes can decide
per prompt whether the served adapter should apply. Routing runs as a
standard hermes plugin (`finetune-routing`, shipped in this skill's
`plugin/` directory) that registers `llm_request` middleware.

How it works — and its honest limits: llama.cpp cannot load an arbitrary
adapter file per request. Its per-request API is scale control over
adapters **preloaded** at server startup (`"lora": [{"id": N, "scale": s}]`,
a llama.cpp extension field accepted through the OpenAI-compatible
endpoint). So the middleware only activates when `/finetune redeploy` has
written a serving manifest (`<hermes-home>/finetune/serving.json`)
describing the server it launched and the adapters that server preloaded.
It then embeds the incoming prompt, matches it against cluster centroids,
and:

- prompt routes to a **served** cluster → that adapter gets scale 1.0
- prompt routes anywhere else → every served adapter gets scale 0.0, so
  **off-domain prompts deliberately fall back to the base model**

The decision is carried on the request payload — no process-global state —
so concurrent sessions can't observe each other's adapter selection. The
manifest is re-read when it changes on disk, so a redeploy or rollback in
another process takes effect without restarting your session.

```bash
/finetune route enable      # copy the plugin into <hermes-home>/plugins/
# then set in ~/.hermes/config.yaml:
#   finetune:
#     routing:
#       enabled: true
/finetune route disable     # remove the plugin
```

The plugin only activates when the request targets the host the manifest's
server is on (loopback aliases — localhost/127.0.0.1/[::1] — are treated
as equivalent); requests to remote providers are never touched. Restart
hermes after enabling; plugins are discovered at session start.

An adapter that has been trained and promoted but never redeployed is not
in the manifest and cannot be routed to — run `/finetune redeploy` first.
With a single served adapter (the current serving model), routing's value
is the off-domain fallback; the manifest's adapter list is the extension
point for multi-adapter serving.

## Retroactive Labeling

The automated quality scorer is conservative — early in the pipeline you'll likely see most assistant turns score around 0.5, which means they fall below the default `min_turn_score: 0.7` threshold and don't contribute to training. The retro flow lets you go back and label specific turns by hand, seeding real ground-truth signal that the trainer will then prefer.

Because the trainer operates on individual turns (see "How training data is built" above), labeling **just the good turns** of a session is exactly the right granularity — there's no need to commit to a verdict on the whole conversation.

### When to use retro

- After your first `/finetune extract` and `/finetune score` run, when not enough turns clear the `min_turn_score` threshold to leave the embryonic stage
- Periodically as new sessions accumulate, to maintain fresh ground-truth signal
- After a regression, to flag the bad turns that the scorer missed
- When you remember a specific session that produced a great answer worth training on

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

Most sessions contain a mix of quality. A 12-turn session might have 8 great turns and 4 mediocre ones. Because each assistant turn becomes its own training example, you should label only the turns worth learning from — not the whole session. Use the turn syntax to target specific turns:

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

After labeling, the next `/finetune score` run honors your labels. The scorer applies them in priority order to compute each turn's effective score:

1. **Per-turn retro labels** — highest priority. Override the automated turn score directly.
2. **Session-level retro labels** — apply to every assistant turn in the session that doesn't have a turn-level label.
3. **In-the-moment feedback** — `Ctrl+Y`/`Ctrl+N` from the CLI (these only fire when the input line is empty and no modal prompt is open, so they never shadow emacs yank/next-line while typing; they require `finetune.enabled` plus `finetune.feedback.cli_keybindings`). Gateway emoji reactions are **not** a feedback source: no platform adapter emits reaction events, and session-level labels from arbitrary chat members would be unsafe training signal. If platform reactions return, it will be as a proper design with per-turn attribution and authorship checks.
4. **Automated heuristic** — falls through if nothing above applies.

The trainer then includes a turn in the training set if and only if its effective score meets `min_turn_score`. A `good` retro label maps to 1.0; a `bad` label maps to 0.0; anything labeled `bad` is therefore guaranteed to be excluded.

You can use `/finetune retro` independently of the rest of the pipeline. The labels just sit in `feedback.jsonl` until the next `/finetune score` (or `/finetune run`) consumes them.

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
| `/finetune route enable` | Install the routing plugin into `<hermes-home>/plugins/` |
| `/finetune route disable` | Remove the routing plugin |
| `/finetune retro <subcommand>` | Retroactively label sessions and turns (see "Retroactive Labeling" above) |
| `/finetune status` | Display pipeline state, active adapters, cluster maturity |
| `/finetune cron weekly` | Create/update a real Hermes cron job (`finetune-retrain`) running the pipeline — gated (`--with-bench`) when serving is configured, ungated otherwise |
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

- **Small datasets**: With fewer than ~50 trainable assistant turns (after `min_turn_score` filtering), the `_general` cluster stays embryonic and training is skipped. Run `/finetune retro` to label more turns explicitly, or temporarily lower `finetune.training.min_turn_score` to 0.45 to admit more automated-score turns.
- **Sparse clusters**: With fewer than ~30 sessions in a domain, HDBSCAN won't form a dedicated cluster — everything goes to `_general`. This is correct for most personal use cases. Domain-specific adapters become useful at 500+ sessions.
- **Chat template mismatch**: Training data MUST use the same chat template as inference. Verify `chat_template` in config matches your model.
- **GGUF base model**: Axolotl trains against HuggingFace safetensors, NOT GGUF. Set `finetune.training.base_model` to the HF repo ID (e.g. `kai-os/Carnice-9b`), not the GGUF path. The GGUF path is only for inference-time serving via llama.cpp.
- **Quantization degradation**: Serving applies the LoRA unmerged on top of the base GGUF. If quality looks degraded, try a higher quant level of the base model (Q6_K, Q8_0).
- **Catastrophic forgetting**: The canary test set catches this. If canary scores drop, the adapter is blocked from promotion regardless of other metrics.
- **VRAM tightness on 12GB cards**: Training a 9B model with QLoRA needs ~10-11GB peak VRAM at `sequence_len: 1024`, `micro_batch_size: 1`. Stop any other CUDA processes (including any local llama.cpp serving the same GPU) before launching training. The default template settings target 12GB cards.
- **Sessions still in progress**: extraction tracks per-session message counts, so a conversation that grows after being extracted is re-extracted in full on the next run. The first extract after upgrading re-extracts everything once (the old watermark format is ignored).
- **What extraction includes**: each session is extracted standalone (compression continuations and `/branch` children are their own conversations, keyed to their root session so a lineage never straddles the train/eval split). Rewound turns, archived sessions, and delegate-subagent sessions are excluded — set `finetune.extract.include_delegates: true` if you want agent-to-agent traffic in your training data.
- **Secrets in training data**: message and tool-output content passes a conservative redaction pass (API keys, bearer tokens, PEM blocks, password assignments → `[REDACTED]`) before being written to training records. It is pattern-based, not exhaustive — review `train.jsonl` before sharing adapters trained on sensitive sessions.

## Verification

After running the pipeline:

1. Check `/finetune status` — adapters should show as "active" with eval scores
2. Run `/finetune route "test prompt"` — should route to a cluster with confidence > 0.6
3. Start a conversation and verify the model uses the fine-tuned adapter (check logs)
