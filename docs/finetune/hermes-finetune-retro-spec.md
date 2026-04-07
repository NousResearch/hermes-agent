# Design: Retrospective Session Labeling (`/finetune retro`)

**Status:** Proposal
**Parent:** `hermes-finetune` design spec
**Component:** `optional-skills/mlops/finetune/` — retro subcommand

---

## Problem

The finetune pipeline has two feedback paths: automated quality scoring (heuristic, always-on) and in-the-moment reactions (thumbs up/down keybindings and gateway emoji). Both miss a critical signal: **delayed judgment**.

You realize a session was good when the code it produced is still running a week later. You realize it was bad when you hit a bug the model introduced three days ago. By then the moment has passed — there's no mechanism to go back and label it.

The deeper problem is volume. Even if users want to retroactively label sessions, presenting 500 unlabeled sessions in chronological order is a dead interface. Nobody will use it.

`/finetune retro` solves both problems: it lets users label historical sessions and turns, and it presents them in an order that maximizes the value of each label.

---

## Priority-Ranked Session Surfacing

The retro flow does not present sessions chronologically. It ranks them by **label value** — how much a human label on this session would improve training data quality.

### Ranking Factors

| Factor | Weight | Rationale |
|---|---|---|
| **Scorer uncertainty** | 0.35 | Sessions in the neutral zone (composite score 0.4–0.7) or with contradictory signals are where a human label resolves the most ambiguity. Labeling a session the scorer already classified as 0.95 or 0.05 adds almost nothing. |
| **Tool density** | 0.25 | Sessions with higher tool call counts are more valuable because tool-calling behavior is what the fine-tune most directly affects. Pure-text conversations contribute less to adapter quality. |
| **Labeling surface area** | 0.15 | Longer sessions with more assistant turns yield more labeled data per review. Efficiency per human minute. |
| **Recency** | 0.15 | Recent sessions rank higher because the user is more likely to remember context and make accurate judgments. Exponential decay, half-life of 14 days. |
| **Cluster boundary proximity** | 0.10 | Sessions whose embeddings are nearly equidistant from two cluster centroids benefit from labeling because the label improves both scorer calibration and cluster assignment. |

### Priority Score

```
priority = (
    0.35 * uncertainty_score +
    0.25 * normalized_tool_density +
    0.15 * normalized_turn_count +
    0.15 * recency_decay +
    0.10 * boundary_proximity
)
```

**Uncertainty score:** Peaks at composite score 0.5, drops toward 0 at the extremes. Contradictory signals (e.g., positive sentiment + explicit correction detected) add a bonus.

```python
def uncertainty_score(composite: float, contradictory: bool) -> float:
    # Highest uncertainty at 0.5, lowest at 0.0 and 1.0
    base = 1.0 - abs(2.0 * composite - 1.0)
    bonus = 0.2 if contradictory else 0.0
    return min(1.0, base + bonus)
```

**Tool density:** `min(1.0, tool_call_count / 10)`. Sessions with 10+ tool calls get maximum weight.

**Normalized turn count:** `min(1.0, assistant_turn_count / 15)`. Caps at 15 to avoid over-weighting very long sessions.

**Recency decay:** `exp(-0.693 * days_old / 14)`. Half-life of 14 days. A 2-week-old session scores 0.5, a month-old session scores 0.25.

**Boundary proximity:** `1.0 - abs(sim_to_nearest - sim_to_second_nearest)`. Highest when the session is equidistant from two clusters. Zero if it's firmly inside one cluster or if clustering hasn't been run yet.

Priority is computed at invocation time over all unlabeled sessions. Sessions with existing labels (from in-the-moment feedback, previous retro passes, or propagation) are excluded from the queue.

---

## Similarity Batching

Reviewing sessions one at a time is inefficient when many sessions cover similar ground. The retro flow groups semantically similar sessions into batches, letting the user label a representative and propagate to neighbors.

### Batching Mechanism

After priority ranking, the top N candidate sessions (default: 100) are grouped using the same embedding space as domain discovery:

1. Compute pairwise cosine similarity across the candidate set.
2. Greedily form batches: take the highest-priority unlabeled session as the batch leader, add all sessions with cosine similarity > 0.8 to the leader, cap batch size at 8.
3. Present batch leaders in priority order.

### Propagation

When the user labels a batch leader, the system offers to propagate that label to the other sessions in the batch:

```
Session: "Rust trait design for P2P-CD node" (3 days ago, 12 turns)
Scorer: 0.52 (neutral — contradictory signals)
Similar unlabeled sessions: 4

[review conversation...]

Label: /finetune retro good 3,5,7

Propagate to 4 similar sessions? [Y/n/review]
```

Propagated labels are stored with metadata:

```json
{
  "session_id": "propagated-session-uuid",
  "label": "good",
  "source": "propagated",
  "propagated_from": "leader-session-uuid",
  "confidence": 0.7,
  "timestamp": "2026-04-06T15:30:00Z"
}
```

Propagated labels receive a confidence weight of 0.7× the direct label. In training data bucketing, this means propagated "good" labels produce a composite override of 0.7 (still above the good threshold) rather than 1.0. The user can review and override any propagated label later.

### Propagation Scope

Propagation applies the session-level label only, not turn-level labels. Turn-level specificity requires reviewing the actual conversation, so it doesn't propagate. If the user labeled specific turns on the leader, the propagated sessions get a session-level label at reduced confidence.

---

## Turn-Level Labeling

Sessions often contain a mix of quality. A 20-turn session might have 15 mediocre turns and 5 excellent ones. Training on all 20 dilutes the signal. Turn-level labeling lets users target the specific turns worth training on.

### Turn Identification

When reviewing a session, assistant turns are numbered sequentially (1-indexed, only counting assistant turns, not user/system/tool turns). The retro preview shows each assistant turn with its automated score:

```
Turn 1 [0.45 neutral]: "Here's how I'd approach the trait boundary..."
Turn 2 [0.82 good]:    "The Node struct should own the PeerHandle..."
Turn 3 [0.38 bad]:     "You could use Arc<Mutex<...>> for the..."
Turn 4 [0.71 good]:    "Actually, a better pattern is..."
Turn 5 [0.55 neutral]: "For the test, you'd want..."
```

### Labeling Syntax

| Command | Effect |
|---|---|
| `/finetune retro good` | Label all assistant turns in the current session as good (override to 1.0) |
| `/finetune retro bad` | Label all assistant turns as bad (override to 0.0) |
| `/finetune retro good 2,4` | Label turns 2 and 4 as good. Other turns keep automated scores. |
| `/finetune retro bad 3` | Label turn 3 as bad. Other turns keep automated scores. |
| `/finetune retro good 2,4 bad 3` | Mixed: turns 2 and 4 good, turn 3 bad. Turns 1 and 5 keep automated scores. |
| `/finetune retro skip` | Skip this session without labeling. It stays in the queue but drops in priority. |
| `/finetune retro note "great tool chain"` | Attach a text note to the session (not a label, just metadata for later reference). |

### Storage

Turn-level labels are stored in the same `feedback.jsonl` as in-the-moment feedback:

```json
{
  "session_id": "uuid",
  "turn_index": 2,
  "signal": "good",
  "override_score": 1.0,
  "source": "retro",
  "timestamp": "2026-04-06T15:30:00Z",
  "note": null
}
```

Session-level labels expand to one entry per assistant turn.

### Interaction with Automated Scoring

Turn-level retro labels are the highest-priority signal in the composite scorer:

```
effective_score = retro_label ?? in_moment_label ?? automated_composite
```

If a turn has both a retro label and an in-the-moment label, the retro label wins (it's more considered). If neither exists, the automated composite score applies.

---

## Retro Modes

### Interactive Triage (default)

```
/finetune retro
```

Presents sessions one at a time in priority order. The user reviews the conversation, labels turns, and advances. A running status line shows progress:

```
[Retro] 12 reviewed | 8 labeled | 3 skipped | 47 remaining (priority queue)
```

The session exits when the user types `/done`, runs out of sessions, or has reviewed 20 sessions (configurable via `finetune.retro.batch_limit`).

### Filtered Retro

```
/finetune retro --cluster c-a7f3e2       # Only sessions in this cluster
/finetune retro --since 2026-03-01       # Only sessions after this date
/finetune retro --tools-only             # Only sessions with tool calls
/finetune retro --neutral-only           # Only sessions scored 0.4–0.7
/finetune retro --search "trait design"  # FTS5 search within session content
```

Filters narrow the candidate pool before priority ranking. They combine: `--cluster c-a7f3e2 --since 2026-03-01` shows only recent sessions in that cluster.

### Quick Retro

```
/finetune retro quick
```

A faster mode that shows only the session title, turn count, automated score, and tool summary — no full conversation preview. The user labels based on the title alone (which they can expand if needed). Useful for power users who remember their sessions by title and want to label 50 sessions in 5 minutes.

### Audit Mode

```
/finetune retro audit
```

Reviews sessions that were already labeled (including propagated labels) rather than unlabeled ones. Useful for auditing propagated labels or reconsidering previous judgments. Sorted by lowest confidence first.

---

## Label Statistics & Impact

After a retro session, the system reports the impact of the new labels:

```
Retro session complete.
  Direct labels:      8 sessions, 23 turns
  Propagated labels:  14 sessions
  Training data impact:
    Good bucket:      +19 turns (was 312, now 331)
    Bad bucket:       +4 turns (was 45, now 49)
    Moved from neutral to good: 15 turns
    Moved from neutral to bad:  4 turns
  Clusters affected:  c-a7f3e2 (protocol-spec), _general
  Retraining recommended: c-a7f3e2 (data grew 6.1%)
```

If the label volume triggers the retraining threshold (default: 20% data growth), the system recommends retraining and offers to kick it off.

---

## Configuration

Added to the `finetune` section in `~/.hermes/config.yaml`:

```yaml
finetune:
  retro:
    batch_limit: 20                  # max sessions per retro invocation
    propagation_threshold: 0.8       # cosine similarity for batch grouping
    propagation_confidence: 0.7      # confidence weight for propagated labels
    recency_halflife_days: 14        # exponential decay half-life
    max_batch_size: 8                # max sessions per similarity batch
    candidate_pool_size: 100         # top-N sessions to consider for batching
```

---

## Skill Commands (additions to §8.1)

| Command | Action |
|---|---|
| `/finetune retro` | Start interactive retro labeling in priority order |
| `/finetune retro quick` | Fast mode — title-only review |
| `/finetune retro audit` | Review existing labels (lowest confidence first) |
| `/finetune retro good [turns]` | Label current session/turns as good |
| `/finetune retro bad [turns]` | Label current session/turns as bad |
| `/finetune retro good [turns] bad [turns]` | Mixed labeling |
| `/finetune retro skip` | Skip current session |
| `/finetune retro note "text"` | Attach a note to current session |
| `/finetune retro stats` | Show label statistics and training data impact |
| `/finetune retro --search "query"` | Search-filtered retro |
| `/finetune retro --cluster {id}` | Cluster-filtered retro |
| `/finetune retro --neutral-only` | Only neutral-scored sessions |

---

## Gateway Considerations

The retro flow is primarily a CLI interaction — it involves reviewing full conversations and making nuanced labeling decisions. On messaging platforms (Telegram, Discord), the experience is degraded because:

- Full conversation previews are long for chat bubbles.
- Turn-level labeling syntax is awkward on mobile keyboards.
- The interactive triage loop doesn't map cleanly to async messaging.

For gateway platforms, `/finetune retro` operates in **quick mode by default** (title + summary only) and supports only session-level labels. Turn-level labeling is available but requires the user to first request the full session via `/finetune retro expand`. The full interactive triage flow is CLI-only.

---

## Implementation Status (Phase 1 MVP)

This spec is shipped in two phases. **Phase 1** is included in the initial finetune skill PR; **Phase 2 and 3** land as follow-ups once the MVP has real usage feedback.

### What ships in Phase 1

**Architectural deviation from the spec, by design:** the spec describes an "interactive triage" loop where the user reviews sessions in order. Hermes runs `/finetune` subcommands as captured subprocesses (`subprocess.run(capture_output=True)`), so a true interactive REPL isn't possible without bypassing prompt_toolkit's terminal ownership. Phase 1 therefore implements retro as a **stateless command-driven** workflow: each `/finetune retro <subcommand>` invocation does one thing and returns. State persists in `feedback.jsonl` between calls. The UX is identical from the user's perspective — a queue they walk through — but each step is a separate command instead of one long-running process.

**Implementation lives at** `optional-skills/mlops/finetune/scripts/retro.py` (~370 LOC).

#### Commands

| Command | Implemented | Notes |
|---|---|---|
| `/finetune retro list [--limit N]` | ✅ | Default limit: 10 |
| `/finetune retro show <session_id>` | ✅ | Prefix matching on session IDs |
| `/finetune retro good <session_id> [turns]` | ✅ | Session-level or turn-spec |
| `/finetune retro bad <session_id> [turns]` | ✅ | Session-level or turn-spec |
| `/finetune retro skip <session_id>` | ✅ | Drops from queue, doesn't pollute scores |
| `/finetune retro stats` | ✅ | Total / queued / labeled breakdown |

#### Priority ranking

Implemented with the spec's formula minus `boundary_proximity` (which requires real cluster centroids). The 0.10 boundary weight is redistributed:

| Factor | Spec | Phase 1 |
|---|---|---|
| Uncertainty | 0.35 | **0.50** |
| Tool density | 0.25 | **0.25** |
| Turn count | 0.15 | **0.10** |
| Recency | 0.15 | **0.15** |
| Boundary proximity | 0.10 | *deferred* |

The uncertainty function (`1 - |2*composite - 1|`), tool density saturation at 10 calls, turn count saturation at 15, and recency exponential decay (14-day half-life) all match the spec exactly.

#### Turn-level labeling

Implemented per spec:
- Turn numbers are 1-based and count only assistant turns.
- Turn spec syntax supports individual turns (`2`), ranges (`2-5`), and mixed (`1,3-5,8`).
- Out-of-range turns are dropped with a warning instead of erroring.
- Labels are stored in `feedback.jsonl` with the spec's record format (adds `turn_index` field).

#### Storage and scorer integration

Per spec:
- All labels write to `~/.hermes/finetune/feedback.jsonl`.
- `source: "retro"` distinguishes retro labels from in-the-moment thumbs up/down.
- Session-level labels expand to one per-turn record plus a session-level marker (so the queue knows the session is fully labeled).
- `score.py::QualityScorer` reads turn-level overrides via `_load_turn_feedback()` and applies them in `score_session()` before computing the composite score.

The override priority is exactly what the spec specifies:
```
effective_score = retro_turn_label
              ?? retro_session_label
              ?? in_moment_label
              ?? automated_composite
```

#### Session deduplication

`load_all_scored()` reads every `scored_*.jsonl` file in `~/.hermes/finetune/data/scored/` and deduplicates by `session_id`, keeping the most recently scored copy. This means re-scoring after labeling doesn't pollute the queue with stale entries.

### Tests

10 unit tests in `tests/test_finetune.py::TestRetro`:

- `test_priority_uncertainty_peaks_at_neutral` — neutral sessions outrank confident ones
- `test_priority_recency_decay` — recent sessions outrank old ones at the same uncertainty
- `test_priority_tool_density_boost` — high tool count increases priority
- `test_parse_turn_spec_simple` — `"1,3,5"` and `""` (= all turns)
- `test_parse_turn_spec_range` — `"2-5"` and `"1,3-5,8"`
- `test_parse_turn_spec_clamps_to_max` — out-of-range turns dropped with warning
- `test_label_writes_session_marker_and_per_turn` — session-level expansion
- `test_labeled_session_ids_excludes_skip` — skip records don't count as labels
- `test_score_session_honors_per_turn_overrides` — scorer integration
- `test_load_all_scored_dedupes_by_id` — re-scored sessions don't duplicate

Total test count after Phase 1: **33 passing** (was 23 before retro).

### Documentation

A `Retroactive Labeling` section was added to `optional-skills/mlops/finetune/SKILL.md` covering: when to use retro, the queue-based command flow, priority ranking factors, turn-level label syntax, scorer override priority, and prefix matching. The Quick Reference table includes all six retro commands.

---

## Deferred to Phase 2

| Feature | Why deferred |
|---|---|
| **Similarity batching** of the candidate pool | Requires `sentence-transformers` already loaded for embeddings — adds startup latency the MVP doesn't need |
| **Label propagation** to similar sessions | Depends on batching; the propagation confidence weighting (0.7×) and the leader-vs-follower distinction add complexity that warrants a focused review |
| **`audit` mode** for reviewing existing labels | Niche — only useful after Phase 2 propagation, since direct labels rarely need auditing |
| **Boundary proximity** priority factor | Requires real cluster centroids — depends on the user having run `/finetune cluster` first |

## Deferred to Phase 3

| Feature | Why deferred |
|---|---|
| **Filtered modes** (`--cluster`, `--since`, `--tools-only`, `--neutral-only`, `--search`) | Each adds ~10-30 LOC; they compose; ship as a batch once we know which filters users actually want |
| **`quick` mode** (title-only review) | Useful for power users but the MVP's `list` already shows enough context to skip-label without `show` |
| **Stats with training-data delta** ("moved 15 turns from neutral to good") | Requires re-running the scorer mid-retro to compute deltas; adds a non-trivial cross-script dependency |
| **`note "text"` attaching** | Trivial to add but the storage schema is the only piece — no consumer reads notes yet |
| **Configuration section** under `finetune.retro` in `config.yaml` | All Phase 1 weights and limits are constants in `retro.py`. Once Phase 2 lands, expose them in config so users can tune. |
| **Gateway-specific behavior** (quick mode default, `expand` command) | Phase 1 retro is CLI-first; gateway support is a separate UX concern |

### Migration path

Phase 1's `feedback.jsonl` schema is the **same schema** Phase 2 and Phase 3 will use. No migration needed when Phase 2 ships — existing labels are forward-compatible:

- Direct labels written by Phase 1 will simply have `source: "retro"` with no `propagated_from` field, which Phase 2 treats as a confidence-1.0 ground-truth label.
- Skip records use `signal: "skip"` which is preserved across all phases.
- Per-turn vs session-level distinction via `turn_index` is the same across phases.

The Phase 1 priority weights will be re-tuned in Phase 2 to reintroduce `boundary_proximity: 0.10` (cutting `uncertainty` back from 0.50 to 0.40). Users won't notice — the queue order is heuristic, not load-bearing.
