# Design: Positive Turn Detection Signals

**Status:** Proposal
**Parent:** `hermes-finetune` design spec (§2 — Automated Quality Scoring)
**Replaces:** Generic sentiment analysis as primary positive signal

---

## Principle

Positive signal detection should be based on **observable outcomes**, not interpreted intent. A tool that succeeds, an artifact that persists, a correction that produces a better response — these are concrete evidence that a turn was good. They don't require guessing what the user meant by their phrasing.

The scorer still uses negative signals (corrections, retries, contradictions) for exclusion, but positive signals are what drive inclusion in the training set. This document defines the positive signal catalog.

---

## Signal Catalog

### 1. Tool Success Chain

**What it detects:** The model called a tool, the tool succeeded, and the user moved forward.

**Evidence chain:**

```
Assistant calls tool
    → Tool returns success (exit_code 0, valid output, no error strings)
    → User's next turn is NOT a correction or retry
    → User's next turn advances to a new subtask or goes deeper
```

Each link in the chain increases confidence. A complete chain is strong positive evidence.

**Scoring:**

| Links completed | Score |
|---|---|
| Tool called + tool succeeded | 0.5 |
| + user did not correct/retry | 0.7 |
| + user advanced to new subtask | 0.9 |
| + user explicitly referenced tool output | 1.0 |

**Detection:**

```python
def tool_success_chain(assistant_turn, tool_results, next_user_turn) -> float:
    # Link 1: tool succeeded
    tool_ok = all(
        r.get("exit_code") == 0 or r.get("error") is None
        for r in tool_results
    )
    if not tool_ok:
        return 0.0

    score = 0.5

    # Link 2: user did not correct
    if next_user_turn and not is_correction(next_user_turn):
        score = 0.7

        # Link 3: user advanced topic
        if is_topic_advance(next_user_turn, assistant_turn):
            score = 0.9

        # Link 4: user referenced tool output
        if references_output(next_user_turn, tool_results):
            score = 1.0

    return score
```

**What "tool succeeded" means per tool type:**

| Tool | Success indicators |
|---|---|
| `terminal` | `exit_code == 0`, no stderr-only output, output is non-empty |
| `file` (read) | File content returned, no "not found" or permission errors |
| `file` (write) | Acknowledgment returned, subsequent read confirms content |
| `web_search` | Results returned, non-empty result set |
| `web_extract` | Content extracted, non-trivial length |
| `skill_manage` | Skill created/updated without error |
| `execute_code` | Code ran, returned output, no uncaught exceptions |

---

### 2. Artifact Longevity

**What it detects:** Content the model produced is still being used many turns later.

**Evidence:** The model generates code, a file, a plan, or structured output in turn N. In turns N+3, N+5, N+10, the user or model references, modifies, or extends that artifact. The longer the artifact survives in the conversation, the higher the quality signal.

**Scoring:**

| Artifact lifespan | Score |
|---|---|
| Referenced 1–2 turns later | 0.5 |
| Referenced 3–5 turns later | 0.7 |
| Referenced 6–10 turns later | 0.85 |
| Referenced 10+ turns later | 1.0 |
| Modified/extended (not just mentioned) | +0.1 bonus |

**Detection:**

Identify artifacts produced by assistant turns: file paths written, code blocks, named entities introduced, structured outputs. Then scan subsequent turns (both user and assistant) for references to those artifacts.

```python
def artifact_longevity(
    assistant_turn_index: int,
    all_turns: list[dict],
) -> float:
    artifacts = extract_artifacts(all_turns[assistant_turn_index])
    if not artifacts:
        return 0.0  # no artifacts produced, signal not applicable

    last_reference = assistant_turn_index
    was_modified = False

    for i in range(assistant_turn_index + 1, len(all_turns)):
        turn = all_turns[i]
        for artifact in artifacts:
            if artifact_referenced(turn, artifact):
                last_reference = i
            if artifact_modified(turn, artifact):
                was_modified = True

    span = last_reference - assistant_turn_index
    if span == 0:
        return 0.0

    if span <= 2:
        score = 0.5
    elif span <= 5:
        score = 0.7
    elif span <= 10:
        score = 0.85
    else:
        score = 1.0

    if was_modified:
        score = min(1.0, score + 0.1)

    return score
```

**Artifact types to track:**

| Type | Extraction method | Reference detection |
|---|---|---|
| File paths | Regex for paths in `write_file` / `terminal` output | Same path appears in later tool calls or user messages |
| Code blocks | Fenced code blocks in assistant content | Function/variable names reappear in later turns |
| Named plans/concepts | Capitalized terms or quoted names the model introduces | Same terms used by the user in later turns |
| Structured output | JSON, YAML, tables in assistant content | Keys/fields referenced in later discussion |

---

### 3. Productive Self-Correction

**What it detects:** The model made an error, received a correction, and produced a better response. The corrected response (not the original) is high-quality training data.

**Evidence chain:**

```
Assistant turn A (initial response)
    → User correction ("no, actually...", "that's wrong", explicit redirect)
    → Assistant turn B (corrected response)
    → User accepts turn B (moves forward, no further correction)
```

**Scoring:**

Turn A: score 0.0 (bad — it caused a correction)
Turn B: score 0.85–1.0 (good — it incorporated the correction successfully)

The corrected response is valuable *because* it demonstrates error recovery in context. The model had the failed attempt, the user's feedback, and then produced the right answer — that's exactly the behavior you want to reinforce.

**Detection:**

```python
def self_correction_signal(
    turns: list[dict],
    assistant_turn_index: int,
) -> float | None:
    """Returns score for this turn if it's a successful correction.
    Returns None if this turn is not part of a correction pattern.
    """
    # Check if the previous user turn was a correction
    prev_user = find_previous_user_turn(turns, assistant_turn_index)
    if prev_user is None or not is_correction(prev_user):
        return None  # not a correction context

    # Check if THIS turn is the corrected response
    # (i.e., the user turn before this one was the correction)
    next_user = find_next_user_turn(turns, assistant_turn_index)
    if next_user is None:
        return 0.6  # corrected, but we can't confirm acceptance

    if is_correction(next_user):
        return 0.0  # corrected response was ALSO wrong

    # User accepted the correction
    return 0.9
```

**Important:** The *original* failed turn must be scored as bad (0.0–0.2) by the negative signal detectors. The self-correction signal only applies to the *response after the correction*. This creates a natural DPO pair: the same prompt context produced a bad response and then a good one.

---

### 4. Resolution Velocity (Credit Assignment)

**What it detects:** Which turns in a completed task actually contributed to the successful outcome.

**Evidence:** The conversation reaches a productive conclusion — the user's goal is achieved, indicated by a terminal success marker (user says "done", "that works", conversation ends naturally after a successful tool chain). Working backward from the conclusion, turns that moved the conversation toward the goal score higher than turns that caused detours.

**Scoring:**

For each turn in a resolved session, compute the contribution:

```python
def resolution_velocity(
    turns: list[dict],
    resolution_index: int,  # turn where the task was completed
) -> dict[int, float]:
    """Returns {turn_index: velocity_score} for each assistant turn."""
    scores = {}
    
    # Walk backward from resolution
    remaining_distance = 0
    for i in range(resolution_index, -1, -1):
        turn = turns[i]
        if turn["role"] != "assistant":
            continue
        
        next_user = find_next_user_turn(turns, i)
        
        if next_user and is_correction(next_user):
            # This turn caused a detour — low velocity
            scores[i] = 0.2
            remaining_distance += 2  # penalty: detour costs extra
        elif next_user and is_retry(next_user):
            # Retry — the turn failed to advance
            scores[i] = 0.1
            remaining_distance += 1
        else:
            # Turn advanced the task
            # Score higher for turns closer to resolution
            proximity = 1.0 / (1.0 + remaining_distance * 0.2)
            scores[i] = max(0.5, proximity)
            remaining_distance = max(0, remaining_distance - 1)
    
    return scores
```

**Resolution detection:** A session is "resolved" if:
- The last user turn contains a completion marker ("thanks", "that works", "done", "perfect")
- The last tool call succeeded and the user didn't follow up (natural end)
- The user explicitly started a new topic (the previous topic was implicitly resolved)

Sessions without a clear resolution don't get velocity scoring — their turns keep their baseline scores from other signals.

---

### 5. Token Efficiency (Normalized by Outcome)

**What it detects:** Among turns that achieved the same outcome, which ones did it more concisely.

**Evidence:** Two turns both resulted in successful tool chains and user acceptance, but one is 200 tokens and the other is 800 tokens. The concise turn is better training data — it teaches the model to be direct.

**Scoring:**

This is a **modifier**, not a standalone signal. It adjusts the score of turns that already scored positively on other signals.

```python
def efficiency_modifier(
    assistant_turn: dict,
    baseline_score: float,
    category_median_tokens: float,
) -> float:
    """Adjust score based on token efficiency.
    Only applies to turns that already scored >= 0.5.
    """
    if baseline_score < 0.5:
        return 0.0  # don't reward concise bad responses
    
    turn_tokens = estimate_tokens(assistant_turn["content"])
    
    if category_median_tokens == 0:
        return 0.0
    
    ratio = turn_tokens / category_median_tokens
    
    if ratio <= 0.7:
        return 0.05   # notably concise — small bonus
    elif ratio >= 2.0:
        return -0.05  # notably verbose — small penalty
    else:
        return 0.0    # normal range — no adjustment
```

The modifier is capped at ±0.05 to prevent it from dominating. It's a tiebreaker, not a primary signal.

**Category median:** Computed per task type. Tool-calling turns are compared against other tool-calling turns. Explanatory turns against other explanatory turns. This prevents penalizing necessarily long responses (e.g., detailed code review) against short ones (e.g., "the file was created").

---

## Composite Integration

These signals plug into the existing composite scorer (design spec §2.5) as additional turn-level factors. The updated composite:

```
score = (
    w1 * conversation_signals +       # existing: abrupt termination, session length
    w2 * negative_turn_signals +       # existing: correction, contradiction, retry
    w3 * positive_turn_signals +       # NEW: this document
    w4 * sentiment_modifier +          # existing: lexicon-based
    w5 * manual_override               # existing: thumbs up/down, retro labels
)
```

Where `positive_turn_signals` is the max of all applicable positive signals for the turn:

```python
def positive_turn_score(turn_index, turns, tool_results) -> float:
    signals = []

    tsc = tool_success_chain(turns[turn_index], tool_results, next_user)
    if tsc > 0:
        signals.append(tsc)

    al = artifact_longevity(turn_index, turns)
    if al > 0:
        signals.append(al)

    sc = self_correction_signal(turns, turn_index)
    if sc is not None and sc > 0:
        signals.append(sc)

    rv = resolution_velocity_scores.get(turn_index)
    if rv is not None:
        signals.append(rv)

    if not signals:
        return 0.5  # no positive signal detected — neutral

    base = max(signals)  # take the strongest signal
    base += efficiency_modifier(turns[turn_index], base, median)
    return min(1.0, base)
```

Using `max` rather than `mean` prevents dilution when only one signal fires. A turn with a perfect tool success chain should score high even if artifact longevity doesn't apply.

### Updated Default Weights

```
w1 (conversation signals):    0.15   # was 0.3 — reduced, less informative per-turn
w2 (negative turn signals):   0.25   # was 0.4 — still important for exclusion
w3 (positive turn signals):   0.35   # NEW — primary inclusion signal
w4 (sentiment modifier):      0.05   # was 0.1 — demoted, least reliable
w5 (manual override):         0.20   # was 0.2 — unchanged, highest confidence when present
```

The shift: positive outcome signals are now the primary driver of inclusion, negative signals handle exclusion, conversation-level signals are context, and sentiment is a weak tiebreaker.

---

## Signal Applicability by Turn Type

Not every signal applies to every turn. A pure-text response with no tool calls won't have a tool success chain. A turn at the start of a conversation can't have artifact longevity yet.

| Signal | Applies when | Does not apply when |
|---|---|---|
| Tool success chain | Turn includes tool calls | Pure text response |
| Artifact longevity | Turn produces referenceable output (code, files, structured data) | Turn is conversational, advisory, or Q&A |
| Self-correction | Turn follows a user correction of a previous assistant error | Turn is the first response or follows a non-correction |
| Resolution velocity | Session reaches a clear resolution | Session was abandoned or is still active |
| Token efficiency | Turn already scored ≥ 0.5 on another signal | Turn scored below 0.5 (don't reward concise bad responses) |

When no positive signals apply (pure conversational turn, no tools, no artifacts), the positive signal score defaults to 0.5 (neutral). The turn's final score is then driven by the other factors — conversation signals, negative signals, sentiment.

---

## Example Scoring Walkthrough

A 6-turn session where the user asks the model to create a Python script:

```
Turn 1 (user):      "Create a Python CLI that counts words in a file"
Turn 2 (assistant):  Calls terminal: writes wordcount.py
                     Tool result: exit_code 0, file created
Turn 3 (user):       "That doesn't handle the case where the file doesn't exist"
Turn 4 (assistant):  Calls terminal: updates wordcount.py with error handling
                     Tool result: exit_code 0, file updated
Turn 5 (user):       "Perfect. Now add a --verbose flag"
Turn 6 (assistant):  Calls terminal: adds argparse --verbose
                     Tool result: exit_code 0
```

**Turn 2 scoring:**
- Tool success chain: tool succeeded (0.5), but user corrected → chain breaks at link 2 → 0.5
- Self-correction: N/A (not following a correction)
- Artifact longevity: wordcount.py referenced in turns 4 and 6 → span 4 → 0.7
- Resolution velocity: caused a detour (user corrected) → 0.2
- Negative signals: next user turn is a correction → penalty
- **Composite: low-to-mid** — tool worked but response was incomplete

**Turn 4 scoring:**
- Tool success chain: tool succeeded (0.5), user did not correct (0.7), user advanced to new subtask (0.9), user said "perfect" → 1.0
- Self-correction: follows correction of turn 2, user accepted → 0.9
- Artifact longevity: updated wordcount.py referenced in turn 6 → span 2 → 0.5
- Resolution velocity: advanced the task efficiently → high
- **Composite: high** — corrected the error, tool succeeded, user happy, artifact persisted

**Turn 6 scoring:**
- Tool success chain: tool succeeded (0.5), session ends naturally after success → 0.7
- Artifact longevity: last turn, can't measure → 0.0
- Resolution velocity: final turn in resolved session → high proximity → 0.9
- **Composite: high** — successful conclusion

The training set gets turns 4 and 6. Turn 2 is excluded or marked neutral. This is the right outcome — turn 4 is excellent training data (error recovery in context), and turn 6 is a clean completion.
