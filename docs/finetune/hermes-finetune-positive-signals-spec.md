# Design: Positive Turn Detection Signals

**Status:** Proposal
**Parent:** `hermes-finetune` design spec (§2 — Automated Quality Scoring)
**Replaces:** Generic sentiment analysis as primary positive signal

---

## Principle

Positive signal detection should be based on **observable outcomes**, not interpreted intent. A tool that succeeds, an artifact that persists, a correction that produces a better response — these are concrete evidence that a turn was good. They don't require guessing what the user meant by their phrasing.

The scorer still uses negative signals (corrections, retries, contradictions) for exclusion, but positive signals are what drive inclusion in the training set. This document defines the positive signal catalog.

---

## Calibration on Real Data

This spec was empirically validated against a real Hermes session DB containing 231 sessions and 14,337 turns from a heavy power-user workload (51% tool-call density, dominated by `read_file`/`patch`/`search_files`/`terminal` operations on a Rust codebase).

Signal coverage on a sample of 30 representative sessions:

| Signal | Coverage | Suitability |
|---|---|---|
| **Tool success chain** | 92% of assistant turns include tool calls | ✅ Primary positive signal |
| **Artifact longevity** | 144 file paths introduced; 39% referenced in later turns | ✅ Strong secondary signal |
| **Token efficiency** | 100% (computable on every turn) | ✅ Tiebreaker only, capped at ±0.05 |
| **Self-correction (text-marker draft)** | **0/30 sessions matched the regex** | ❌ Replaced — see §3 |
| **Resolution velocity (text-marker draft)** | **0/30 sessions had a resolution marker** | ❌ Replaced — see §4 |

The two failures are not bugs in the underlying concepts — they are bugs in the **detection methods** of the early draft. Power users do not say "no, that's wrong" or "thanks, perfect" the way casual chat users do. They correct silently by providing more constraints, and they end sessions by walking away. Sections 3 and 4 below have been rewritten with tool-outcome-based detectors that fire on real power-user data.

The text-marker detectors are preserved as fallback layers — they still work for casual chat data and contribute additional confidence when both layers agree. But the primary detection must work without them.

**The big finding**: with the tool-outcome-based detectors and 92% tool-call coverage, an estimated **40–60% of assistant turns** in this dataset would score ≥ 0.7 under the new positive signal model. The current sentiment-based scorer produces ~4% above 0.7. That is a roughly 10–15× improvement in usable training signal from identical raw data.

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

**Link 4 token hygiene:** "referenced tool output" means the user's next turn repeats a *distinctive* token from the tool result — not any 4-character word. Tool output is full of common English words ("this", "with", "from"), and matching those would saturate nearly every conversational session at 1.0, erasing the 0.7/0.9/1.0 gradation. A candidate token only counts as a reference when it (a) is not on the identifier stoplist shared with the artifact-path detector, and (b) looks identifier-like: contains a digit, `.`, `/`, `_`, `-`, or an internal capital, or is at least 8 characters long. Matching is word-boundary-aware (`name` never matches inside `rename`).

**What "tool succeeded" means per tool type:**

| Tool | Success indicator | Discriminative power |
|---|---|---|
| `terminal` | `exit_code == 0`, no stderr-only output, non-empty output | High |
| `read_file` | Content returned, no "not found" or permission errors | **Low** — read_file almost always succeeds (it accounts for ~43% of tool calls in the calibration sample, with a ~99% success rate). The chain links beyond "tool succeeded" — user advanced topic, user referenced output — carry the signal weight for read_file turns. Don't treat a successful read_file alone as strong evidence. |
| `write_file` / `patch` | "patch applied" / no error string in result; ideally a subsequent read confirms the change | High |
| `search_files` | `total_count > 0` AND results referenced in subsequent tool calls | High. **Empty results (`total_count: 0`) count as a soft failure** even though the tool didn't error — the model searched for something that doesn't exist, which usually means the search query was wrong. |
| `web_search` | Results returned, non-empty result set | High. Empty results count as soft failure. |
| `web_extract` | Content extracted, length > 100 chars | High |
| `skill_manage` | Skill created/updated without error | Low frequency in most sessions |
| `execute_code` | Code ran, returned output, no uncaught exceptions | High |

**Soft-failure rule**: a tool that returned cleanly but produced an unusable result (empty search, zero matches, an "ok" with no payload) is a **failure** for scoring purposes. The signal we want is "the agent accomplished something," not "the agent invoked a function without crashing."

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
| **File paths** | Regex for paths in `write_file`, `patch`, `terminal` output, and fenced code blocks | Same path appears in later tool calls or user messages. **Highest-yield artifact type** — 39% reference rate observed on real data. |
| **Function / class identifiers** | snake_case or CamelCase identifiers introduced inside fenced code blocks (filtered against a stoplist of common stdlib names) | Same identifier reappears in later assistant or user turns. Critical for code-focused workloads where the model writes a function, then a later turn extends, tests, or calls it. |
| **Bash commands** | Commands shown in `terminal` tool calls or fenced bash blocks (e.g. `cargo test`, `git status`, `rg "pattern"`) | Same command pattern reappears in later tool calls. Captures the iteration loop where the model proposes a command, the user runs it, and the model proposes a follow-up based on the output. |
| Code blocks | Fenced code blocks in assistant content | Function/variable names reappear in later turns (subsumed by "Function / class identifiers" above) |
| Named plans/concepts | Capitalized terms or quoted names the model introduces | Same terms used by the user in later turns |
| Structured output | JSON, YAML, tables in assistant content | Keys/fields referenced in later discussion |

The first three rows (paths, identifiers, commands) account for the bulk of usable artifact signal in code-heavy workloads. On the calibration data, expanding artifact tracking from "file paths only" to "paths + identifiers + commands" is estimated to push the artifact-longevity coverage from ~39% of turns to ~55–65% of turns.

---

### 3. Productive Self-Correction

**What it detects:** The model made an error, received corrective context, and produced a better response. The corrected response (not the original) is high-quality training data.

**The concept is correct; the detection method needs to be tool-outcome-based.** The early draft of this signal used regex matching on user-text patterns like "no, that's wrong" or "actually, I meant". On the calibration data, **zero of 30 sessions matched any of those patterns**. Power users do not correct verbally — they correct by providing more constraints, narrowing scope, or re-issuing the request with new requirements. The detector has to read tool outcomes, not user phrasing.

**Tool-outcome-based evidence chain:**

```
Assistant turn A
    → Includes one or more tool calls
    → At least one tool failed (per the per-tool success heuristics in §1)
       OR all tools "succeeded" but produced empty / unusable results
       OR no tool calls at all and the user's next turn references the same
       artifacts the model just discussed (semantic redirect)
User turn between A and B
    → Adds new constraints, narrows scope, or re-issues differently
       (heuristic: turn is significantly longer than the original prompt
        AND contains domain terms not in the original)
Assistant turn B
    → Includes tool calls that succeed (per per-tool heuristics)
User after B
    → Does not trigger the same correction pattern again
```

**Scoring:**

Turn A: score 0.0–0.2 (bad — it failed and caused a correction loop)
Turn B: score 0.85–1.0 (good — it incorporated the correction successfully)

Turn B is valuable *because* it demonstrates error recovery in context. The model had the failed attempt, the user's feedback (in whatever form), and then produced the right answer — that's exactly the behavior you want to reinforce. This creates a natural DPO pair when DPO is added in a future phase.

**Detection (tool-outcome-based primary):**

```python
def self_correction_signal(
    turns: list[dict],
    assistant_turn_index: int,
) -> float | None:
    """Returns score for this turn if it's a successful correction.
    Returns None if this turn is not part of a correction pattern.
    """
    prev_asst = find_previous_assistant_turn(turns, assistant_turn_index)
    if prev_asst is None:
        return None

    # Did the previous assistant turn fail at the tool level?
    prev_results = collect_tool_results(turns, prev_asst)
    prev_failed = (
        not prev_results  # no tool calls and we have other failure signal
        or not all_tools_succeeded(prev_results)
        or any(is_soft_failure(r) for r in prev_results)
    )
    if not prev_failed:
        # Fall back to text-marker detection for casual chat data
        if not text_marker_correction(turns, assistant_turn_index):
            return None

    # Did the user's intervening turn add new constraints?
    user_between = find_user_between(turns, prev_asst, assistant_turn_index)
    if user_between is None:
        return None
    if not adds_new_constraints(user_between, prev_asst):
        return None

    # Did THIS turn's tool calls succeed?
    this_results = collect_tool_results(turns, assistant_turn_index)
    if this_results and not all_tools_succeeded(this_results):
        return 0.0  # second attempt also failed

    # Did the user accept (no further failure pattern in next turn)?
    next_user = find_next_user_turn(turns, assistant_turn_index)
    if next_user is None:
        return 0.6  # corrected, but we can't confirm acceptance
    next_asst = find_next_assistant_turn(turns, assistant_turn_index)
    if next_asst is not None:
        next_results = collect_tool_results(turns, next_asst)
        if next_results and not all_tools_succeeded(next_results):
            return 0.4  # user re-corrected, second attempt also failed
    return 0.9
```

**The "adds new constraints" heuristic** is the hard part of this signal. Two layered checks:

1. **Length-and-novelty heuristic**: the user turn between A and B is significantly longer than the original prompt that started the exchange, AND contains domain-specific terms (identifiers, file paths, commands) that did not appear in the original prompt.
2. **Artifact-overlap check**: the user turn references the same files, functions, or commands that the failed assistant turn touched. This catches the common power-user pattern of "I see what you did, here's what you should have done instead" without saying it that way.

Either heuristic firing is sufficient. Both firing increases confidence (the resulting score caps higher).

**Phase 2 enhancement (LLM-judge)**: see the "Phase 2" section below. The hardest case — where the model's tool call succeeded technically but produced a wrong outcome that the user then corrected — requires an LLM judge to detect reliably.

**Important:** The *original* failed turn must be scored as bad (0.0–0.2) by the negative signal detectors. The self-correction signal only applies to the *response after the correction*.

---

### 4. Tool Chain Resolution (Credit Assignment)

**What it detects:** Which turns in a completed task actually contributed to the successful outcome.

**The concept is correct; the resolution-detection method needs to be tool-chain-based.** The early draft of this signal detected resolution by matching user-text patterns like "thanks" or "perfect". On the calibration data, **0 of 30 sessions had any such marker**. Power users do not verbally acknowledge completion — they get what they wanted and walk away. Resolution has to be inferred from tool outcomes and conversational silence, not from politeness phrases.

**Tool-chain-completion-based resolution detection:**

A session is "resolved" if **all** of the following hold:

1. The session has at least N turns total (default N=3 — single-turn sessions are short, not resolved)
2. The last assistant turn includes one or more tool calls
3. **All those tool calls succeeded** (per the per-tool success heuristics in §1, including the soft-failure rule)
4. The user did not send a follow-up message after that final assistant turn
5. No correction pattern (text-marker OR tool-outcome from §3) appeared in the last 3 user turns

This catches the actual completion pattern for power users: "the agent did the thing successfully and I moved on without typing anything else." It does not require any verbal acknowledgment.

Sessions without a clear resolution under these criteria don't get velocity scoring — their turns keep their baseline scores from other signals.

**Optional supplementary signal — followup-session inheritance** (Phase 2): if the user starts a new session within 60 minutes of the previous session ending, AND the new session does not reference any artifacts from the previous session, the previous session is **abandoned** rather than resolved. This distinguishes "user got what they wanted" from "user gave up." The check is heuristic and only downgrades — if both conditions don't hold, the session keeps its resolved status.

**Velocity scoring:**

Once a session is detected as resolved, walk backward from the resolution point and assign per-turn velocity scores. Turns close to the resolution that did not cause corrections score highest. Turns that caused tool failures or required retries score low.

```python
def resolution_velocity(
    turns: list[dict],
    resolution_index: int,
) -> dict[int, float]:
    """Returns {turn_index: velocity_score} for each assistant turn."""
    scores = {}
    remaining_distance = 0

    for i in range(resolution_index, -1, -1):
        turn = turns[i]
        if turn["role"] != "assistant":
            continue

        # Check the immediate aftermath of this assistant turn
        tool_results = collect_tool_results(turns, i)
        next_user = find_next_user_turn(turns, i)
        prev_asst_failed = (
            tool_results and not all_tools_succeeded(tool_results)
        )
        next_user_corrected = (
            next_user is not None
            and (is_correction_text(next_user)
                 or adds_new_constraints(next_user, turns[i]))
        )

        if prev_asst_failed or next_user_corrected:
            # This turn caused a detour — low velocity
            scores[i] = 0.2
            remaining_distance += 2  # penalty: detour costs extra
        elif next_user and is_retry(next_user):
            scores[i] = 0.1
            remaining_distance += 1
        else:
            # Turn advanced the task
            proximity = 1.0 / (1.0 + remaining_distance * 0.2)
            scores[i] = max(0.5, proximity)
            remaining_distance = max(0, remaining_distance - 1)

    return scores
```

The velocity score is highest for turns that are close to the resolution AND that didn't cause detours. Distant successful turns still get credit but lower than the immediately-preceding ones.

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

These weights live in `~/.hermes/config.yaml` under `finetune.scoring.weights_positive` — a dict deliberately separate from the legacy mode's `finetune.scoring.weights`, since the two share key names and must never bleed into each other. At scoring time, w1–w4 are renormalized to sum to 1.0 (w5/manual_override is not part of the composite — manual labels override the score outright), so a perfect session can actually reach 1.0.

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

## Phase 2: LLM-Judge for Implicit Corrections

The tool-outcome-based self-correction detector in §3 catches the common case where the model's tool call **failed** and the user provided more constraints. It misses the harder case where the model's tool call **succeeded technically** but produced a **wrong outcome** that the user then corrected — for example, the model wrote a working but incorrect implementation, and the user redirected with "actually, the function should take a path, not a file handle."

For these cases, an LLM judge is the only reliable detection method. The judge compares two consecutive (assistant, user) pairs and answers: "did the user's second message indicate that the model's previous response was wrong, even though the tool calls succeeded?"

**Configuration**: gated behind `finetune.scoring.use_llm_judge: true` (default false). When enabled:

- Uses Hermes's auxiliary model routing (`agent/auxiliary_client.py`) to dispatch to a fast cheap model (Gemini Flash, Haiku, or a local model via the routing hook)
- **Only invoked on turns where the per-tool success chain scored 0.5–0.7** — the ambiguous middle. Turns with strong signals don't need the judge; turns with strong negative signals are already excluded.
- Cost-bounded: with the 0.5–0.7 gate, judge invocations are limited to ~10–15% of turns in typical data, and each call is one short prompt with a one-token answer.
- Result is cached per (session_id, turn_index) so re-scoring doesn't re-invoke

**Followup-session inheritance** (Phase 2): the resolution detector in §4 can be enhanced by checking whether the user started a new session shortly after the previous one ended. If they did, AND the new session does not reference any artifacts from the previous one, the previous session is downgraded from "resolved" to "abandoned." This catches the failure mode where the user gave up on a session and started fresh rather than completing it.

Both Phase 2 enhancements are **strictly additive** — they tighten the discrimination of an already-functional Phase 1 system. Phase 1 is implementable today against the current data without any new dependencies.

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
