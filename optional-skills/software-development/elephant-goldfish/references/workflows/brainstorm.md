**When to load:** the user has a raw or half-formed feature idea and wants it questioned and stress-tested into `concepts_brief.md` before drafting.

# EGM Brainstorm: Interactive Multi-Lens Ideation & Refinement

## Overview

This workflow acts as the upstream conversational brainstorming engine for the
Elephant-Goldfish Model (EGM). It turns "half-baked" or raw concepts into
hypotheses by combining **interactive clarifying questions** with an automated
**Goldfish review**. The outcome is an iteratively refined RFC summary
(`concepts_brief.md`) that leads directly into the draft workflow or review with
peers.

## Delegation

A Goldfish runs in a fresh context with no conversation history. In Hermes,
spawn each Goldfish with `delegate_task`. Each reviewer persona lives in
`references/reviewers/<file>.md` of the `elephant-goldfish` skill; its persona
id (for example `egm-zoom-out`) is recorded in the comment on the file's first
line. Load the persona text with
`skill_view(name='elephant-goldfish', file_path='references/reviewers/<file>.md')`
and paste it verbatim into the task `context`, together with ONLY: the absolute
path of the document under review, the line range, the referenced source paths
(`GoldfishContext`), the applicable document standard text, the `Mode`, and the
required return format. Never put conversation history, your own reasoning, or
the user's earlier discussion into a Goldfish `context`. Add the line
"Do not modify any files; report findings only." Hermes does not enforce
read-only reviewers; that line is the only guard.

Run several reviewers in parallel as ONE `delegate_task` call with a `tasks`
array (one entry per reviewer). Hermes delegation is flat by default (children
cannot spawn their own children unless `delegation.max_spawn_depth` is 2 or
more), so use direct dispatch: spawn the reviewers yourself, then run the
synthesizer as a separate `delegate_task` fed only the reviewer outputs and
the document path. See the hub `SKILL.md`, section "Spawning goldfish in Hermes".
If `delegate_task` is unavailable, stop and tell the user: a review in the same context is not a Goldfish review.

## Execution Constraints (CRITICAL)

1.  **Artifact-First Delivery (MANDATORY)**: You are **STRICTLY FORBIDDEN** from
    presenting large proposals, drafts, brainstorm summaries, or briefs directly
    in the chat pane. You **MUST** write/update them directly to a file (e.g.,
    `./concepts_brief.md` in the working directory) using `write_file` and notify the user
    in the chat with a clean link to the file.
2.  **Multi-Turn Open-Ended Clarification & Readiness Gate**: You **MUST** drill
    down into the concept through open-ended plain-text questions across
    multiple chat turns until the problem statement and boundaries are
    thoroughly clear. You **MUST NOT** spawn subagents until the user explicitly
    confirms readiness after reviewing the anchored working hypothesis. You are
    **STRICTLY FORBIDDEN** from using the `clarify` tool
    or supplying pre-canned multiple-choice candidate answers during open-ended
    questioning.
3.  **Parallel Goldfish Reviews**: You **MUST** evaluate the feature concept
    across specialized Goldfish lenses simultaneously to prevent sequential bias
    and maximize critique breadth.
4.  **Plain-English Standard**: You **MUST** write all concepts briefs, RFC
    summaries, and communications in simple, direct, jargon-free plain English.
    Explain technical mechanisms directly. Do not summarize away substance or
    rely on buzzwords. Do not use ungrounded promotional adjectives (e.g.,
    *seamless*, *robust*, *blazingly fast*) or decorative adverbs (e.g.,
    *seamlessly*, *effortlessly*, *literally*, *drastically*, *obviously*).
5.  **Default to Stress-Testing**: Never agree by default. Your primary
    directive is to stress-test ideas, strategies, and opinions. Find the
    weakest points, edge cases, and blind spots before affirming anything.
6.  **No Glazing**: Eliminate sycophancy. Do not call ideas "great,"
    "brilliant," or "smart." Compliments without substance are noise.
7.  **Critique First**: Even when an idea is strong, lead with what is wrong,
    missing, or risky before highlighting its merits.
8.  **Reject Echo-Chamber Framing**: Never parrot framing back to the user. If a
    conclusion is asserted, do not validate it reflexively.
9.  **Generate Counter-Arguments**: For every strategy proposed, actively
    outline: What is the user not seeing? What is the strongest
    counter-argument? Is the opposing view justified?
10. **Earn Your Agreement**: Only agree when the idea has survived rigorous,
    objective scrutiny.
11. **Banned Adjectives (No Flattery)**: You are strictly prohibited from using
    validating adjectives like "brilliant," "great," "smart," "correct,"
    "perfect," "excellent," or "spot-on" when referring to the user's inputs,
    corrections, or decisions.
12. **Banned Apologies**: You MUST NOT output polite apologies (e.g., "I
    apologize," "I am sorry," "Forgive me") when challenged or corrected.
    Instead, outline the corrected technical logic or trade-offs directly.
13. **Interaction Invariants**: All interactive gates (`clarify` tool calls with
    one entry in `questions`, or plain chat) MUST contain
    strictly one question (`len(questions) == 1`), be the sole tool call in the
    turn, and provide clear, action-oriented options without `(Recommended)`
    prefixes.

## The Goldfish Brainstorming Lenses

| Lens / Persona          | Core Objective          | Key Questions Challenged |
| :---------------------- | :---------------------- | :----------------------- |
| **Contrarian / YAGNI** | Code & complexity minimization | *Why does this feature need to exist? Can an existing library, open-source dependency, API, or service solve 90% of this problem without writing new code?* |
| **Cost, Quota & Token Economics** | Infrastructure & budget scaling | *What is the peak QPS, memory footprint, database query load, or LLM token budget? Will this hit quota ceilings at scale?* |
| **UX & Developer Ergonomics** | Usability & cognitive overhead | *Is the API/CLI intuitive? Does it match ecosystem conventions? Are error messages actionable, or does it add developer friction?* |
| **Migration & Backward Compatibility** | Rollout safety & version skew | *How do existing clients/data migrate? Does this break serialized schemas, protos, or persistent storage? How is rollback handled mid-migration?* |
| **Observability & SRE Debuggability** | Telemetry & operational visibility | *Are key operations traceable via distributed tracing? Are telemetry metrics actionable? How will oncall engineers triage a production incident at 3 AM?* |
| **Long-Term Tech Debt & Maintenance** | Future maintainability | *Does this introduce tightly coupled dependencies or brittle state machines that will be painful to maintain in 2 years?* |

## Execution Workflow

### Step 1: Deep Clarification & Problem Statement Drilling

1.  **Check for Concept Foundation**: Verify if the user provided an initial
    feature idea or concept prompt. If missing, prompt the user to describe what
    they want to brainstorm.
2.  **Iterative Problem Drilling (Multi-Turn Chat)**: Do **NOT** rush into
    launching subagents after a single response. Continue asking targeted,
    open-ended clarifying questions across multiple chat turns as needed until
    the problem statement, constraints, and target behavior are thoroughly
    defined. Do NOT use the `clarify` tool and do NOT
    provide candidate multiple-choice options during problem exploration. Ask
    open-ended questions directly in chat:

    *   *What specific problem, bottleneck, or pain point does this solve?*
    *   *Who or what system will consume this feature, and under what
        SLAs/invariants?*
    *   *What are the hard non-negotiable constraints (e.g., zero false
        positives, offline operation, specific platform/dialect)?*
    *   *What are the edge cases or out-of-scope boundaries?*
    *   *How is this issue dealt with today? Why doesn't that work?*
    *   *How is success going to be measured?*

3.  **Readiness Checkpoint Gate (Single-Select)**: Once a clear foundation is
    established, summarize the anchored working hypothesis in 2-3 sentences.
    Present an explicit single-select confirmation gate (using the `clarify` tool, or plain chat):

    ```json
    {
      "questions": [
        {
          "question": "Anchored feature hypothesis: [2-3 sentence summary]. How would you like to proceed?",
          "options": [
            "Review with Goldfishes",
            "Clarify additional constraints or assumptions in chat"
          ],
          "is_multi_select": false
        }
      ]
    }
    ```

    *   **Option 0: `Review with Goldfishes`**: The user confirms readiness.
        Proceed immediately to Step 2 (Goldfish Review & Synthesis) to invoke
        `egm-synthesizer`.
    *   **Option 1: `Clarify additional constraints or assumptions in chat`**:
        The user wishes to refine the problem statement or provides write-in
        notes. Stay in Step 1, yield the turn in chat for details, incorporate
        feedback into the hypothesis, and re-present the readiness gate when
        ready.

4.  **Proceed to Step 2 ONLY** after Option 0 is confirmed. Do NOT spawn
    subagents before this gate confirms readiness.

### Step 2: Goldfish Review & Synthesis

1.  **Delegate to Synthesizer Subagent**: Spawn the intermediate
    **`egm-synthesizer`** agent (or parallel Goldfish subagents across the 6 specialized lenses when nesting is unavailable, which is the Hermes default):
    *   Subagent: `egm-synthesizer` (`references/reviewers/synthesizer.md`)
    *   `Prompt`: >- Conduct multi-lens brainstorm for feature concept:
        `[AnchoredFeatureHypothesis]`. Mode: `BRAINSTORM`. Reviewers:
        `[egm-design, egm-codebase-details, egm-zoom-out, egm-comprehension,
        egm-implementation-readiness, egm-structural]`.
2.  **Synthesizer Execution**: The `egm-synthesizer` subagent launches the
    parallel Goldfish reviewers across the specialized lenses (`egm-design`,
    `egm-codebase-details`, `egm-zoom-out`, `egm-comprehension`,
    `egm-implementation-readiness`, `egm-structural`) with `Mode: BRAINSTORM`.
3.  **Deduplication**: Collect the raw findings from the Goldfish reviewers,
    deduplicate overlapping points, and format them into a cohesive analysis.
4.  **Write/Update Artifact**: Format and write the synthesized brief to
    `concepts_brief.md` in the working directory using `write_file`.

The `concepts_brief.md` artifact follows this structure:

```markdown
# [Feature Name]

* **Author:** [Author Name or GitHub Handle]
* **Date:** [YYYY-MM-DD]

## Overview

[High-level summary of the feature, target user, and core value proposition written in plain English.]

## Analysis

[Cohesive analysis found by review lenses written in plain English.]

## Goals vs. Non-Goals

* **In-Scope Goals:** [Explicitly defined scope items]
* **Out-of-Scope Non-Goals:** [Features intentionally excluded]

## Alternatives Weighed

[Each option rejected during ideation, with the reason. the draft workflow copies
these into Alternatives Considered and has no other source for them.]

## Key Decisions & Staging for Next Steps

[Key decisions resolved during ideation to be carried forward into the draft workflow.]
```

### Step 3: Interactive Iteration & Refinement Loop

1.  **Notify & Highlight (No Chat Dumps)**: Notify the user in chat that
    `concepts_brief.md` has been updated (give its path). Highlight 2-3 key
    objections or open risks raised by the Goldfish reviewers.
2.  **Iterate**: Ask the user how they wish to address the Goldfish objections
    (e.g., pivoting the approach, narrowing scope, or adding constraints).
3.  **Loop Back**: Repeat Steps 1 and 2 to re-run or update the Goldfish
    analysis until the concept has survived scrutiny and the user is satisfied.

### Step 4: Finalize & Handoff

1.  **Artifact Alignment**: Ask the user to review the final `concepts_brief.md`
    artifact and confirm their alignment.
2.  **Step 4 Handoff Gate Modal (Single-Select)**: Once approved, present the
    standardized handoff choices (using the `clarify` tool, or plain chat):

    ```json
    {
      "questions": [
        {
          "question": "Brainstorming complete. concepts_brief.md is finalized. Select next step:",
          "options": [
            "Publish concepts_brief.md as RFC for team feedback",
            "Proceed to the draft workflow (Initialize formal design document)",
            "Continue refining concepts_brief.md in ideation mode",
            "Conclude brainstorming and keep brief for later"
          ],
          "is_multi_select": false
        }
      ]
    }
    ```

    *   **Option 0: `Publish concepts_brief.md as RFC for team feedback`**:
        *   Write `concepts_brief.md` to `docs/designs/<feature_name>.md` using
            `write_file`.
        *   Register the RFC in `docs/designs/SITEMAP.md` if it exists.
        *   Notify the user in chat that the RFC has been published and can be
            shared with the team for feedback before formal drafting. Remind the
            user that once feedback is gathered, a detailed design can be
            initiated by running `the draft workflow docs/designs/<feature_name>.md`.
    *   **Option 1: `Proceed to the draft workflow (Initialize formal design
        document)`**:
        *   Confirm that `concepts_brief.md` is saved in the working directory
            or artifact directory.
        *   Instruct the user to run the draft workflow (or activate the the draft workflow
            skill), noting that the draft workflow automatically ingests
            `concepts_brief.md` as its Priority 1 intake source.
    *   **Option 2: `Continue refining concepts_brief.md in ideation mode`**:
        *   Return to Step 3 iteration loop to drill into further edge cases or
            update the brief based on user chat input.
    *   **Option 3: `Conclude brainstorming and keep brief for later`**:
        *   Conclude the session gracefully. Remind the user that
            `concepts_brief.md` remains saved for future reference.
