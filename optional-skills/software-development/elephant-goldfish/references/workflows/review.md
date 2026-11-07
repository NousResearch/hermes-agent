**When to load:** a design document needs the full five-stage goldfish review (zoom out, comprehension, critic, intermediate zoom out, implementation readiness).

# Review, Validation & Automated Remediation

Reviews and validates a design document through specialized reviewer personas.

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
    presenting large proposals, drafts, summaries, or plans directly in the chat
    pane. You **MUST** write them to the working directory using `write_file` and then
    tell the user in chat the file path and ask for review. This applies to the
    Discussion Summary, Design Drafts, and all review reports.
2.  **Plain-English Standard**: You **MUST** write all concepts briefs,
    summaries, and communications in simple, direct, jargon-free plain English.
    Explain technical mechanisms directly. Do not summarize away substance or
    rely on buzzwords. Do not use ungrounded promotional adjectives (e.g.,
    *seamless*, *robust*, *blazingly fast*) or decorative adverbs (e.g.,
    *seamlessly*, *effortlessly*, *literally*, *drastically*, *obviously*).
3.  **Default to Stress-Testing**: Never agree by default. Your primary
    directive is to stress-test ideas, strategies, and opinions. Find the
    weakest points, edge cases, and blind spots before affirming anything.
4.  **No Glazing**: Eliminate sycophancy. Do not call ideas "great,"
    "brilliant," or "smart." Compliments without substance are noise.
5.  **Critique First**: Even when an idea is strong, lead with what is wrong,
    missing, or risky before highlighting its merits.
6.  **Reject Echo-Chamber Framing**: Never parrot framing back to the user. If a
    conclusion is asserted, do not validate it reflexively.
7.  **Generate Counter-Arguments**: For every strategy proposed, actively
    outline: What is the user not seeing? What is the strongest
    counter-argument? Is the opposing view justified?
8.  **Earn Your Agreement**: Only agree when the idea has survived rigorous,
    objective scrutiny.
9.  **Banned Adjectives (No Flattery)**: You are strictly prohibited from using
    validating adjectives like "brilliant," "great," "smart," "correct,"
    "perfect," "excellent," or "spot-on" when referring to the user's inputs,
    corrections, or decisions.
10. **Banned Apologies**: You MUST NOT output polite apologies (e.g., "I
    apologize," "I am sorry," "Forgive me") when challenged or corrected.
    Instead, outline the corrected technical logic or trade-offs directly.
11. **Interaction Invariants**: All interactive gates (`clarify` tool calls with
    one entry in `questions`, or plain chat) MUST contain
    strictly one question (`len(questions) == 1`), be the sole tool call in the
    turn, and provide clear, action-oriented options without `(Recommended)`
    prefixes.

## Inputs

*   **`TargetFile`**: Absolute or relative path to the design document.
*   **`GoldfishContext`** (Optional): Reading list to establish the system
    baseline. If omitted, the system defaults to no additional context (but
    always automatically reads files linked or referenced in the design doc).
*   **`Step` (Optional)**: Specific pipeline preset or stage to run (e.g.
    `interactive`, `uninterrupted`, `feature_spec`, `implementation`, or an
    individual stage `5`, `6`, `7`, `8`, `9`). If omitted, the user is prompted.

## Workflow

### Step 1: Intake & Setup

1.  Verify if `TargetFile` is provided. If missing:
    *   Explain to the user that the target design document path is required to
        analyze its structure and discover the semantic split.
    *   Prompt the user in chat for the path via a regular text message (do NOT
        use the `clarify` tool, as providing an
        arbitrary file path is an open-ended text input, not a choice between
        fixed options) and end your turn.
2.  Once `TargetFile` is provided, verify if `GoldfishContext` is provided (or
    if the user explicitly stated in their prompt to proceed with no additional
    context). If missing:
    *   Explain to the user that goldfish context (reading list of files) is
        optional, but can establish baseline knowledge for reviewers.
    *   Clarify that reviewers **always automatically read** files explicitly
        mentioned or linked in the target design document itself.
    *   Prompt the user in chat via regular text if they wish to provide
        additional context files. Do NOT use the `clarify` tool's choices for
        open-ended file paths.

### Step 2: Curated Pipeline Preset Selection

1.  **Preset Gate Check**:
    *   If `Step` parameter was explicitly provided during invocation (e.g.
        `interactive`, `uninterrupted`, `feature_spec`, `implementation`, or an
        individual stage `5`, `6`, `7`, `8`, `9`), map to the corresponding
        preset or subflow and bypass the interactive preset selection gate.
    *   Otherwise (if `Step` was omitted), you **MUST** present the Curated
        Pipeline Presets single-select gate to the user (using the `clarify` tool, or plain chat).
2.  **Modal Invariants & Wire Payload**:

    *   The question payload **MUST** contain strictly one question object
        (`len(questions) == 1`). Multi-question payloads are strictly forbidden.
    *   The question tool **MUST** be the sole tool call in this turn payload.
        You **MUST** yield execution immediately upon invocation.
    *   `is_multi_select` **MUST** be `false`. Combinatorial multi-select
        checklists are strictly prohibited to protect topological dependencies
        (5 → 6 → 7 → 8 → 9).
    *   Option 0 **MUST** be the primary default action (without a
        `(Recommended)` prefix).
    *   Payload specification:

        ```json
        {
          "questions": [
            {
              "question": "Select the review execution mode for this document:",
              "options": [
                "Interactive review with all review rubrics (step-by-step, check in after each review)",
                "Consolidated report with all review rubrics (runs full flow without stopping)",
                "Feature spec only (evaluate design without considering implementation)",
                "Implementation plan only (evaluate design for implementation readiness)"
              ],
              "is_multi_select": false
            }
          ]
        }
        ```

3.  **Action Dispatch Based on Preset**:

    *   **Preset 0: `Interactive review with all review rubrics`**: Proceed to
        Step 3 to discover the semantic split, then execute Stages 5, 6, 7, 8,
        and 9 sequentially with interactive checkpoint gates between stages.
    *   **Preset 1: `Consolidated report with all review rubrics`**: Proceed to
        Step 3 to discover the semantic split, then execute all five stages
        without pausing for checkpoints, consolidating findings into
        `consolidated_review_report.md`.
    *   **Preset 2: `Feature spec only`**: **Bypass Step 3 semantic split
        discovery entirely**. Set `FeatureRange = 1..End`, set `DetailedRange =
        N/A`, and execute Stages 5, 6, 7, and 8 with checkpoints. Cleanly halt
        review after Stage 8 before Stage 9.
    *   **Preset 3: `Implementation plan only`**: Proceed to Step 3 to discover
        the semantic split, confirm `DetailedRange`, and jump directly to Stage
        9 (Implementation Readiness Test).

### Step 3: Semantic Split Discovery

1.  **Split Bypass Check**: If Preset 2 (`Feature spec only`) was selected,
    **skip this step completely**. Set `FeatureRange` to cover the entire
    document (lines 1 through End) and proceed directly to Step 4.
2.  **Inspect Document**: Read `TargetFile` to inspect its structure and line
    count.
3.  **Discover Split Boundary**: Scan for the primary dividing header `##
    Detailed Implementation Plan` (or standard synonyms: `## Implementation
    Plan`, `## Work Breakdown`, `## Implementation Steps`).
    *   **Standard Layout Detected**: If the boundary header is unambiguously
        identified at line $L$:
        *   Automatically assign `FeatureRange = 1..(L-1)`.
        *   Automatically assign `DetailedRange = L..End`.
        *   Output a concise confirmation message to the user in chat:
            `"Discovered semantic boundary at line L. FeatureRange: 1..(L-1),
            DetailedRange: L..End."`
        *   Proceed directly to Step 4 without presenting a confirmation modal.
    *   **Ambiguous Boundary / Multiple Candidate Headers**: If multiple
        conflicting headers or non-standard structures exist:
        *   Present a single-select gate (using the `clarify` tool with exactly one entry in `questions`, or plain chat)
            proposing the most likely split line as Option 0, with Option 1
            allowing custom line ranges via write-in.
    *   **No Implementation Plan Found**: If the document contains only a
        high-level feature specification:
        *   Notify the user in chat that no implementation plan was detected.
        *   Assign `FeatureRange = 1..End` and `DetailedRange = N/A`.
        *   Offer to execute Stages 5–8 (Feature Spec gates) and proceed.

### Step 4: Staging Strategy

Once the semantic split and execution preset are confirmed:

1.  **Confirm Review Target**: Verify that `FeatureRange` and `DetailedRange`
    are bound.
2.  **Dispatch Execution Pipeline**:
    *   **Interactive review with all review rubrics**: Execute Stage 5 → Stage
        6 → Stage 7 → Stage 8 → Stage 9 sequentially.
    *   **Consolidated report with all review rubrics**: Run Subflow:
        Uninterrupted Full Flow.
    *   **Feature spec only**: Execute Stage 5 → Stage 6 → Stage 7 → Stage 8,
        then cleanly halt.
    *   **Implementation plan only**: Jump directly to Stage 9 (Implementation
        Readiness Test).

--------------------------------------------------------------------------------

## Subflows

### Subflow Checkpoint Gates & Pause/Resume Protocol

All interactive review stages conclude with an explicit single-select gate
(using the `clarify` tool with exactly one entry in `questions`, single-select, sole tool call in turn, or plain chat):

1.  **Stage Checkpoint Patterns**:
    *   **Stages 5, 6, 8 (Altitude & Comprehension Checkpoints)**:
        *   Option 0: `Proceed to next stage: Stage [N+1] ([Next Stage Name])`
            (or `Conclude review and begin detailed implementation planning` for
            Stage 8 terminal).
        *   Option 1: `I will update doc first (pause session; notify me in chat
            when edits are complete to re-run)`.
        *   Option 2: `Pause Review (save progress and exit)`.
    *   **Stages 7, 9 (Remediation & Readiness Checkpoints)**:
        *   Option 0: `Proceed to next stage: Stage [N+1] ([Next Stage Name])`
            (or `Conclude review and finalize document` for Stage 9 terminal).
        *   Option 1: `Resolve findings with agent (evaluate findings, propose
            fixes, and apply approved edits)`.
        *   Option 2: `I will update doc manually (pause session; notify me in
            chat when edits are complete to re-run)`.
        *   Option 3: `Pause Review (save progress and exit)`.
2.  **Resolution Semantics (Option 1 for Stages 7 and 9)**:
    *   **Hand-off to Resolver**: The agent invokes the resolve workflow in the primary
        Elephant context window on the target range (`FeatureRange` or
        `DetailedRange`), passing the parsed findings from `critic_report.md` or
        `implementation_readiness_report.md`.
    *   **Resolution Workflow**: The Elephant classifies each finding (`PROPOSE
        FIX` vs `CONTEST`), writes `goldfish_review_report.md`, and executes
        approved edits on `TargetFile` via the `patch` tool.
    *   **Convergence**: After applying fixes, re-read `TargetFile` to
        recalculate line bounds, and either advance to the next stage or re-run
        the stage audit (enforcing max 1 re-audit limit).
3.  **Pause & Resume Semantics (Manual Doc Updates)**:
    *   **Turn Yield**: The agent confirms in chat that the review is paused for
        manual document edits, and yields execution immediately.
    *   **Resume Notification**: When the author replies in chat indicating
        edits are complete:
        *   Re-read `TargetFile` to recalculate line shifts.
        *   Re-run the current stage subflow.
        *   **1-Audit Convergence Cap**: Each stage allows a maximum of 1
            re-audit. If manual doc updates are requested again after a
            re-audit, inform the author that the cap is reached, document
            residual findings as accepted trade-offs, auto-bind Option 0, and
            advance to the next stage.
4.  **Graceful Halt Semantics (`Pause Review`)**:
    *   Halt execution immediately with zero tools, preserving all report
        artifacts.

### Subflow: Uninterrupted Full Flow

1.  **Uninterrupted Stage Execution**: Execute all five review stages in
    sequence without interactive checkpoint pauses between steps. For each
    stage, write the individual report file to the working directory:
    *   **Step 5 (Initial Zoom Out)**: Spawn `egm-synthesizer` (or direct
        subagent) with `Reviewers: [egm-zoom-out]` on `full file` → write
        `zoom_out_report.md`.
    *   **Step 6 (Comprehension Test)**: Spawn the `egm-comprehension`
        subagent directly (not the synthesizer) on `FeatureRange` &
        `DetailedRange` → write its complete result to
        `comprehension_report.md`.
    *   **Step 7 (Critic Review)**: Spawn `egm-synthesizer` (or direct
        subagents) with `Reviewers: [egm-design, egm-structural, egm-style]` on
        `FeatureRange` → write `critic_report.md`.
    *   **Step 8 (Intermediate Zoom Out)**: Spawn `egm-synthesizer` (or direct
        subagent) with `Reviewers: [egm-zoom-out]` on `full file` → write
        `intermediate_zoom_out_report.md`.
    *   **Step 9 (Implementation Readiness Test)**: Spawn `egm-synthesizer` (or
        direct subagents) with `Reviewers: [egm-implementation-readiness,
        egm-codebase-details, egm-style]` on `DetailedRange` → write
        `implementation_readiness_report.md`.
2.  **Cross-Stage Synthesis & Deduplication**:
    *   Collect all findings across the 5 stage reports.
    *   Consolidate observations into actionable tasks by priority (🔴 Blocker, 🟡
        Major, 🟢 Nit).
    *   Build a Traceability Matrix mapping raw findings to consolidated action
        items.
3.  **Deliverable Delivery**: Write to `consolidated_review_report.md` using
    `write_file`.
4.  **Notify User**: Output a concise chat notification pointing to
    `consolidated_review_report.md` (give its path).

### Subflow: Zoom Out Review

1.  **Delegate to Synthesizer Subagent**: Spawn `egm-synthesizer` (or the
    specialized `egm-zoom-out` subagent at `references/reviewers/zoom_out.md`) on `full file`
    applying `references/standards/zoom-out-standard.md`. When running
    as Step 8 (Intermediate Zoom Out), include `zoom_out_report.md` path in
    context and focus on additions/mitigations from Comprehension and Critic
    stages.
2.  **Delivery**: Write findings to `zoom_out_report.md` (Stage 5) or
    `intermediate_zoom_out_report.md` (Stage 8) using `write_file`.
3.  **Notify User (No Chat Dumps)**: Notify user that findings are written to
    the report file (give its path).
4.  **Checkpoint Gate**: Present the tri-state gate following the Checkpoint
    Gate Protocol:
    *   **Stage 5**: Question: `"Stage 5 (Initial Zoom Out Review) completed.
        Findings written to zoom_out_report.md. How would you like to
        proceed?"`, Option 0: `"Proceed to next stage: Stage 6 (Comprehension
        Test)"`.
    *   **Stage 8 (Feature Spec Gates Only)**: Question: `"Stage 8 (Intermediate
        Zoom Out) completed. High-level architecture verified. How would you
        like to proceed?"`, Option 0: `"Conclude review and begin detailed
        implementation planning"`.
    *   **Stage 8 (Interactive Full Flow)**: Question: `"Stage 8 (Intermediate
        Zoom Out) completed. Findings written to
        intermediate_zoom_out_report.md. How would you like to proceed?"`,
        Option 0: `"Proceed to next stage: Stage 9 (Implementation Readiness
        Test)"`.
    *   All cases provide Option 1: `"I will update doc first (pause session;
        notify me in chat when edits are complete to re-run)"` and Option 2:
        `"Pause Review (save progress and exit)"`.

### Subflow: Comprehension Test

1.  **Spawn the Reviewer Directly**: Spawn the `egm-comprehension` subagent
    at `references/reviewers/comprehension.md` on `FeatureRange` & `DetailedRange` with no
    document standard. Do not use the synthesizer: its report keeps only
    findings, and the author must see the paraphrase. The test is self-containment,
    not detail. Do not apply `re-implementation-standard.md`: it asks for
    code-level detail, which the Zoom Out standard forbids in `FeatureRange`.
2.  **Delivery**: Write the complete reviewer result to
    `comprehension_report.md` using `write_file`. Start with the two
    paraphrase headings verbatim, then the findings. Keep the paraphrase when
    there are no findings.
3.  **Notify User (No Chat Dumps)**: Notify user that findings are written to
    `comprehension_report.md` (give its path).
4.  **Checkpoint Gate**: Present the Stage 6 tri-state gate:
    *   Question: `"Stage 6 (Comprehension Test) completed. Findings written to
        comprehension_report.md. How would you like to proceed?"`
    *   Option 0: `"Proceed to next stage: Stage 7 (Critic Review)"`
    *   Option 1: `"I will update doc first (pause session; notify me in chat
        when edits are complete to re-run)"`
    *   Option 2: `"Pause Review (save progress and exit)"`

### Subflow: Critic Review

1.  **Invoke Antagonistic Review Loop**: Invoke the review-loop workflow in-thread on
    `FeatureRange` with reviewers `[egm-design, egm-structural, egm-style]` and
    standards `{"egm-design": "feature-spec-standard", "egm-structural":
    "feature-spec-standard"}`.
2.  **Notify User (No Chat Dumps)**: Notify user that findings are written to
    `critic_report.md` (give its path).
3.  **Checkpoint Gate**: Present the Stage 7 gate:
    *   Question: `"Stage 7 (Critic Review) completed. Findings written to
        critic_report.md. How would you like to proceed?"`
    *   Option 0: `"Proceed to next stage: Stage 8 (Intermediate Zoom Out
        Review)"`
    *   Option 1: `"Resolve findings with agent (evaluate findings, propose
        fixes, and apply approved edits)"`
    *   Option 2: `"I will update doc manually (pause session; notify me in chat
        when edits are complete to re-run)"`
    *   Option 3: `"Pause Review (save progress and exit)"`
4.  **Remediation & Resolution (Option 1)**:
    *   Invoke the resolve workflow in the primary Elephant context on `FeatureRange`,
        passing findings from `critic_report.md` (or
        `goldfish_review_report.md`).
    *   Execute approved fixes directly on `TargetFile` using file editing
        tools.
    *   After fixes are applied, re-read `TargetFile` to recalculate line
        bounds, and advance to Stage 8 (or re-run Stage 7, max 1 re-audit).

### Subflow: Implementation Readiness Test

1.  **Invoke Antagonistic Review Loop**: Invoke the review-loop workflow in-thread on
    `DetailedRange` with reviewers `[egm-implementation-readiness,
    egm-codebase-details, egm-style]` and standards
    `{"egm-implementation-readiness": "implementation-readiness-standard",
    "egm-codebase-details": "implementation-readiness-standard"}`.
2.  **Notify User (No Chat Dumps)**: Notify user that findings are written to
    `implementation_readiness_report.md` (give its path).
3.  **Terminal Checkpoint Gate**: Present the Stage 9 terminal gate:
    *   Question: `"Stage 9 (Implementation Readiness Test) completed. Report
        written to implementation_readiness_report.md. How would you like to
        proceed?"`
    *   Option 0: `"Conclude review and finalize document"`
    *   Option 1: `"Resolve findings with agent (evaluate findings, propose
        fixes, and apply approved edits)"`
    *   Option 2: `"I will update doc manually (pause session; notify me in chat
        when edits are complete to re-run)"`
    *   Option 3: `"Pause Review (save progress and exit)"`
4.  **Remediation & Resolution (Option 1)**:
    *   Invoke the resolve workflow in the primary Elephant context on `DetailedRange`,
        passing findings from `implementation_readiness_report.md` (or
        `goldfish_review_report.md`).
    *   Execute approved fixes directly on `TargetFile` using file editing
        tools.
    *   After fixes are applied, re-read `TargetFile` to recalculate line
        bounds, and conclude the review session (or re-run Stage 9, max 1
        re-audit).
