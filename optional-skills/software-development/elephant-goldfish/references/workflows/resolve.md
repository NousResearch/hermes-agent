**When to load:** goldfish review findings exist and the elephant (the session that wrote the document) must classify, defend, or apply fixes.

# Goldfish Review: Resolve

## Overview

This workflow instructs the agent on how to operate within the primary Elephant
context window to parse flat deficiency reports from concurrent Goldfish
reviewers. It defends accurate design choices, applies automated textual
repairs, and manages interactive negotiation checkpoints.

## Prerequisites & Context Strategy

*   **Context Recommendation (Elephant Memory)**: This workflow should be run in
    the persistent conversation session that generated the target document (the
    Elephant) to recall earlier reasoning and craft higher-quality fixes.

## Inputs

The caller or orchestration driver provides exactly these parameters:

*   **`TargetFile`**: Absolute or relative path to target document.
*   **`LineRange`**: Exact line numbers (or `full file`).
*   **`FindingsList`**: Aggregated flat list of findings from Goldfish
    reviewers.

## Execution Workflow

### Step 1: Intake Validation & Gather Context

1.  **Hybrid Execution UX**: Detect if execution arguments are present:

    *   *Automated Mode*: If all parameters are provided, execute silently as a
        subprocess.
    *   *Standalone Human Mode*: If parameters are omitted, trigger a friendly
        onboarding flow. Prompt for one piece of input at a time, waiting for
        the user's response after each turn. Once all inputs are specified,
        display the final setup and verify the user is ready to continue.

2.  Read `TargetFile` at `LineRange` by reading it and parse the `FindingsList`.

### Step 2: Evaluate & Categorize

Give each finding exactly one category:

*   **CONTEST**: The critique is inaccurate or relies on a misunderstanding of
    the codebase. Cite evidence that the reviewer could read: a file and
    symbol, or a decision recorded in the document.
*   **ESCALATE**: The critique is inaccurate, but the evidence is outside every
    file the reviewer could read (for example, a team convention or a verbal
    decision). State that fact. Do not fix the finding. Give it to the user.
*   **PROPOSE FIX**: The critique is valid. Define precise textual replacements
    or structural formatting adjustments to resolve the gap.
*   **DECLINE**: The finding is a Nit. Record it. Do not fix it. You can use
    PROPOSE FIX for a Nit if the fix is worth it.

Apply these rules:

*   **A contest must be checkable.** "This was intentional" is not evidence.
    If you cannot cite evidence the reviewer could read, use ESCALATE or
    PROPOSE FIX.
*   **Do not add context only to satisfy a reviewer.** Add context when a human
    reader of the document needs it. If only a zero-context subagent needs it,
    use ESCALATE.
*   **Do not contest a comprehension gap.** If a fresh reader could not follow
    the document, the document is missing context.
*   **Do not change a severity.** If you think a Blocker is a Nit, CONTEST it
    and state the reason.
*   You wrote the document under review. A wish to contest is not evidence
    that the reviewer is wrong.

### Step 3: Report Fixes

Construct the final report and save it named
`goldfish_review_report.md` in the working directory
using `write_file`. Use this layout:

```markdown
# Goldfish Annotated Review Report

**Target:** `[TargetFile]` (`[LineRange]`)

## Prioritized Findings & AI Suggestions

{{range findings, ordered by Severity}}
### [Issue 1] [`[ReviewerPersona]`] - Severity: [High | Medium | Low]
*   **Affected Scope:** [Lines Start-End]([RelativePath]#L[Start]-L[End]) ([Sub-header])
*   **Discrepancy Description:** [copied from FindingsList]
*   **Rationale:** [copied from FindingsList]
*   ***AI Suggestion:*** [CONTEST | ESCALATE | PROPOSE FIX | DECLINE]
*   ***Justification:*** [Detailed reason]
*   ***Proposed Edit:***
    *   Location: [Line/Header]
    *   Change: [Precise replacement string or deletion note]
{{end}}
```

### Step 4: Interactive Resolution Gate

If any finding is `ESCALATE`, list those findings first in the chat message.
They need the user's knowledge, not only approval.

Before proceeding, output a visible chat message to the user: *"I have compiled
the audit findings into `goldfish_review_report.md`. Please review the findings there and choose how you would like to
proceed:

-   **Comment directly on the file**: Review `goldfish_review_report.md` in
    chat or the file and leave comments/edits to approve, reject, or contest
    specific items.
-   **Auto accept**: Reply "auto accept" to batch-apply all proposed fixes
    (skipping contested items).
-   **Continue to next phase**: Reply "continue" or "no changes" to proceed
    without applying fixes.
-   **Interactive breakdown**: Reply "interactive breakdown" for a guided,
    item-by-item walkthrough in chat with detailed explanations for each
    proposed edit or contested finding before making decisions.
-   **Custom instruction / Wildcard request**: Reply with any custom
    instruction, question, or out-of-band request."*

You **MUST** immediately yield active execution by calling **zero tools** in
this turn payload. You must stop execution and go idle, waiting for the human
developer to comment on/edit the artifact or reply in the chat.

Interpret the user's action/response as follows:

-   **"Auto accept" / "Accept all suggestions" / User clicks "Proceed" without
    contesting**: Apply all fixes where the AI suggestion was `PROPOSE FIX`
    directly to `TargetFile` at `LineRange` in a single batch using
    `patch`. Any items categorized as `CONTEST`, `ESCALATE`, or
    `DECLINE` are explicitly bypassed. Then proceed to Step 5.
-   **"Continue to next phase" / "No changes"**: Do not apply any automated
    fixes. Then proceed to Step 5.
-   **"Interactive breakdown"**: Enter a sequential breakdown loop over ALL
    finding items (all categories except `DECLINE`):

    1.  For each item `i`, output a visible chat message formatting the full
        finding details and explanation in rich markdown:

        ```markdown
        ### Breakdown Item [i]: [Issue Title] [`[ReviewerPersona]`]
        * **Scope:** [Lines Start-End]([RelativePath]#L[Start]-L[End]) ([Sub-header])
        * **Description:** [Description]
        * **AI Suggestion:** [CONTEST | ESCALATE | PROPOSE FIX]
        * **Explanation & Justification:** [Detailed reason and background explaining why the fix is proposed or why the finding is contested]
        * **Proposed Edit:** [Precise replacement text or deletion note]
        ```

        Then prompt the user in chat for their decision on this item (calling
        **zero tools** and going idle):

        *   If `PROPOSE FIX`: Ask the user to reply with **Accept proposed
            fix**, **Skip / ignore**, or **Discuss**.
        *   If `CONTEST`: Ask the user to reply with **Skip / agree with
            contest**, **Overrule contest (propose fix)**, or **Discuss**.
        *   If `ESCALATE`: Ask the user to reply with **Confirm (no change)**,
            **Add the fact to the document**, or **Discuss**.

    2.  Record the user's decisions in an internal memory checklist (e.g., *Item
        1: Accepted*, *Item 2: Ignored*, *Item 3: Overruled*). **Do NOT call any
        the `patch` tool during this sequential loop.**

    3.  Once all items have been decided, summarize the accepted edits, execute
        them using `patch`, and proceed to Step 5.

-   **Custom instruction / Out-of-band edit**: If the user provides targeted
    instructions, apply them and proceed.

### Step 5: Post-Resolution Synchronization & Exit

1.  **Re-verify Lines**: If changes were made to `TargetFile`, verify that the
    file remains structurally coherent and valid markdown.
2.  Mark the report as resolved (add a `Status: resolved` line at the top).
3.  **Handoff**: Exit cleanly back to the caller (such as the review-loop workflow)
    with the count of accepted fixes, contested items, escalated items,
    declined items, and remaining unaddressed issues.
