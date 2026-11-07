**When to load:** a document needs a configured group of goldfish reviewers run in repeated rounds until findings converge.

# Goldfish Review: Loop (Orchestration Driver)

## Overview

This workflow acts as the central orchestration engine driving the multi-agent
review loop. It coordinates two distinct agent memory models:

*   **The Goldfish (Adversarial Auditors)**: Subagents spawned in isolated,
    fresh context windows to catch unstated assumptions without historical bias.
*   **The Elephant (Persistent Resolver)**: The primary agent context holding
    institutional knowledge to evaluate findings and apply approved repairs.

The driver manages argument intake, resolves paths for reviewers and standards,
unconditionally re-parses target line ranges before every iteration, spawns
concurrent specialized Goldfish auditors based on configuration, and coordinates
with the Elephant context to resolve findings.

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

## Inputs

The caller or user provides exactly these parameters:

*   **`TargetFile`**: Absolute or relative path to target document.
*   **`SemanticTarget`**: Descriptive targeting string (e.g., *"Target the H4
    block with SS-ID 'S1.12'"*), or `full file`.
*   **`Reviewers`**: List of reviewer names (shortnames or paths).
*   **`Standards`**: Mapping of reviewer names to standard names (shortnames or
    paths).
*   **`GoldfishContext`**: A reading list (source files, directories, or
    documentation URLs) fed to reviewers to establish their baseline.

## Path & Standard Resolution Logic

When resolving names in `Reviewers` and `Standards`:

*   **Reviewer Shortnames**: Map reviewer shortname (e.g., `design_critic`,
    `structural`, `style`, `implementation_readiness`,
    `codebase_details`, `comprehension`, `zoom_out`) to `references/reviewers/{name}.md`.
    The persona file defines the reviewer's core mandates, audit vectors, and
    output format.
*   **Standard Shortnames**: Map standard shortname (e.g., `feature-spec-standard`,
    `implementation-readiness-standard`, `zoom-out-standard`, `heilmeier-standard`,
    `re-implementation-standard`) to `references/standards/{name}.md`.
*   **Custom Paths**: If the name contains a slash (`/`), treat it as an
    absolute or workspace-relative path and read it directly.

## Execution Workflow

### Step 1: Intake & Configuration UX

Evaluate the provided inputs (`TargetFile`, `SemanticTarget`, `Reviewers`,
`Standards`, `GoldfishContext`).

1.  **Identify Missing Inputs**: Determine which of the required parameters are
    missing.
2.  **Interactive Onboarding**:
    *   If **all** inputs are present, proceed to Step 2.
    *   If **some or all** inputs are missing, interactively prompt the user
        **only for the missing inputs**, one at a time via chat messages.
3.  **Reviewer Recommendations**:
    *   *Feature Specs*: Recommend `design_critic` (mapped to
        `references/standards/feature-spec-standard.md`) and
        `structural` (mapped to `references/standards/feature-spec-standard.md`).
    *   *Implementation Plans*: Recommend `implementation_readiness`
        (mapped to `references/standards/implementation-readiness-standard.md`)
        and `codebase_details`.
4.  **Confirm Configuration**: Display the consolidated plan (Target, Reviewers,
    and mapped Standards) and ask the user to confirm before proceeding.

### Step 2: The Sharpening Loop (Iterative Cycle)

Execute the following sequence in a continuous loop:

#### 2.1 Unconditional Target Resolution

Before every iteration, unconditionally re-scan `TargetFile` by reading it
and using the stable `SemanticTarget` string (e.g., locating the target header) to
dynamically determine the updated absolute start and end line numbers. This
ensures absolute accuracy even if the file was edited in prior rounds. (If `full
file` mode is specified, confirm full-file line bounds).

#### 2.2 Review Swarm Execution

1.  **Launch Subagent Swarm**:
    Spawn subagents to execute the review:
    *   **Hierarchical Dispatch (Synthesizer)**: Spawn the intermediate
        synthesizer subagent (`references/reviewers/synthesizer.md` / `egm-synthesizer`).
        The synthesizer spawns the Goldfish review swarm in
        parallel, deduplicates findings, writes the report artifact, and
        returns a single summary message to avoid context pollution in the root
        conversation. Pass it the Failed Reviewers rule below. Its summary must
        give a status, `completed` or `not run`, for each requested reviewer.
    *   **Direct Dispatch (Flat Swarm)**: Alternatively, spawn each requested
        reviewer subagent in `Reviewers` directly.
        Instruct each subagent to first read its persona definition in
        `references/reviewers/{name}.md`, then audit `TargetFile` at the resolved line range
        against its designated standard in `references/standards/`,
        supplying `GoldfishContext`.
    *   Each reviewer produces findings with severity (🔴 Blocker, 🟡 Major, 🟢
        Nit), line numbers, discrepancy descriptions, and rationales.
2.  **Consolidate Findings**: Collect all findings across reviewers into a single
    flat `FindingsList`. Give each finding a stable id
    (`<reviewer>-<round>-<n>`) and record its reviewer, severity, and affected
    scope.
3.  **Failed Reviewers**: A reviewer that times out, fails, or returns no report
    did not review. Retry it once. If it fails again, record it as **not run**
    in the report. Never count it as a reviewer with zero findings. If the
    synthesizer summary has no status for a requested reviewer, record that
    reviewer as not run.

#### 2.3 Aggregated Resolution Hand-off

1.  Pass `TargetFile`, the active line range, and the aggregated `FindingsList`
    to the **the resolve workflow** workflow.
2.  Execute the resolution workflow in the primary Elephant context:
    *   Categorize findings into `CONTEST` vs. `PROPOSE FIX`.
    *   Save `goldfish_review_report.md` with `write_file`.
    *   Present the interactive decision gate (Auto accept, Continue, Interactive
        breakdown).
    *   Apply approved text repairs to `TargetFile` using `patch`.

#### 2.4 Iteration Check & Turn Boundary

Fresh reviewers sample differently each round, so a round with zero findings
is rare even when the document improves. Use these rules instead:

*   **Settled findings**: A finding is settled when it is contested or
    recorded without a fix (accepted risk). Keep the settled list across
    rounds. If a new reviewer raises a settled finding again (same affected
    scope, same issue), do not re-open it.
*   **Fixed findings**: An applied fix does not settle a finding. If a later
    reviewer raises the same issue at the same scope, the fix failed or a
    later edit reintroduced the issue. Re-open the finding.
*   **Clean round**: A round is clean when no reviewer raises a Blocker or
    Major finding that is not settled. Nits do not block. A reviewer recorded
    as not run makes the round not clean. Decide this from the review results,
    before you apply the fixes of that round.
*   **Converged**: Two consecutive clean rounds. One clean round is not enough,
    because the next fresh reviewer can find a different gap in the same text.
*   **Cap**: 5 rounds.
*   **Churn**: A new finding is at the scope of an applied fix, and reverting
    that fix would resolve it.

On converged, go to Step 3. On cap or churn, write a short list of the open
findings by severity, then ask the user one question: accept and proceed,
continue past the cap, or return the document to drafting.

**Crucial Turn-Taking Requirement**: After the resolution phase finishes for a
given round, the loop driver **MUST respect the turn-taking checkpoint in
the resolve workflow**. The driver must remain idle across human turns,
only looping back to Step 2.1 when the user explicitly confirms they wish to run
another round of review.

### Step 3: Output Boundary & Final Checkpoint

Upon completion (converged, cap or churn accepted by the user, or manual
halt):

*   **Programmatic Callers**: If running as a subflow of the review workflow, exit
    cleanly to hand execution back to the parent workflow.
*   **Manual Invocations**: Output a summary of completion in chat: *"The review
    loop has completed after [N] rounds ([converged | accepted at cap | halted]).
Open findings are in the report. Would you like to proceed to
    the next stage, or do you have any final questions?"*
