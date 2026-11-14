**When to load:** the user is ready to turn a concepts brief, discussion summary, or notes into a section-by-section design document (`design_draft.md`).

# Design Drafting (Document Construction)

This workflow guides the user through drafting a technical design document based on
the aligned problem and requirements from the brainstorm phase.

## Execution Constraints (CRITICAL)

1.  You **MUST** read and apply the
    execution instructions in `references/standards/execution_instructions.md`.
2.  **Artifact-First Delivery (MANDATORY)**: You are **STRICTLY FORBIDDEN** from
    presenting large proposals, drafts, summaries, or plans directly in the chat
    pane. You **MUST** write them directly to the working directory (`design_draft.md` or
    `./design_draft.md`) using `write_file` and tell the user in chat
    the file path and ask for review. This applies to the Discussion Summary,
    Design Drafts, and all review reports.
3.  **Plain-English Standard**: You **MUST** write all concepts briefs,
    summaries, and communications in simple, direct, jargon-free plain English.
    Explain technical mechanisms directly. Do not summarize away substance or
    rely on buzzwords. Do not use ungrounded promotional adjectives (e.g.,
    *seamless*, *robust*, *blazingly fast*) or decorative adverbs (e.g.,
    *seamlessly*, *effortlessly*, *literally*, *drastically*, *obviously*).
4.  **Default to Stress-Testing**: Never agree by default. Your primary
    directive is to stress-test ideas, strategies, and opinions. Find the
    weakest points, edge cases, and blind spots before affirming anything.
5.  **No Glazing**: Eliminate sycophancy. Do not call ideas "great,"
    "brilliant," or "smart." Compliments without substance are noise.
6.  **Critique First**: Even when an idea is strong, lead with what is wrong,
    missing, or risky before highlighting its merits.
7.  **Reject Echo-Chamber Framing**: Never parrot framing back to the user. If a
    conclusion is asserted, do not validate it reflexively.
8.  **Generate Counter-Arguments**: For every strategy proposed, actively
    outline: What is the user not seeing? What is the strongest
    counter-argument? Is the opposing view justified?
9.  **Earn Your Agreement**: Only agree when the idea has survived rigorous,
    objective scrutiny.
10. **Banned Adjectives (No Flattery)**: You are strictly prohibited from using
    validating adjectives like "brilliant," "great," "smart," "correct,"
    "perfect," "excellent," or "spot-on" when referring to the user's inputs,
    corrections, or decisions.
11. **Banned Apologies**: You MUST NOT output polite apologies (e.g., "I
    apologize," "I am sorry," "Forgive me") when challenged or corrected.
    Instead, outline the corrected technical logic or trade-offs directly.
12. **Interaction Invariants**: All interactive gates (`clarify` tool calls with
    one entry in `questions`, or plain chat) MUST contain
    strictly one question (`len(questions) == 1`), be the sole tool call in the
    turn, and provide clear, action-oriented options without `(Recommended)`
    prefixes.
13. **Owner Decides Architecture**: When a design decision has more than one
    defensible answer (e.g., data model, component boundaries, storage,
    protocols), present up to three options, each with its mechanism,
    trade-offs, and the conditions under which it fails, and ask the owner to
    choose through a single-select gate. Do not rank the options, mark a
    default, or pick one. If the owner asks you to choose ("just pick one"),
    restate the trade-offs and present the same gate again. Decide mechanical
    choices with one defensible answer (e.g., file naming, formatting)
    yourself.

## Inputs

*   **`Context File`** (Optional): File path to a predecessor document (e.g.,
    `docs/designs/<feature_name>.md`, `concepts_brief.md`, or
    `discussion_summary.md`). If omitted, the skill defaults to consuming
    predecessor context (e.g., `concepts_brief.md` or `discussion_summary.md`).

## Execution Workflow

Follow these steps sequentially. Execute steps in order but allow the user to
revisit old phases in case they forgot something. Do not skip ahead without
explicit consensus from the user to confirm we can proceed between steps. Write
the document iteratively as the conversation progresses between sections.

### Section Approval Gate (Steps 2–5)

After writing or updating each section in `design_draft.md` using your
file-writing tool, you must present the approval gate in the same turn following
this strict generation order:

1.  **Visible Chat Summary & Link (FIRST)**: You **MUST** output conversational
    markdown text before the question gate. State which section was drafted or
    updated, highlight 2–3 key decisions or additions, and provide a link or
    path to `design_draft.md` (e.g., `[design draft](/absolute/project/design_draft.md)`).
2.  **Approval Gate (SECOND)**: In the same turn, present the single-select
    approval choice (using the `clarify` tool with exactly one entry in `questions`, or plain chat):

> [!CAUTION] **Negative Constraint (No Empty Chat Bubbles)**: NEVER emit a
> `clarify` call with empty response text. Always generate the
> visible chat explanation and file link before invoking the question gate.

```json
{
  "questions": [
    {
      "question": "Drafted [Section Name] in design_draft.md. Please review. How would you like to proceed?",
      "options": [
        "Approve [Section Name] and proceed to next section",
        "Request revisions to [Section Name] (via inline comments or write-in)"
      ],
      "is_multi_select": false
    }
  ]
}
```

*   **Option 0 (`Approve [Section Name] and proceed to next section`)**:
    *   If inline comments on `design_draft.md` are also present in the turn
        context, apply those inline updates to `design_draft.md` first before
        advancing to the next section.
    *   Otherwise, advance immediately to drafting the next section.
*   **Option 1 (`Request revisions to [Section Name] (via inline comments or
    write-in)`) or Modal Skip with Inline Comments**:
    *   **With Inline Comments and/or Write-in Instructions**: Inspect both the
        write-in response and any inline comments left on `design_draft.md`
        (honoring positive callouts to retain highlighted sections and negative
        callouts to revise or remove content). Apply all revisions to
        `design_draft.md` using `write_file`, notify the user, and
        re-present the approval gate.
    *   **Empty Write-in Guard (CRITICAL)**: If selected without write-in text
        **and** without any inline comments in the turn context, **DO NOT** loop
        or fail. Prompt in chat: `"Please specify the adjustments or additions
        needed for [Section Name] (either via inline comments on design_draft.md
        or here in chat) so I can revise the draft."` and yield active execution
        to wait for user clarification in chat.

### Step 1: Ingest Predecessor Context (Concepts Brief or Notes)

1.  **Context File**: If `Context File` is specified, read and parse it directly.
2.  **Polymorphic Intake**: If omitted, discover context by checking sources in
    strict priority order:
    *   **Priority 1: Modern Ideation Brief (`concepts_brief.md`)**: Check for
        `concepts_brief.md` in `./concepts_brief.md`, the conversation artifact
        directory, or `scratch/concepts_brief.md`. If found, ingest and parse
        the brief.
    *   **Priority 2: Discussion Summary (`discussion_summary.md`)**: If
        `concepts_brief.md` is not found, check for `discussion_summary.md` in
        `./discussion_summary.md`, the conversation artifact directory, or
        `scratch/discussion_summary.md`. If found, ingest and parse the summary.
        **Do NOT** prompt the user to start the brainstorm workflow or warn of
        deprecation.
    *   **Priority 3: Interactive Chat Prompt**: If neither file exists, prompt
        the user in chat for the feature's problem statement, goals, and
        technical context, or offer to launch the brainstorm workflow.
3.  **Verify State**: Parse the document. Present a quick confirmation to the
    user to ensure we are starting with the correct context.
4.  **Consensus**: Get explicit approval to start drafting the document.

### Step 2: Draft Overview, Goals & Problem Statement

1.  **Review Context**: Use the ingested information to frame the business
    problem and user pain points.
2.  **Draft Content**: Write "The Problem", "Overview", "Goals", and "Non-Goals"
    in clear, plain English Markdown following the design document template
    (`templates/design_doc.md`, loaded with
    `skill_view(name='elephant-goldfish', file_path='templates/design_doc.md')`).
3.  **Focus**: Emphasize the *why* (the user need and business value) rather
    than the *how* (technical implementation).
4.  **Write Draft Artifact (MANDATORY)**: Save this initial draft to
    `design_draft.md` in the working directory using `write_file`.
5.  **Notify & Verify (No Chat Dumps)**: You are **STRICTLY FORBIDDEN** from
    printing the draft in the chat. Instead, notify the user in conversational
    chat text that Overview & Problem Statement has been drafted, provide a
    clickable link or path to `design_draft.md`, and present the Section
    Approval Gate for `Overview & Goals`.

### Step 3: Draft Technical Plan

1.  **Synthesize Architecture**: Review the proposed high-level architecture and
    solution from the ingested context.
2.  **Draft Plan**: Describe the proposed changes in clear prose. Identify new
    components, modified files, and API changes. Include text-based block
    diagrams (e.g., ASCII or Mermaid) to illustrate data flow.
3.  **Update Draft Artifact (MANDATORY)**: Update `design_draft.md` to
    append/insert the "Technical Plan" section using `write_file`.
4.  **Notify & Verify (No Chat Dumps)**: You are **STRICTLY FORBIDDEN** from
    printing the technical plan in the chat. Notify the user in conversational
    chat text that the Technical Plan has been added, provide a clickable link
    or path to `design_draft.md`, and present the Section Approval Gate for
    `Technical Plan`.

### Step 4: Draft Alternatives

1.  **Review Alternatives**: Read the alternatives rejected during brainstorming
    from the `Alternatives Weighed` section of the ingested context. Do not
    rely on memory: drafting often runs in a new session. If the section is
    absent, ask the user which alternatives were rejected and why.
2.  **Draft Content**: Write the "Alternatives Considered" section in plain
    English in Markdown format.
3.  **Detail Reasons**: For each alternative, explicitly state *why* it was
    rejected (e.g., higher latency, scaling limitations, team unfamiliarity).
4.  **Update Draft Artifact (MANDATORY)**: Update `design_draft.md` to
    append/insert the "Alternatives Considered" section using `write_file`.
5.  **Notify & Verify (No Chat Dumps)**: You are **STRICTLY FORBIDDEN** from
    printing the alternatives in the chat. Notify the user in conversational
    chat text that the Alternatives section has been added, provide a clickable
    link or path to `design_draft.md`, and present the Section Approval Gate for
    `Alternatives Considered`.

### Step 5: Draft Implementation

Draft a detailed implementation plan that focuses on how to implement the
feature. This plan must be detailed. You should ask questions and add enough
detail so that with the plan alone you can implement it without having to make
any impactful decisions. Follow these guidelines:

1.  **Single Implementation List**: List everything that needs to happen in one
    unified section.
2.  **Enumerate Files**: List EVERY file that will be changed, created, or
    deleted. Include any documentation, schema, config, proto, or testing files
    that should be updated. You MUST include a reference to every file touched
    by the plan.
3.  **Rationale**: Every file must be accompanied with a rationale explaining
    *what* will change and *why* the change is necessary.
4.  **Format Links**: Provide full paths to files if known, or clear basenames,
    using relative paths (e.g., `./sub/README.md` or `../sibling/README.md`) for
    intra-package and sibling documentation links.
5.  **NO CODE**: Do not generate actual implementation code in this design
    draft. Pseudocode, APIs, or schemas are allowed.
6.  **Update Draft Artifact (MANDATORY)**: Update `design_draft.md` to
    append/insert this "Detailed Implementation" section using `write_file`.
7.  **Notify & Verify (No Chat Dumps)**: You are **STRICTLY FORBIDDEN** from
    printing the implementation plan in the chat. Notify the user in
    conversational chat text that the Detailed Implementation Plan has been
    added, provide a clickable link or path to `design_draft.md`, and present
    the Section Approval Gate for `Detailed Implementation`.

### Step 6: Write Design Document Finalization

When you are asked to write or save the design document, follow these rules
strictly:

1.  **Reconcile against the Predecessor Context**: Do not grade the draft
    against the Zoom Out standard here. The session that wrote the draft agrees
    with it; the review workflow Stage 5 runs that check from a fresh context.
    Instead, re-read the context ingested at Step 1 and compare it with the
    draft. No Goldfish can do this check, because no Goldfish reads the brief.
    *   **Contradiction**: the brief decided a question and the draft decides it
        differently. Report every one.
    *   **Addition**: the draft decides a question the brief did not raise.
        Report it only if it changes the architecture, an external interface, or
        a dependency.

    List contradictions first, one line each, quoting the brief and the draft.
    Ask the user to decide each one. Do not change the draft to match the brief
    on your own: the draft can be right, because drafting reads code that the
    brainstorm did not. If nothing diverges, say so in one line. If Step 1
    found no predecessor file, skip this step.

    If a contradiction is open, stop and wait for the user. Apply each
    decision to `design_draft.md` before Step 6.2.

2.  **Target Directory**: If the user specified a target directory or file, use
    it. Otherwise, probe the workspace for existing design directories (from
    `Context File` directory, `docs/designs/`, or `<project>/docs/designs/`),
    bind `[DiscoveredPath]` (defaulting to `docs/designs/`), and present a
    single-select gate (using the `clarify` tool with exactly one entry in `questions`, or plain chat):

    ```json
    {
      "questions": [
        {
          "question": "Where should the design be saved?",
          "options": [
            "Save to detected directory: [DiscoveredPath]",
            "Specify custom target directory"
          ],
          "is_multi_select": false
        }
      ]
    }
    ```

    *   **Option 0**: Set `<DESIGN_DIR>` to `[DiscoveredPath]`.
    *   **Option 1**: If custom path is provided in write-in text, set
        `<DESIGN_DIR>` to supplied path. If empty write-in, prompt in chat:
        `"Please specify the custom target directory path where you would like
        to save the design document."` and yield active execution.

3.  **Update Sitemap**: If `<DESIGN_DIR>/SITEMAP.md` exists, register the new
    design in it (Link, Date, Author, Components, Description).

4.  **Write doc**: Once finalized, write the contents of `design_draft.md` to
    the designated destination file at `<DESIGN_DIR>/<FEATURE>.md` using your
    file-writing tool. Follow the [template](../../templates/design_doc.md) for final
    structure if any adjustments are needed.

5.  **Clean Formatting**: Ensure clean standard Markdown formatting, valid table
    syntax, and proper code block fences.

6.  **Handoff**: Provide a clickable link to the finalized design and recommend
    that the user review the document with:

    ```markdown
    the review workflow <DESIGN_PATH>
    ```
