# Unified knowledge review

The prompt adapts native Hermes's combined memory/skill review at upstream
`6e69a8933adda7dbbff7cf3009a259a4524477e9` to three separate destinations:
memory, connection manuals, and responsibilities. Runtime text lives in
`agent/employee_review.py`; the native reference is
`agent/background_review.py::_COMBINED_REVIEW_PROMPT` and its shared blocks.

One review handles memory, connection manuals and responsibilities together.
The memory cadence defaults to ten foreground turns and honors native config.
Failed, interrupted, suppressed or refused starts leave the review due. Reset
its counter only after the local review thread starts. Native single-run
admission prevents concurrent reviews on the same agent; nested review nudges
remain disabled. Different conversations may review independently.

Use the native local thread/idle-queue lifecycle, not the hosted durable job
service. No separate skills/curator pass is enabled. This intentionally does
not copy hosted leases, workspace-job persistence or infrastructure retries.

Permitted tools are `memory`, `read_file`, `search_files`, `write_file`, and
`patch`, intersected with the parent's actual tools and memory flags. The
source-only `report_issue` is excluded, as agreed. Other tool schemas stay in
the inherited snapshot for cache parity but dispatch rejects those calls.
Review writes remain limited to knowledge; schedule/webhook changes are
proposals in STATE.md. Personal-memory attribution uses frozen participants.

## Prompt behavior

Previously the prompt copied the source product's unified filing instructions.
It now follows native review's lesson-writing guidance, with separate sections
and update priorities for all three destinations:

- Memory distinguishes named-person facts from shared organization/environment
  facts. Cross-cutting preferences belong here; procedures do not.
- Connection manuals own verified service mechanics. Prefer a consulted manual,
  then an existing service manual, then a topical support file. A new manual
  requires verified access.
- Responsibilities own task-specific corrections, duties, authority, working
  knowledge, and current state. Prefer existing packages; a one-off task does
  not justify a new responsibility. Read the authoring guide before creating
  or substantially reshaping one.

Native's proactive tone remains. Procedures include concrete steps and the
reason for pitfalls; reusable rules must not duplicate already-loaded guidance.
Unresolved attempts must not become validated workflows. Current blockers and
exact-item approvals may live in state without becoming standing rules.

Each preference has one owner. Review rereads targets and searches before
creating files, consolidates superseded knowledge, and reports successful writes
only. It no longer directs migration merely because an existing filename is
dated; new reusable files are named by topic. Native skill ownership/curator
instructions are replaced by the existing knowledge-write and authority bounds.

This changes the review user message on every surface using the native review
worker, including explicit `/refine` (whose focus still takes precedence).
The trigger, tools, personal attribution, history replay, and inherited warm
system prompt are unchanged. No skills pass is restored.

The memory tool's default description equals the source product's
`HOSTED_MEMORY_TOOL_DESCRIPTION`. It describes operations and personal/shared
scopes, not a second knowledge policy. Single-store configuration narrows both
the description and target default. Native operation implementations remain.
