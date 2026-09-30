# Unified knowledge review

The source is the local `ai-employee` implementation:
`hosted/background_review.py::HOSTED_UNIFIED_REVIEW_PROMPT`, its memory-turn
trigger, and `agent/background_review.py::_review_tool_whitelist`.

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

The review prompt is the source unified prompt, with local path substitutions
and the established local-connection adjustment: verified access can establish
a manual because there is no hosted connection gateway creating folders.
Everything else is copied. See `agent/employee_review.py`.

The memory tool's default description equals the source product's
`HOSTED_MEMORY_TOOL_DESCRIPTION`. It describes operations and personal/shared
scopes, not a second knowledge policy. Single-store configuration narrows both
the description and target default. Native operation implementations remain.
