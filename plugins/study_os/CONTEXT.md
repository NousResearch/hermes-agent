# StudyOS Learning

StudyOS supports a learner pursuing observable capabilities across exams, engineering, research, and other domains. It separates source knowledge, performed activities, immutable evidence, and derived judgments so progress remains explainable.

## Application Boundary

`StudyOSApplication` is the transport-independent boundary for HTTP and
non-model clients. Callers submit a `StudyQuery` or `StudyCommand`; the
application owns workspace resolution, validation, persistence ordering, and
stable error semantics. The FastAPI adapter lives in
`dashboard/plugin_api.py`, mounted canonically at
`/api/plugins/study_os`. The bundled `/api/study` prefix is a compatibility
alias only.

The model-facing surface remains the two established tools,
`study_activity` and `study_coach`. Their schemas are deliberately not folded
into the HTTP query/command catalog, preserving the narrow model-tool waist.

## Language

**Learner**:
The human whose demonstrated capabilities StudyOS models.
_Avoid_: User profile, student record

**Learning Project**:
A bounded learning purpose that owns objectives, sessions, evidence, and planning constraints.
_Avoid_: Course, vault, exam plan

**Objective**:
An observable capability the learner intends to demonstrate under stated success criteria.
_Avoid_: Topic, chapter, vague goal

**Learning Contract**:
The explicit agreement for one Session: its mode, objective, time budget, assistance level, and required evidence dimensions.
_Avoid_: Prompt settings, study preferences

**Session**:
An ordered, resumable sequence of Activities governed by one Learning Contract.
_Avoid_: Chat, daily log

**Activity**:
A bounded task offered to the Learner to produce a response or Artifact that can be evaluated.
_Avoid_: Tool call, lesson content, schedule event

**Artifact**:
A learner-produced output such as an answer, derivation, code change, benchmark, critique, or experiment result.
_Avoid_: Generated explanation, source material

**Evidence Event**:
An immutable observation of learner performance, including evaluator provenance, assistance used, and links to its Activity and Artifacts. An Attempt is the interactive-response form of an Evidence Event.
_Avoid_: Mastery update, progress score

**Competency Snapshot**:
A derived, revisable projection of demonstrated capability across evidence dimensions at a point in time.
_Avoid_: Mastery state, permanent level

**Evidence Dimension**:
The kind of capability an Evidence Event supports: recall, recognition, execution, explanation, near transfer, or far transfer.
_Avoid_: Difficulty, review level

**Intervention**:
An evidence-backed recommendation for what should happen next and why.
_Avoid_: Generic advice, automatic schedule mutation

**Intervention Queue**:
A time-sensitive, derived ordering of the most valuable current Interventions across a Learning Project. It is a read model, not durable learning truth or a Schedule.
_Avoid_: To-do list, fixed curriculum, mastery queue

**Plan Proposal**:
A durable candidate that preserves selected Interventions and their evidence provenance for Learner review. Acceptance records a decision; applying it to a Schedule remains a separate explicit act.
_Avoid_: Automatic plan, scheduled task, accepted Schedule

**Source Anchor**:
A version-aware reference to the exact source location supporting an Activity or claim.
_Avoid_: Unqualified URL, free-form citation

**Domain Pack**:
A domain-specific vocabulary, activity policy, and rubric that uses the shared StudyOS evidence and Session model.
_Avoid_: Separate learning backend, prompt-only persona

**Activity Adapter**:
A Domain Pack implementation that proposes Activities and validates domain-specific Evidence Events at the StudyOS seam.
_Avoid_: Tool wrapper, domain database
