# StudyOS Learning

StudyOS supports a learner pursuing observable capabilities across exams, engineering, research, and other domains. It separates source knowledge, performed activities, immutable evidence, and derived judgments so progress remains explainable.

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

**Source Anchor**:
A version-aware reference to the exact source location supporting an Activity or claim.
_Avoid_: Unqualified URL, free-form citation

**Domain Pack**:
A domain-specific vocabulary, activity policy, and rubric that uses the shared StudyOS evidence and Session model.
_Avoid_: Separate learning backend, prompt-only persona

**Activity Adapter**:
A Domain Pack implementation that proposes Activities and validates domain-specific Evidence Events at the StudyOS seam.
_Avoid_: Tool wrapper, domain database

