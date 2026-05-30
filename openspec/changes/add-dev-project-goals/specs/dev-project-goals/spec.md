## ADDED Requirements

### Requirement: Goal hierarchy persistence
The system SHALL persist project goals as a self-referential hierarchy in a
`dev_project_goals` table in the existing `hermes_state` SQLite database, where
each node has a `kind` of `vision`, `goal`, `milestone`, or `subgoal`, a
`project_id` resolved via the existing project-scope helper, and a
`parent_goal_id` that is non-null for every kind except `vision`.

#### Scenario: Create a vision node
- **WHEN** a goal is created with `kind = "vision"` and no `parent_goal_id`
- **THEN** the node is persisted with a generated `goal_id`, the resolved
  `project_id`, `status = "proposed"`, and `progress = 0.0`

#### Scenario: Reject a non-vision node without a parent
- **WHEN** a goal is created with `kind` in {`goal`, `milestone`, `subgoal`} and
  a null `parent_goal_id`
- **THEN** the create is rejected with a validation error and nothing is
  persisted

#### Scenario: Persisted nodes survive process restart
- **WHEN** the store is reopened against the same database after a restart
- **THEN** previously created nodes are readable with their status, criteria,
  and rolled-up progress intact

### Requirement: Verifiable acceptance criteria
Each goal node SHALL store acceptance criteria using the existing
acceptance-criteria v2 schema, distinguishing machine-checkable criteria
(`verification_method` of `test` or `command`) from advisory `manual` criteria.

#### Scenario: Store machine-checkable criteria
- **WHEN** a subgoal is created with a criterion whose `verification_method` is
  `command` and `machine_checkable` is true
- **THEN** the criterion is normalized and stored in the v2 shape and is
  eligible for deterministic evaluation

#### Scenario: Manual criteria are advisory
- **WHEN** a node has a criterion with `verification_method = "manual"`
- **THEN** that criterion is never used to auto-transition the node to
  `achieved` and is surfaced for human confirmation instead

### Requirement: Bottom-up status and progress rollup
The system SHALL recompute a parent node's `progress` as the weighted mean of
its children's progress and SHALL recompute parent status from children, then
propagate the recomputation upward to the root. `progress` SHALL be computed by
the system and never authored directly.

#### Scenario: Milestone achieved when all subgoals achieved
- **WHEN** every child subgoal of a milestone has `status = "achieved"`
- **THEN** the milestone transitions to `status = "achieved"` with `progress =
  1.0` and the recomputation propagates to its parent goal

#### Scenario: Blocked child blocks the parent
- **WHEN** any child of a node has `status = "blocked"`
- **THEN** the parent node is set to `status = "blocked"`

#### Scenario: Partial progress is averaged
- **WHEN** a milestone has two equally weighted subgoals at progress 1.0 and 0.0
- **THEN** the milestone's computed `progress` is 0.5 and its status is not
  `achieved`

### Requirement: Evidence-based re-evaluation of leaf subgoals
The system SHALL evaluate only leaf subgoals (never parent nodes) by assembling
an evidence digest from existing verification, CI, and execution stores,
checking machine-checkable criteria deterministically before invoking the
project goal judge, and the judge SHALL be fail-open so that a judge failure
never wedges the loop or auto-completes a goal.

#### Scenario: Deterministic gate precedes the judge
- **WHEN** a subgoal's machine-checkable criteria are not all satisfied by the
  evidence digest
- **THEN** the project goal judge is not invoked and the subgoal remains in its
  current status

#### Scenario: Judge confirms completion after gate passes
- **WHEN** all machine-checkable criteria pass and the judge returns a `done`
  verdict for the soft criteria
- **THEN** the subgoal transitions to `achieved`, the verdict and reason are
  appended to the node's audit trail, and rollup runs on its parent

#### Scenario: Judge failure is fail-open
- **WHEN** the project goal judge call errors or returns unparseable output
- **THEN** the subgoal status is left unchanged and the re-evaluation loop
  continues without raising

#### Scenario: Re-evaluation is idempotent
- **WHEN** the re-evaluation tick runs again over an already-`achieved` subgoal
- **THEN** no duplicate transition or audit entry is produced

### Requirement: Goals are never hard-deleted
The system SHALL support an `abandoned` status for goal nodes and SHALL NOT
provide a hard-delete path; abandoning a node SHALL trigger rollup recomputation
on its parent.

#### Scenario: Abandon instead of delete
- **WHEN** a caller requests removal of a goal node
- **THEN** the node is set to `status = "abandoned"` with an `abandoned_at`
  timestamp and remains retrievable, and its parent's rollup is recomputed

### Requirement: Goal control API and re-evaluation gate
The system SHALL expose Dev control routes to create goals, list/filter goals,
read the assembled tree with rolled-up progress, and trigger manual
re-evaluation, and the automated re-evaluation loop SHALL be gated behind a
configuration flag that defaults to disabled.

#### Scenario: Read the goal tree
- **WHEN** a client requests the goal tree for a `project_id`
- **THEN** the response contains the hierarchy with each node's computed
  `progress` and `status`

#### Scenario: Automated loop is off by default
- **WHEN** the re-evaluation configuration flag is unset
- **THEN** the `goals_tick` step does not run automatically, while the manual
  reevaluate route remains available
