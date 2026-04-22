# Minos coding control plane architecture contract

Status: adopted for Task 1
Audience: operators and maintainers of the Minos coding workflow
Scope: control-plane ownership, role boundaries, publish authority, and GitHub remote policy

Purpose
- Define who decides, who builds, who reviews, and who approves publication.
- Prevent role collapse, uncontrolled autonomy, and unsafe repository publication.
- Keep GitHub usage private by default for this system.

Roles

1. Minos
- Owns the control plane.
- Selects the builder for each work item or execution track.
- Assigns the execution budget: scope, timebox, token budget, iteration cap, or other explicit stopping constraint.
- Defines acceptance criteria, workspace boundaries, and review checkpoints.
- Reviews returned artifacts, decides whether rework is required, and determines whether work is complete.
- Is the only role that can authorize publish, upstream sync, or equivalent release movement for this system.
- Enforces the private-GitHub-only policy for remotes, repositories, and publication paths unless the human explicitly changes policy.

2. builder
- Executes the assigned implementation work inside the approved workspace and budget.
- Produces code, tests, docs, diffs, and concise execution notes.
- Does not redefine scope, extend the budget unilaterally, or choose a different publication target.
- Does not push upstream directly.
- Stops at the assigned checkpoint and returns artifacts to Minos for review.

3. gate
- Performs independent validation on the builder output.
- Checks the requested quality bar: tests, lint, policy compliance, acceptance criteria, and repository safety constraints.
- Reports pass/fail findings and required fixes to Minos.
- Has no publish authority and does not replace Minos artifact review.

4. human
- Sets business intent, risk tolerance, and any policy overrides.
- Approves exceptions that change system policy, especially any deviation from private GitHub defaults.
- Resolves priority conflicts or release decisions that Minos escalates.

Control-plane contract
- Every coding task has one named Minos owner, zero or one active builder, zero or one gate pass per review cycle, and one human sponsor.
- Minos must set the builder and the budget before execution starts.
- Builder execution must terminate at a defined checkpoint, budget boundary, or explicit stop condition.
- Builder artifacts are not publishable until Minos reviews them.
- Gate validation informs the decision, but Minos retains final internal approval authority before publish is considered.
- Human approval is required for policy exceptions, risk exceptions, or scope moves that exceed Minos authority.

Private GitHub policy
- Private GitHub repositories are the default and expected hosting mode.
- Public GitHub repositories, public remotes, or any publication path that makes the code publicly accessible are out of scope by default.
- Minos must reject or halt any plan that creates, points to, or publishes to a public GitHub repository unless the human explicitly authorizes that exception.
- Builders and gate agents must treat unknown remote visibility as non-compliant until Minos or the human confirms it is private.
- If remote visibility cannot be verified, the safe action is to stop and escalate.

Forbidden anti-patterns
- One agent does everything: planning, implementation, review, and release in a single unchecked loop.
- Builder pushes upstream directly.
- Multiple builders mutate the same shared workspace in parallel.
- Unbounded autonomous execution without a predeclared budget or stop condition.
- Accidental creation or use of public GitHub repositories or public remotes.

Operational defaults
- Prefer one isolated workspace per builder task.
- Prefer short review loops over long autonomous runs.
- Prefer explicit re-delegation over silent scope expansion.
- Treat publication as a separate decision after artifact review, not as the default end of implementation.
