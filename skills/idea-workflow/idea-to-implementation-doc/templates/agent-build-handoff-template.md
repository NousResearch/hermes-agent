# <Project Name> Agent Build Handoff

## Mission
State exactly what the build agent should create in one or two paragraphs.

## Product Vision
Explain why this product exists, who it serves, and what experience it should deliver.

## User Experience Goals
- Goal 1
- Goal 2
- Goal 3

## Non-Negotiable Requirements
These are mandatory. The build is not complete unless all are satisfied.

- Requirement 1
- Requirement 2
- Requirement 3

## Out of Scope
Prevent scope creep by naming what should not be built in this pass.

- Out-of-scope item 1
- Out-of-scope item 2

## Technical Architecture
Describe the intended architecture, major components, and how they interact.

## App Topology / Surfaces
State whether this is one app or multiple pieces, such as desktop app, web app, API, background worker, upload agent, mobile app, or CLI. Define what each piece owns and how they communicate.

## Data Model
List entities, fields, relationships, storage needs, and persistence assumptions.

## Database / Storage Recommendation
State the recommended database/storage choice, why it fits this build, whether the user accepted it or changed it, and any important alternatives considered. If no database is needed, say so explicitly.

## Hosting / Data Location / Deployment
State where data and services should live for this build: local-only, self-hosted, Cloudflare, AWS, another cloud/provider, or explicitly undecided. Include deployment assumptions, operational constraints, and what must not leave the local machine if applicable.

## Platform Targets
State the required app platforms for this build: browser/web-only, Windows desktop, Mac desktop, cross-platform desktop, mobile web, iOS, Android, or some combination. Separate MVP targets from future targets and explicitly say if desktop or mobile apps are out of scope.

## Technical Stack Recommendation
List recommended defaults and user overrides for frontend/UI, backend/runtime, auth, database, file/object storage, queues/jobs, realtime/sync, search, observability/logging, testing, and deployment/CI. The build agent should not have to guess these choices unless this section explicitly marks them flexible.

## Auth / Secrets / Sharing Model
State how admin login, public/private access, shareable links, API keys, environment variables, and client/server secret boundaries should work. Never include real secrets. Say which credentials must remain server-side/local-only and what must not be exposed to public clients.

## Integrations
List external tools, APIs, CLIs, SDKs, local services, agents, or platforms involved.

## Implementation Phases

### Phase 1: <Name>
Goal:
Tasks:
Verification:

### Phase 2: <Name>
Goal:
Tasks:
Verification:

## Build Tasks
Each task should be executable by an AI coding agent.

### Task 1: <Small objective>
Objective:
Files/areas likely involved:
Steps:
Verification:
Expected result:

### Task 2: <Small objective>
Objective:
Files/areas likely involved:
Steps:
Verification:
Expected result:

## Testing Requirements
Describe required tests by level.

- Unit tests:
- Integration tests:
- End-to-end tests:
- Manual checks:
- Regression checks:

## Verification Commands / Checks
Use exact commands when known. If unknown, specify discovery steps.

```bash
# example
npm test
npm run build
```

## Acceptance Criteria
The implementation is acceptable when:

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

## Done Means
The agent may only say “done” when:

- [ ] The implementation satisfies all non-negotiable requirements.
- [ ] Required tests pass with fresh output.
- [ ] Build/lint/typecheck checks pass where applicable.
- [ ] Manual verification steps have been completed or explicitly marked unavailable.
- [ ] Known limitations are documented.
- [ ] Any generated artifacts or files are present at the expected paths.

## Known Risks
- Risk 1 — mitigation
- Risk 2 — mitigation

## Open Questions
List unresolved questions. If any question blocks implementation, the spec review should not be PASS.

- Question 1
- Question 2

## Prompt for Build Agent
Use this prompt to start implementation. It intentionally aligns with Superpowers rather than replacing it:

```text
You are implementing <Project Name> from this build handoff.

First, read the entire document and inspect the target repository. Do not start coding immediately. Create or follow an implementation plan that names files, tests, commands, expected results, and risks. Preserve the scope and non-goals in this handoff. Build incrementally, verify each task, request review, and do not claim completion until the Verification Commands / Checks, Acceptance Criteria, and Done Means checklist pass with fresh evidence.

If Superpowers skills are available, use the Superpowers Build Handoff sequence below instead of inventing your own process.
```

## Superpowers Build Handoff
Use the `superpowers-gpt` workflow on this document.

Start with:
1. `superpowers-using-superpowers` to route the work.
2. `superpowers-writing-plans` to inspect the target repo and turn this handoff into exact implementation tasks, files, commands, tests, and expected results.
3. `superpowers-executing-plans` or `superpowers-subagent-driven-development` to implement the approved plan without scope drift.
4. `superpowers-requesting-code-review` to review the implementation.
5. `superpowers-verification-before-completion` to prove the work satisfies the acceptance criteria before claiming completion.

The goal is to convert this spec into an executable implementation plan, build it with tests, review the result, and verify before claiming completion.
