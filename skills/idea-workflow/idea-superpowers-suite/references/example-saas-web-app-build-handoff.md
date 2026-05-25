# Team Pulse SaaS Agent Build Handoff

## Mission

Build a small web app where a team can submit weekly status updates and view a simple team pulse summary page.

## Product Vision

A lightweight alternative to bloated project-status tools: one weekly prompt, one summary page, minimal administration.

## User Experience Goals

- Team members submit a short weekly update in under two minutes.
- Managers see blockers, morale, and priorities at a glance.
- The app feels calm and low-friction.

## Non-Negotiable Requirements

- Users can create/edit their own weekly update.
- Summary page groups updates by week.
- Updates include priorities, blockers, confidence/mood, and notes.
- Basic auth or single-user/team access must exist if deployed beyond localhost.

## Out of Scope

- Payroll/HR features.
- Complex performance reviews.
- Slack bot for v1.
- Advanced analytics.

## Technical Architecture

- Web frontend with form and summary views.
- Backend/API or full-stack framework depending on repo conventions.
- Persistent database for users, weeks, and updates.
- Server-rendered or simple client-rendered UI is acceptable.

## Data Model

User:
- `id`
- `name`
- `email`

WeeklyUpdate:
- `id`
- `user_id`
- `week_start`
- `priorities`
- `blockers`
- `confidence_score`
- `notes`
- `created_at`
- `updated_at`

## Integrations

None required for v1.

## Implementation Phases

1. Inspect repo and choose existing app patterns.
2. Create data model/migration.
3. Build weekly update form.
4. Build summary page grouped by week.
5. Add auth/access handling appropriate to the project.
6. Add tests and deployment docs.

## Build Tasks

- Identify framework, routing, database, and test conventions.
- Add model/schema and persistence tests.
- Add form validation tests.
- Implement submission/edit flow.
- Implement summary view.
- Add seeded sample data if useful.

## Testing Requirements

- Model validation tests.
- Form submission tests.
- Summary page rendering tests.
- Access-control checks where auth exists.

## Verification Commands / Checks

- Run the project test suite.
- Start the app locally.
- Submit an update as a user.
- Edit the update.
- Confirm the summary page shows the update under the correct week.

## Acceptance Criteria

- A user can submit one update per week.
- A user can edit their current update.
- Summary page summarizes current-week updates.
- Blockers are easy to identify.

## Done Means

- Tests pass.
- Manual local flow works end-to-end.
- The summary page is usable with sample or real data.
- Auth/access assumptions are documented.

## Known Risks

- Auth choices can expand scope.
- Summary page complexity should stay minimal for v1.

## Open Questions

- Should managers see all users while members see only their own history?
- Should week boundaries use local timezone or UTC?

## Prompt for Build Agent

Inspect the repository and current web stack first. Write an implementation plan with exact files, migrations, tests, and commands. Keep v1 simple and avoid adding integrations unless explicitly requested.

## Superpowers Build Handoff

Use the `superpowers-gpt` workflow on this document. Start with `superpowers-using-superpowers`, then `superpowers-writing-plans`, then execute, review, and verify with the appropriate Superpowers skills.
