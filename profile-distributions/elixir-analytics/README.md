# Elixir Analytics Hermes Profile

This profile turns Hermes into a Slack-first analytics agent for Elixir.

It keeps the self-improvement loop enabled while narrowing the active runtime
to analytics work: Slack, read-only query execution, skills, memory, session
search, source maintenance, and scheduled review.

The intended Slack app for this profile is `macros`. Its Slack tokens should
live only in the installed `elixir-analytics` profile's `.env`.

## Install Locally

From the Hermes repo:

```bash
hermes profile install ./profile-distributions/elixir-analytics --name elixir-analytics
```

Then copy the generated `.env.EXAMPLE` inside the installed profile to `.env`
and fill in the required Slack tokens plus any analytics credentials needed for
the current milestone.

## Runtime Boundary

Hermes owns AI reasoning and analytics execution. The Next.js analytics app
renders temporary visualization artifacts. Source-of-truth analytics changes
should happen through GitHub PRs against the analytics repo.

## Roadmap And Rollout

Use `ROADMAP.md` as the canonical product roadmap and production rollout
checklist for this profile. It tracks local implementation milestones,
remaining external blockers, and the Slack E2E prompts that must pass before
calling the analytics agent production-ready.
