# Slack Workflow

Use this reference when productizing `cyber-vc-analyst` into Hermes Slack
workflows.

## Invocation Rules

- In Slack threads, prefer `!cyber-vc-analyst ...`.
- In top-level Slack slash-command entrypoints, `/hermes cyber-vc-analyst ...`
  is acceptable when the workspace is already routing Hermes commands.
- Avoid assuming native Slack slash commands are available inside threads.
  Slack blocks them there.

## Recommended Slack Entry Shapes

These can be invoked either through the compatibility entrypoint or through the
new narrower skill names.

### Company analysis

```text
!cyber-vc-analyst company Red Access Security
!cyber-vc-company Red Access Security
```

### Thematic analysis

```text
!cyber-vc-analyst theme SOC Automation / AI SOC
!cyber-vc-theme SOC Automation / AI SOC
```

### Comparison

```text
!cyber-vc-analyst compare Red Access Security vs Noma Security
!cyber-vc-compare Red Access Security vs Noma Security
```

### Triage

```text
!cyber-vc-analyst triage <company>
!cyber-vc-triage <company>
```

### Competitor landscape

```text
!cyber-vc-analyst competitors browser security
!cyber-vc-competitors browser security
```

## Response Contract

For Slack, default to a compact response contract:

1. Acknowledge the requested mode and subject.
2. If the run is likely to take a while, post a short progress note.
3. Return a concise result summary in-thread.
4. If a vault note was written, include the save path.
5. Keep the long-form memo as the durable vault artifact unless the user asks
   for the full memo in Slack.

## Mode-Specific Slack Output

### Company mode

Return:

- one-line investment view
- 3 to 5 highest-signal points
- recommendation
- confidence caveat
- vault path if saved
- follow-up question asking whether to save the full company memo if it was not already requested

### Theme mode

Return:

- one-line thesis view
- why-now summary
- representative companies or market structure
- key risk
- vault path if saved
- follow-up question asking whether to save the theme memo if it was not already requested

### Compare mode

Return:

- which company looks stronger now
- 3 decision-driving differences
- what evidence most weakens the current ranking
- whether a full memo is warranted for one or both companies
- follow-up question asking whether to save the comparison or expand one company into a full memo

### Triage mode

Return:

- likely category
- one-line why-now read
- top 2 or 3 risks
- recommendation on whether to escalate to a full memo
- follow-up question asking whether to save the triage note or expand it into a full memo

## Operational Guidance

- Use vault context first.
- Use Return on Security MCP second for market structure and company landscape.
- Use public web only to close important gaps.
- Separate private vault context, ROS market intelligence, and public evidence
  in the final memo.
- If ROS MCP is unavailable, continue and explicitly call out the degraded
  market-intelligence path.
- Accept an optional leading depth token:
  - `quick`
  - `standard`
  - `deep`
- For longer compare, theme, or competitor runs, create or update a resumable
  `research-state` artifact instead of starting cold every time.

## Productization Path

The recommended productization sequence is:

1. Stabilize the skill contract.
2. Standardize Slack prompts and result shapes.
3. Add schemas and fixtures for regression checks.
4. Add phase contracts, depth tiers, and resumable state.
5. Move canonical docs and fixtures into a dedicated private repo.
