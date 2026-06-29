# Investigation: {title}

## Hand-off Brief

1. **What happened.** {one-sentence problem statement, evidence-graded}
2. **Where the case stands.** {status, last finding, what would unblock progress}
3. **What's needed next.** {single recommended action with rationale}

## Case Info

| Field            | Value                                                                      |
| ---------------- | -------------------------------------------------------------------------- |
| Ticket           | {ticket-id or "N/A"}                                                       |
| Date opened      | {date}                                                                     |
| Status           | Active                                                                     |
| System           | {OS, version, relevant environment details}                                |
| Evidence sources | {diagnostic archive, logs, crash dump, code, version control, etc.}        |

## Problem Statement

{User-reported description; the initial claim. May be refined or contradicted by evidence.}

## Evidence Inventory

| Source   | Status                          | Notes     |
| -------- | ------------------------------- | --------- |
| {source} | {Available / Partial / Missing} | {details} |

## Investigation Backlog

| # | Path to Explore | Priority              | Status                                | Notes     |
| - | --------------- | --------------------- | ------------------------------------- | --------- |
| 1 | {description}   | {High / Medium / Low} | {Open / In Progress / Done / Blocked} | {context} |

## Timeline of Events

| Time        | Event               | Source                | Confidence            |
| ----------- | ------------------- | --------------------- | --------------------- |
| {timestamp} | {event description} | {log file, commit, …} | {Confirmed / Deduced} |

## Confirmed Findings

### Finding 1: {title}

**Evidence:** {citation — `path:line`, log timestamp, or commit hash}

**Detail:** {description}

## Deduced Conclusions

### Deduction 1: {title}

**Based on:** {which Confirmed Findings}

**Reasoning:** {logical chain}

**Conclusion:** {what follows}

## Hypothesized Paths

### Hypothesis 1: {title}

**Status:** {Open / Confirmed / Refuted}

**Theory:** {description}

**Supporting indicators:** {what makes this plausible}

**Would confirm:** {specific evidence that would prove this}

**Would refute:** {specific evidence that would disprove this}

**Resolution:** {when Status changes from Open, what evidence settled it}

## Missing Evidence

| Gap              | Impact                               | How to Obtain   |
| ---------------- | ------------------------------------ | --------------- |
| {what's missing} | {what it would confirm or eliminate} | {how to get it} |

## Source Code Trace

| Element       | Detail                                      |
| ------------- | ------------------------------------------- |
| Error origin  | {file:line, function name}                  |
| Trigger       | {what causes this code to execute}          |
| Condition     | {what state produces the observed behavior} |
| Related files | {other files in the same code path}         |

## Conclusion

**Confidence:** {High / Medium / Low}

{Summary stating what is Confirmed vs. what remains Hypothesized. If a root cause is identified, state it; otherwise
name the most promising hypothesized paths and what would resolve the remaining uncertainty.}

## Recommended Next Steps

### Fix direction

{What needs to change and why. Categorize by mechanism when multiple issues combine.}

### Diagnostic

{Steps to confirm the root cause: additional logging, targeted tests, data to collect.}

## Reproduction Plan

{Setup, trigger, expected results. Scale from isolated proof to full system reproduction.}

## Side Findings

Tangential observations surfaced during the investigation, evidence-graded, with citation when applicable.

- {observation}

## Follow-up: {date}

### New Evidence

### Additional Findings

### Updated Hypotheses

### Backlog Changes

### Updated Conclusion
