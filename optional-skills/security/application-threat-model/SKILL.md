---
name: application-threat-model
description: "Use when threat-modeling an app: assets, STRIDE, controls."
version: 0.1.0
author: rlaope (github.com/rlaope), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [threat-model, stride, appsec, architecture-review, defensive-security]
    category: security
    related_skills: [web-pentest, security-event-response]
---

# Application Threat Model Skill

Turn an application the user operates into a defensive threat model: assets worth taking, trust
boundaries, STRIDE scenarios per boundary, one owned decision per scenario, and the security test
that fails when each control is removed. It is analysis over architecture the user supplies — not
a scan, a penetration test or a compliance attestation — and it never produces exploit code.

Ported from [rlaope/oh-my-hermes](https://github.com/rlaope/oh-my-hermes) (MIT), `omh-application-threat-model`.

## When to Use

- "Build a threat model for our payment service / API / SaaS architecture."
- A design review that must say where trust changes hands and what stops an attacker there.
- Before a launch, a new third-party integration, or multi-tenant storage.

Don't use for: actively testing a live target (`web-pentest`, with written authorization), an event
that already happened — a CVE, leaked secret, advisory (`security-event-response`), or reviewing a
single diff for bugs.

## Prerequisites

None beyond the standard toolset. Read the system from what the user provides or the repo itself:
`read_file` / `search_files` over architecture docs, IaC, `docker-compose`, route tables and auth
middleware; `vision_analyze` for a diagram image.

## Procedure

1. **System map.** Name every component, data store, queue, external dependency and where each runs,
   plus which calls which. Ask for anything missing — never substitute a generic checklist for the
   real system. Done when every data flow has a source, a sink and a transport.
2. **Scope.** Record threat actors in scope (external, authenticated tenant, insider, compromised
   dependency) and explicit exclusions. Done when both lists are written.
3. **Assets.** Register what an attacker wants, each with a data class and exactly one loss:
   disclosure, corruption, unavailability or fraud (`references/threat-model-method.md` §1). A row
   with no loss is context, not an asset.
4. **Trust boundaries.** For each: what crosses it, what authenticates the crossing, what the
   receiver assumes without checking, what happens when that assumption is false (§2). Done when
   every flow from step 1 is inside one boundary or crosses a listed one.
5. **Scenarios.** Run all six STRIDE prompts against every boundary (§3). Each scenario has entry
   point, path, precondition, impact. Drop an unreachable one with the sentence that says why.
6. **Decisions.** One per scenario — mitigate, transfer, accept, eliminate — with an owner and the
   evidence the decision requires (§4). Two decisions on one scenario means it is undecided.
7. **Controls and tests.** Mark each control `deployed`, `planned` or `unverified` (default). Only
   observed configuration or a passing test makes it `deployed`. Give every mitigating control a
   test whose observable fails when the control is removed (§6).
8. **Residual risk.** List accepted scenarios, unverified controls and missing tests, and say which
   of scan / pentest / attestation the user still needs.

Write the model to a markdown file with `write_file` (sections: scope, assets, boundaries,
scenarios, decisions, controls & tests, residual risk) so a rerun can diff it.

## Pitfalls

- **Wrong subject.** The agent's own prompts, tools and credentials are not this application;
  model them only if the user's product *is* an agent, and then as a component with its own boundary.
- **Diagram ≠ control.** "We use mTLS" on a slide is `unverified` until config or a test shows it.
- **Checklist drift.** A scenario with no boundary and no asset is noise; drop it with the reason.
- **No weapons.** Scenarios name entry point, path and precondition plus a detection signal — never a
  payload, exploit code or runnable attack script.
- Never paste secrets, connection strings or live customer records into the model as examples.

## Verification

- Every asset has one data class and one loss; every boundary answers its four questions.
- Every boundary shows six STRIDE outcomes (scenario or recorded reason).
- Every scenario has exactly one decision and an owner; every mitigate has a control and a test.
- No control reads `deployed` without cited config or test output.
