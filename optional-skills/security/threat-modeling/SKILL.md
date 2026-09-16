---
name: threat-modeling
description: Use when modeling threats for a system or feature.
version: 0.1.0
author: Thomas Yiu, Hermes Agent
license: MIT
category: security
triggers:
  - "threat model a system"
  - "perform STRIDE analysis"
  - "analyze an attack surface"
  - "identify threats in an architecture"
toolsets:
  - terminal
  - file
  - web
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, threat-modeling, architecture, risk]
    related_skills: []
---

# Threat Modeling Skill

Produce a traceable, assumption-aware threat model for a system, feature, data flow, or architecture change. Focus on defensive analysis and prioritized mitigations; do not invent facts or provide exploit instructions.

## When to Use

- A user asks for STRIDE, attack-surface, abuse-case, or threat analysis.
- A new feature, trust boundary, integration, deployment, or data flow needs security review.
- A design decision needs explicit security risks and mitigations.

Don't use for: live exploitation, credential recovery, malware development, or unauthorized testing.

## Prerequisites

- System scope and intended security objectives.
- Architecture, data-flow, deployment, and trust-boundary details when available.
- Clearly label missing information as an assumption or open question.

## Procedure

1. Define scope, assets, actors, entry points, trust boundaries, and security objectives. Completion criterion: the in-scope and out-of-scope components are explicit.
2. Build a data-flow inventory from user or repository evidence. Record component, data, protocol, authentication, authorization, storage, and boundary crossings. Completion criterion: every named flow has source, destination, and protection notes.
3. Enumerate threats using STRIDE plus abuse cases: spoofing, tampering, repudiation, information disclosure, denial of service, elevation of privilege, supply-chain risk, and privacy misuse. Completion criterion: each asset and boundary has at least one considered threat or a recorded reason it is not applicable.
4. Assess likelihood and impact using a simple Low/Medium/High scale, explaining the driver for each rating. Completion criterion: every accepted threat has a severity and rationale.
5. Recommend controls in priority order: eliminate, prevent, detect, contain, and recover. Map each control to a threat and an owner or implementation location. Completion criterion: every High and Medium threat has a concrete mitigation or explicit acceptance decision.
6. Identify residual risk, assumptions, validation questions, and security tests. Completion criterion: unresolved items are listed with an evidence-gathering action.

## Output Format

Use these headings:

- Scope and objectives
- Architecture and trust boundaries
- Assets and data flows
- Threat register: ID, component/flow, threat, STRIDE category, likelihood, impact, severity, evidence
- Mitigations: threat ID, control, priority, owner, validation
- Residual risk and assumptions
- Verification plan

## Pitfalls

- Do not treat a diagram as complete evidence; ask about identity, secrets, logging, failure modes, and administrative paths.
- Do not rate severity without stating impact and exposure assumptions.
- Do not recommend encryption as a generic fix without naming key ownership, endpoint authentication, and lifecycle controls.
- Separate design weaknesses from confirmed vulnerabilities.

## Verification

Check that every asset, trust boundary, threat, and mitigation is traceable. Confirm that high-risk items have owners and testable acceptance criteria, and that assumptions are visibly separated from verified facts.