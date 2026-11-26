---
name: adversarial-debate
description: "Synchronous 4-agent adversarial debate with DQI scoring."
version: 1.0.0
author: Ramiz Mehran (ramizmehran) <ramiz@example.com>
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [debate, multi-agent, decision-quality, consensus, cross-examination]
    related_skills: [team-of-thoughts, kanban-orchestrator]
    config:
      delegation.max_concurrent_children: 4
---

# Adversarial Debate

Orchestrate a synchronous 3-round adversarial debate among Clio, Hephaestus, Solon, and Talaria using `delegate_task`. Agents produce structured JSON claims with evidence types, rebut each other by claim ID, and converge to a weighted Decision Quality Index (DQI).

**Core principle:** Structured cross-examination beats distributed monologue. Agents must defend, concede, or rebut by evidence — not opinion.

## When to Use

Use this skill when:
- A decision has significant irreversible consequences (one-way door)
- There are ≥2 plausible approaches with strong trade-offs
- A previous kanban ToT produced weak or conflicting synthesis
- The user explicitly asks for "full team debate", "adversarial review", or "cross-examine this"

For routine feature design, low-risk bug fixes, and deliverable validation, use the kanban ToT (`team-of-thoughts` skill) instead.

## Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| `delegation.max_concurrent_children` | 4 | In `config.yaml` — needed for parallel agent spawn |
| `delegation.child_timeout_seconds` | 600 | Per-agent timeout for each round |
| Hermes profiles | clio, hephaestus, solon, talaria | Must exist and be operational |
| `delegate_task` tool | Available | Included in all platform toolsets |

## How to Run

### From agent prompt (primary path)

Load this skill, then follow the Procedure below. You (Hermes/Default) serve as debate moderator: spawn agents via `delegate_task`, collect outputs, calculate DQI, produce synthesis.

### From CLI (debugging / test)

```bash
~/.hermes/skills/adversarial-debate/scripts/debate-orchestrator.sh \
  --topic "Topic" \
  --context-file /tmp/context.md \
  --format rca
```

## Quick Reference

```
Setup → Round 1 (4 parallel agents, opening statements)
      → Round 2 (4 parallel agents, cross-examination by claim ID)
      → DQI calculation → if DQI < 0.6 → Round 3 (convergence)
      → Weighted synthesis → kanban root card
```

DQI formula per claim: `confidence × evidence_multiplier × rebuttal_status`
  - evidence: direct=1.0, inferred=0.7, speculative=0.4
  - rebuttal: unrebutted=1.0, conceded=0.7, unresolved=0.5

## Procedure

### 1. Setup

1. Inform the user the debate is starting.
2. Create workspace: `/tmp/hermes-debate-<timestamp>/` with `round-1/`, `round-2/`, `round-3/` subdirectories.
3. Write the context file including topic, format (`rca` or `feature`), project root, background, and success criteria.
4. Create a kanban root card for tracking:
   ```bash
   hermes kanban create "Debate: <topic>" --triage --body "$(cat context)"
   ```
5. Initialize the transcript:
   ```bash
   echo '{"debate_id":"debate-<ts>","topic":"<topic>","format":"<rca|feature>","rounds":[],"dqi":null}' > transcript.json
   ```

### 2. Round 1 — Opening Statements

Read the Round 1 template from `templates/round-1-opening.md`. Render placeholders with the actual topic, format, and agent name. Spawn all 4 agents in parallel via `delegate_task`:

```python
for agent in ["clio", "hephaestus", "solon", "talaria"]:
    delegate_task(
        subagent_type="general",
        description=f"Adversarial debate Round 1 — {agent}",
        prompt=rendered_round1_prompt(agent, topic, context)
    )
```

Each agent returns structured JSON with claims, evidence citations, confidence (0–100), and evidence strength (`direct` | `inferred` | `speculative`).

On timeout or malformed JSON, record `{"agent":"<name>","error":"timeout|parse_error"}` and continue.

Append all 4 outputs to the transcript.

### 3. Round 2 — Cross-Examination

Render the Round 2 template from `templates/round-2-cross-examination.md`. Append the full transcript JSON to each prompt so agents see every other agent's claims.

Each agent MUST:
- Rebut ≥1 specific claim from another agent by `claim.id`
- Update their confidence (may increase or decrease)
- Set `position_changed: true/false`

Spawn all 4 in parallel. Collect outputs, append to transcript.

### 4. DQI Calculation

After Round 2, run the DQI formula using all claims from the latest round:

```python
evidence_mult = {"direct": 1.0, "inferred": 0.7, "speculative": 0.4}
rebuttal_mult = {"unrebutted": 1.0, "conceded": 0.7, "unresolved": 0.5}

weights = []
for claim in all_claims:
    c = claim["confidence"] / 100
    e = evidence_mult.get(claim["evidence_strength"], 0.4)
    r = rebuttal_mult.get(claim.get("rebuttal_status", "unrebutted"), 1.0)
    weights.append(c * e * r)

dqi = sum(weights) / len(weights) if weights else 0.0
```

| DQI | Action |
|-----|--------|
| ≥ 0.8 | Skip Round 3. Proceed to adjudication. |
| 0.6 – 0.8 | Skip Round 3. Flag tensions in synthesis. |
| < 0.6 | Enter Round 3 — Convergence. |

Also count position changes: ≥2 agents changed position signals healthy convergence. 0 changes signals entrenched positions — note in synthesis.

### 5. Round 3 — Convergence (only if DQI < 0.6)

Render the Round 3 template from `templates/round-3-convergence.md`. Include the full transcript and a list of unresolved disagreements (claims defended in Round 2).

Each agent MUST end with one of:
- **accept** — full agreement
- **conditional_accept** — accept if condition is met
- **minority_report** — maintain dissenting position with rationale

Spawn all 4 in parallel. Collect outputs. Recalculate DQI with Round 3 claims included.

### 6. Adjudication

Build the synthesis JSON:

```python
synthesis = {
    "debate_id": "...",
    "topic": "...",
    "rounds_completed": 2 or 3,
    "dqi": <float>,
    "dqi_assessment": "high" | "moderate" | "low",
    "consensus_claims": [unrebutted or conceded claims],
    "tensions": [defended claims with split positions],
    "minority_reports": [agents who filed minority reports],
    "execute_automatically": dqi >= 0.8 and no minority reports
}
```

Post the synthesis to the kanban root card:
```bash
hermes kanban comment <root_card_id> "$(cat synthesis.json)"
hermes kanban complete <root_card_id> --summary "DQI: <score>"
```

Report to the user:
- Debate complete, DQI score and assessment
- Whether auto-execute or human review needed
- Any minority reports or tensions

## Pitfalls

### Agent returns malformed JSON
Try `jq` parse first. On failure, extract JSON block via regex `/```json\n([\s\S]*?)\n```/`. On complete failure, treat as `confidence: 0, evidence_strength: speculative`.

### Agent times out (≥600s)
Continue with remaining 3 agents. Note the missing agent in synthesis. Recalculate DQI without their claims. If ≥2 agents fail, abort and inform the user.

### False consensus (DQI > 0.9 after Round 1)
May indicate groupthink. Optionally inject an adversarial probe round where each agent argues AGAINST their initial position before proceeding to Round 2.

### Cross-profile project contamination
Include in the context file:
```markdown
**WORKING DIRECTORY:** /home/user/projects/your-project
**DO NOT** analyze files outside this directory.
```

### Round 2 rebuttals target wrong claim IDs
The prompt enforces claim ID format (`<agent>-claim-<N>`) but agents may hallucinate IDs. Validate: if a rebuttal targets a non-existent ID, log it but skip weighting.

## Verification

After a debate completes, verify:
1. All agent JSON files exist and parse: `ls round-*/<agent>.json`
2. Round 2 rebuttals cite valid claim IDs (check transcript)
3. DQI was calculated with the correct formula
4. Synthesis includes consensus claims, tensions, and minority reports
5. Kanban root card has the synthesis comment

For a dry-run test without real agents, call the orchestrator script with a small context and inspect the output:
```bash
~/.hermes/skills/adversarial-debate/scripts/debate-orchestrator.sh \
  --topic "Dry run test" \
  --context-file <(echo "## Test context") \
  --format feature \
  --max-rounds 1
```
