---
name: scriptization-evaluator
description: When to write a deterministic script (no_agent) vs. a skill vs. a tool. A structured decision framework that extends the "Should it be a Skill or a Tool?" guide in CONTRIBUTING.md with the critical third dimension: scriptification.
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [script, skill, tool, automation, architecture, decision-framework, no_agent, cron]
    category: software-development
---

# Scriptization Evaluator

> **Extends** [CONTRIBUTING.md: Should it be a Skill or a Tool?](../../CONTRIBUTING.md#should-it-be-a-skill-or-a-tool)

Hermes gives you three implementation options for any automated behaviour:

| Option | Token cost | Reliability | Best for |
|--------|:----------:|:-----------:|:---------|
| **Tool** (`plugin/tools/`) | Low per-call | ★★★★★ | Binary data, streaming, real-time events, auth flows |
| **Skill** (`skills/` or `optional-skills/`) | Variable (LLM-driven) | ★★★☆☆ | Tasks that need reasoning, judgment, or in-context adaptation |
| **Script** (`~/.hermes/scripts/` + `no_agent`) | Zero per-run | ★★★★★ | Deterministic, repeatable, high-frequency operations |

This guide helps you choose. The trade-off is always the same: **LLM judgment costs tokens and introduces variance; scripts are deterministic and free, but can't think.**

---

## The Decision Tree

```
Is the operation purely deterministic
(no LLM judgment needed)?
│
├── YES → Can it be expressed as shell commands
│         or a standalone Python script?
│         │
│         ├── YES → ⭐ SCRIPT (no_agent cron / standalone script)
│         │
│         └── NO  → ⭐ TOOL (custom Python integration)
│
└── NO → Does it require real-time binary/stream handling
         or end-to-end auth integration?
         │
         ├── YES → ⭐ TOOL
         │
         └── NO  → ⭐ SKILL (LLM-driven)
```

The most common mistake is **defaulting to a Skill when a Script would do**. Every LLM call burns tokens — scripts cost nothing to run and never hallucinate.

---

## The Scoring Matrix

For a quantitative assessment, score each dimension on 0–10:

| Dimension | Score 0–3 | Score 4–7 | Score 8–10 |
|-----------|:---------:|:---------:|:----------:|
| **Determinism** | Needs judgment (paraphrase, summarize, classify) | Partially predictable (format varies) | Fully deterministic (file ops, API calls, math) |
| **Failure Cost** | Cosmetic / log-only | Operational disruption (blocked pipeline) | Data loss, security breach, governance failure |
| **Frequency** | < 1/day | 1–10/day | > 10/day (cron-level) |
| **Input Stability** | Free-form, varies per call | Semi-structured (markdown, JSON) | Fixed schema (timestamp, file path, env vars) |
| **Side Effects** | Read-only / no state | Read-write, idempotent | State mutation, destructive ops |

### Scoring formula

```
Total = (Determinism × FailureCost) + (Frequency × 2) + InputStability + (SideEffects × 2)
```

### Thresholds

| Score | Priority | Recommended action |
|:-----:|:--------:|:-------------------|
| **≥80** | P0 — must script now | Critical path: script + cron watchdog + alert |
| **40–79** | P1 — should script | Convert when bandwidth allows |
| **15–39** | P2 — optional | Skill is fine; script if it starts costing too much |
| **<15** | — keep as skill | LLM reasoning adds real value here |

---

## Real-World Examples

| Behaviour | Determinism | Failure Cost | Freq | Total | Verdict |
|:----------|:-----------:|:------------:|:----:|:-----:|:--------|
| Pre-action guard (validate env, check preconditions) | 10 | 10 | daily×3 | 130 | **P0 — Script** |
| API key / credential verification | 10 | 10 | hourly | 120 | **P0 — Script** |
| Memory watermark / storage cleanup | 9 | 7 | 1/h | 97 | **P0 — Script** |
| Cron job failure detection & alert | 8 | 6 | 1/h | 70 | **P1 — Script** |
| Daily aggregation / summarization | 5 | 4 | 1/d | 37 | **P2 — Skill** |
| Research / web discovery | 2 | 2 | as-needed | 12 | **Keep Skill** |
| Creative writing / persona interaction | 1 | 1 | on-demand | 5 | **Keep Skill** |

---

## Integration with Cron & no_agent

When a behaviour scores P0 or P1, implement it as a `no_agent=True` cron script:

```yaml
# ~/.hermes/config.yaml
cron:
  jobs:
    - name: preflight-check
      schedule: "*/30 * * * *"
      script: scripts/preflight.sh
      no_agent: true         # zero-token execution
      deliver: local         # silent unless script produces stdout
```

Benefits of no_agent scripts:
- **Zero token cost** — the LLM never runs for this job
- **Perfect determinism** — always the same logic, every time
- **Lightweight** — sub-second execution vs. multi-second agent spin-up
- **Silent failure** — non-zero exit / timeout sends an error alert; empty stdout = no noise

When no_agent can't work (the job needs reasoning), fall back to a regular cron skill.

---

## Pitfalls

**Over-scripting.** Not everything that *can* be a script *should* be a script. If the logic changes weekly, keep it as a skill — modifying a shell script is harder than updating skill prose.

**Under-scripting.** The most common trap: "I'll just write it as a skill for now" on something that runs every 5 minutes. After a week that's 2,016 unnecessary LLM calls. Script it early.

**Ignoring the third option.** Most people know Skill vs Tool. The Script option (`no_agent`) is newer and often overlooked — but it's the cheapest and most reliable execution mode Hermes offers.

---

## Related

- [CONTRIBUTING.md: Should it be a Skill or a Tool?](../../CONTRIBUTING.md#should-it-be-a-skill-or-a-tool) — the canonical Skill vs Tool decision guide
- [Cron Job Configuration](../../cron/README.md) — detailed no_agent setup reference
- [no_agent Script Pattern](../no-agent-pattern/README.md) — best practices for deterministic scripts
