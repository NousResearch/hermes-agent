# Agent Pattern Catalog for Hermes OS

## Purpose

Use this as a reference when designing workflows. Do not apply every pattern automatically.

For each task, Hermes should choose the simplest reliable pattern.

## Patterns

1. **Prompt Chaining** — sequential steps with validation.
2. **Routing** — send task to correct specialist.
3. **Parallelization** — split independent work across workers.
4. **Reflection** — draft → critique → revise.
5. **Tool Use** — select and call external tools safely.
6. **Planning** — goal → dependency graph → tasks.
7. **Multi-Agent Collaboration** — coordinated specialists.
8. **Memory Management** — short-term, episodic, long-term.
9. **Learning/Adaptation** — feedback improves prompts/policies.
10. **Goal Monitoring** — KPIs, checkpoints, drift detection.
11. **Exception Recovery** — retry, fallback, escalate.
12. **Human-in-the-Loop** — approval for risk/edge cases.
13. **RAG** — retrieve grounded knowledge.
14. **Inter-Agent Messaging** — structured agent communication.
15. **Resource-Aware Optimization** — cheap model vs strong model routing.
16. **Reasoning Techniques** — CoT, ToT, debate, self-consistency.
17. **Evaluation & Monitoring** — tests, drift, cost, quality.
18. **Guardrails** — safety, permissions, injection protection.
19. **Prioritization** — value/risk/effort/urgency scoring.
20. **Dependency Management** — order tasks before execution.

## Workflow Classification Rule

When designing or executing a workflow, first classify the task:

- Simple task → single agent or tool use.
- Multi-step task → prompt chaining or planning.
- Many independent tasks → parallelization.
- Uncertain routing → router + clarification.
- High-risk action → human-in-the-loop.
- Quality-sensitive output → reflection + evaluation.
- Cost-sensitive task → resource-aware routing.
- Long-running project → planning + memory + monitoring.
