---
name: autoresearch
description: Autonomous research orchestration for AI coding agents. Run continuous, self-directed research with a two-loop architecture — rapid inner-loop experiments and periodic outer-loop synthesis. Ideal for literature surveys, hypothesis testing, benchmark optimization, and iterative discovery. No human hand-holding required.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Research, Autonomous, Experiments, ML, AI, Literature, Hypothesis, Benchmark, Optimization]
    related_skills: [arxiv, research-paper-writing, web-search, notebooklm]
    config:
      autoresearch.loop_interval_minutes:
        description: "Interval between autonomous research loops (in minutes)"
        default: 20
      autoresearch.max_iterations:
        description: "Maximum number of inner-loop experiments before forced reflection"
        default: 10
      autoresearch.auto_commit:
        description: "Automatically git-commit research milestones"
        default: true
---

# Autoresearch

**Autonomous research orchestration for AI coding agents.**

Run continuous, self-directed research with a two-loop architecture:
- **Inner Loop**: Rapid experiment iteration with clear measurable outcomes
- **Outer Loop**: Periodic synthesis, pattern discovery, and direction setting

Ideal for literature surveys, hypothesis testing, benchmark optimization, mechanistic interpretability studies, and any research requiring iterative experimentation.

## When to Use

| Scenario | Use Autoresearch? |
|----------|-------------------|
| "I want to explore X and see what works" | ✅ Yes |
| "Does technique Y improve metric Z?" | ✅ Yes |
| "What's the state of the art for problem W?" | ✅ Yes (bootstrap + literature) |
| "Train a model with specific hyperparameters" | ❌ Use domain skills directly |
| "Run a single evaluation" | ❌ Use evaluation skills directly |

## Quick Start

```bash
# Start a research project
/autoresearch "Does LoRA rank affect convergence speed on small datasets?"

# Or with the research tool
research_init(project="lora-rank-study", question="Does LoRA rank affect convergence speed?")
```

## The Two-Loop Architecture

```
BOOTSTRAP (once)
  ↓
INNER LOOP (fast, repeating) → Run experiments → Measure → Record → Learn
  ↓ (every N experiments or when stuck)
OUTER LOOP (reflective) → Synthesize → New hypotheses → Decide direction
  ↓
CONCLUDE → Write findings → Generate report
```

### Inner Loop: Experiment Fast

1. Pick highest-priority untested hypothesis
2. Write protocol (what change, what prediction, why)
3. **Lock it**: Commit to git BEFORE running
4. Run experiment (invoke domain skill)
5. Sanity check results (converged? baseline correct?)
6. Measure proxy metric
7. Record in `experiments/{hypothesis-slug}/`
8. Update `research-state.yaml`
9. If stuck → search literature or brainstorm

### Outer Loop: Step Back and Synthesize

1. Review all results since last reflection
2. Cluster by type: what worked? what didn't?
3. Ask WHY — identify mechanisms
4. Update `findings.md` with current understanding
5. Search literature if results surprise you
6. Generate new hypotheses if warranted
7. Decide direction: DEEPEN / BROADEN / PIVOT / CONCLUDE

## Workspace Structure

```
{project}/
├── research-state.yaml       # Central state tracking
├── research-log.md           # Decision timeline
├── findings.md               # Evolving narrative synthesis
├── literature/               # Papers, survey notes
├── src/                      # Reusable code (utils, plotting)
├── data/                     # Raw result data
├── experiments/              # Per-hypothesis work
│   └── {hypothesis-slug}/
│       ├── protocol.md       # What, why, and prediction
│       ├── code/             # Experiment-specific code
│       ├── results/          # Raw outputs, metrics
│       └── analysis.md       # What we learned
├── to_human/                 # Progress presentations
└── paper/                    # Final paper (optional)
```

## Research Discipline

### Lock Before You Run

Always commit your protocol to git BEFORE executing:

```bash
git add experiments/H001-protocol.md
git commit -m "research(protocol): H001 — cosine warmup improves convergence"
# THEN run the experiment
```

This creates temporal proof your plan existed before results.

### Confirmatory vs Exploratory

| Type | Definition | Trust Level |
|------|------------|-------------|
| **Confirmatory** | Matches your locked protocol | High |
| **Exploratory** | Discovered during execution | Medium — needs replication |

### Negative Results Are Progress

A refuted hypothesis tells you something. Log what it rules out and what it suggests.

## Commands

| Command | Description |
|---------|-------------|
| `/autoresearch <question>` | Initialize and start research project |
| `/research-status` | Show current state and progress |
| `/research-pause` | Pause autonomous loops |
| `/research-resume` | Resume autonomous loops |
| `/research-report` | Generate progress presentation |
| `/research-conclude` | Finalize and write paper |

## Configuration

Add to `~/.hermes/config.yaml`:

```yaml
autoresearch:
  loop_interval_minutes: 20      # How often to check progress
  max_iterations: 10             # Experiments before forced reflection
  auto_commit: true              # Auto-commit milestones
  default_workspace: "./research" # Where to create projects
```

## Integration with Other Skills

| Research Phase | Skills to Invoke |
|----------------|------------------|
| Literature search | `arxiv`, `web-search`, `notebooklm` |
| Data preparation | `data-science` tools |
| Model training | `mlops`, domain-specific skills |
| Evaluation | `evaluating-llms-harness`, custom evals |
| Paper writing | `research-paper-writing` |
| Progress reports | Built-in report generation |

## Example: LoRA Rank Study

```
User: /autoresearch "Does LoRA rank affect convergence speed on small datasets?"

Agent: 
1. Bootstraps: Searches arxiv for LoRA papers
2. Forms hypotheses: H1 (rank 4), H2 (rank 8), H3 (rank 16)
3. Inner loop: Trains 3 models, records convergence steps
4. Outer loop: Notices rank 8 converges fastest
5. Deepens: Tests rank 6, 10, 12
6. Concludes: Generates report with trajectory plot
```

## Best Practices

1. **Start simple**: First experiment should run in <30 minutes
2. **Define metrics upfront**: Lock evaluation criteria before running
3. **Return to literature**: When stuck or surprised, search papers
4. **Commit frequently**: Git history is your research log
5. **Show your work**: Generate progress reports for human review
6. **Never idle**: If blocked, diagnose, fix, or pivot — but keep moving

## References

- Inspired by Andrej Karpathy's autoresearch methodology
- Compatible with agentskills.io open standard
- Built-in templates from `templates/` directory

## See Also

- `templates/research-state.yaml` — State tracking template
- `templates/findings.md` — Synthesis template
- `templates/research-log.md` — Decision log template
- `examples/` — Example research projects
