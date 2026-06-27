---
name: research-paper-writing
title: Research Paper Writing Pipeline
description: "Write ML papers for NeurIPS/ICML/ICLR: design→submit."
version: 1.1.0
author: Orchestra Research
license: MIT
dependencies: [semanticscholar, arxiv, habanero, requests, scipy, numpy, matplotlib, SciencePlots]
platforms: [linux, macos]
metadata:
  hermes:
    tags: [Research, Paper Writing, Experiments, ML, AI, NeurIPS, ICML, ICLR, ACL, AAAI, COLM, LaTeX, Citations, Statistical Analysis]
    category: research
    related_skills: [arxiv, ml-paper-writing, subagent-driven-development, plan]
    requires_toolsets: [terminal, files]

---

# Research Paper Writing Pipeline

## Trigger
Use for ML/AI research papers and preprints: designing experiments, literature review, LaTeX drafting, citation hygiene, review simulation, revision, and submission packages for NeurIPS/ICML/ICLR/ACL/AAAI/COLM-style venues.

## Hard Rules
1. Never hallucinate citations. Fetch/verify with scholarly tools or mark `[CITATION NEEDED]`.
2. Keep one-sentence contribution, claims, and experiments aligned.
3. Treat the process as an iterative loop: results can change the framing and reviewers can trigger new analysis.
4. Commit completed experiment batches and major draft revisions with descriptive messages.
5. Preserve full details in `references/full-runbook-20260503.md`; load it for templates, LaTeX preamble/TikZ, human eval, submission details, and review-simulation prompts.

## Minimal Phase Loop
1. **Project setup:** inspect repo, notes, results, configs, `.bib`, and draft files. Create TODOs.
2. **Literature review:** search arXiv/Semantic Scholar; record exact metadata/URLs; build related-work clusters.
3. **Experiment design:** map every experiment to a paper claim; define baselines, metrics, seeds, ablations, and failure cases.
4. **Execution/monitoring:** run reproducible scripts, save raw outputs, logs, configs, and environment details.
5. **Analysis:** aggregate statistics, uncertainty, significance where applicable, error analysis, and claim support table.
6. **Drafting:** write abstract→intro→method→experiments→related work→limitations/ethics with verified citations.
7. **Self-review/submission:** simulate reviewers, fix blocking issues, compile clean PDF, validate venue checklist.

## Starter Commands
```bash
git status --short
python -m pytest -q  # if repo has tests
rg -n "result|conclusion|finding|baseline|ablation" .
rg -n "TODO|CITATION NEEDED|\cite" paper/ || true
latexmk -pdf -interaction=nonstopmode paper.tex
```

## Output Shape
Return concrete artifacts: contribution sentence, claim→evidence matrix, experiment plan, citation table, draft section, reviewer concerns, or submission checklist. Avoid generic writing advice.

## Linked Reference
- `references/full-runbook-20260503.md` — full pre-slim paper-writing runbook with detailed steps, templates, LaTeX/TikZ snippets, tool recipes, and troubleshooting.
