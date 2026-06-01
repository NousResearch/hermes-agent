---
name: code-scan
hermes.tags: [on-demand, code-analysis, project-mapping]
---

# Code Scan — JIT Orchestration Skill

## Purpose
Load on-demand to run the code-scan pipeline against any target directory.
Never auto-injected; triggered explicitly by the user.

## Mode Selection

Use the **UA-005 mode router** (`scripts/code-scan/run_ua.py`) for all new code-scan
runs.  It replaces the manual orchestration steps below with explicit mode selection.

```bash
python scripts/code-scan/run_ua.py --target <dir> --out <bundle_dir> [--mode <mode>]
```

### Available Modes

| Mode | Pipeline Stages | Use When… |
|---|---|---|
| **inventory** (default: no) | scan + imports only | Quick inventory of files and dependencies. |
| **structure** (default) | scan + imports + graph + validation | Default — balanced structural analysis with graph validation. |
| **review** | structure + analytics + context envelope + report | Deep review with analytics, subagent context, and markdown report. |
| **delta** | incremental scan + delta summary against prior manifest | Detect changes since a prior run (use `--prior-manifest`). |
| **preflight** | structure + entrypoints/hubs + subagent context | Subagent handoff preparation with structure and context. |
| **full** | all available deterministic enrichers | Everything available — analytics, context, report. |

### Mode Routing Guarantees

- **Validation failures are never hidden.** Any mode that runs graph assembly
  also runs the validation gate and writes `validation.json`.
- **Quick modes avoid unnecessary graph analytics.** `inventory` and `structure`
  do not produce `analytics.json` or `subagent-context.json`.
- **Missing enrichers are recorded as skipped.** When upstream beads (UA-002/003/004)
  are not present, optional artifacts are omitted and noted in `artifacts_missing`
  metadata — nothing is fabricated.
- **Default mode is `structure`**, preserving backward compatibility with the
  existing pipeline behavior (scan → imports → graph → validation).

## Orchestration Steps (Legacy Manual Path)

> **Prefer** `run_ua.py` mode selection above.  The manual steps below remain
> documented for compatibility but are superseded by the mode router.

1. **Confirm target:** Ask user for target directory or use working directory.
2. **Check fingerprints & run scan:**
   - If `.hermes/code-state/fingerprints.json` exists → use `--incremental` for a fast diff scan.
   - If fingerprints are absent → `--incremental` is safe and behaves like a full scan.
   - If user requests a forced full rescan → use `--full` instead.
   - Command: `python scripts/code-scan/scan_project.py <target_dir> --incremental --output <temp_scan.json>`
3. **Run import extraction:** `python scripts/code-scan/extract_imports.py <temp_scan.json> > <temp_imports.json>`
4. **Read artifacts:** Read both JSON outputs.
5. **Synthesize (LLM-only):** From scan data produce:
   - Project name (from directory or package.json/pyproject.toml)
   - One-line description (inferred from frameworks + files, not hallucinated)
   - Framework/stack narrative (from detected frameworks + language distribution)
6. **Render summary:** Output as structured markdown (format below).
7. **Clean up:** Temp files can be deleted; they are not tracked artifacts.

## Constraints
- Never hallucinate file structures — only report what scan scripts return.
- If scan fails, report the error; do not guess.
- .hermesignore rules are already enforced by scan_project.py.
- Only synthesize name, description, and framework fields. Everything else is deterministic.
- `--incremental` relies on `.hermes/code-state/fingerprints.json`; omit or use `--full` to force a complete rescan.

## Output Format

## Project: <name>
- **Description:** <one-line>
- **Languages:** <detected language distribution>
- **Frameworks:** <detected frameworks array>
- **Structure:** <top-level dirs + key files>
- **Import map:** <top 5 most-imported modules>
- **Files:** <total_files> total, <files_with_imports> with imports
