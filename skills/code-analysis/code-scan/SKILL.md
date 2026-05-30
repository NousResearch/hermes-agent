---
name: code-scan
hermes.tags: [on-demand, code-analysis, project-mapping]
---

# Code Scan — JIT Orchestration Skill

## Purpose
Load on-demand to run the code-scan pipeline against any target directory.
Never auto-injected; triggered explicitly by the user.

## Orchestration Steps

1. **Confirm target:** Ask user for target directory or use working directory.
2. **Run scan:** `python scripts/code-scan/scan_project.py <target_dir> --output <temp_scan.json>`
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

## Output Format

## Project: <name>
- **Description:** <one-line>
- **Languages:** <detected language distribution>
- **Frameworks:** <detected frameworks array>
- **Structure:** <top-level dirs + key files>
- **Import map:** <top 5 most-imported modules>
- **Files:** <total_files> total, <files_with_imports> with imports
