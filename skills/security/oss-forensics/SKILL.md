---
name: oss-forensics
description: Supply chain investigation, evidence recovery, and forensic analysis for GitHub repositories.
category: security
triggers:
  - "investigate this repository"
  - "check for supply chain compromise"
  - "recover deleted commits"
  - "forensic analysis of [owner/repo]"
---

# OSS Security Forensics Skill

An investigation framework for researching open-source supply chain attacks, recovering deleted evidence, and producing structured forensic reports.

## 0. Initialization
- Create an investigation directory: `mkdir investigation_[repo_name]`.
- Initialize evidence store: `python scripts/evidence-store.py --store investigation_[repo_name]/evidence.json list`.
- Load [forensic-report.md](./templates/forensic-report.md) as a starting point for the investigation findings.

## 1. Phase 1: Planning and IOC Extraction
- Analyze the user prompt to identify:
  - Target repository (`owner/repo`)
  - Target actors (GitHub handles)
  - Time window of interest
  - Indicators of Compromise (SHAs, domains, IPs, file paths)
- **Tool**: Reasoning only or `execute_code` for string extraction.

## 2. Phase 2: Parallel Evidence Collection
Use `delegate_task` to spawn specialized sub-agents (Investigators):

### Git Investigator
- **Action**: Clone repo, check `git log`, `git reflog`, `git branch -a`.
- **Goal**: Find force-pushes, deleted branches, or uncharacteristic commit patterns.
- **Reference**: [recovery-techniques.md](./references/recovery-techniques.md)

### GitHub API Investigator
- **Action**: Query Issues, PRs, releases, and contributor lists.
- **Goal**: Identify deleted/modified issues or PRs, and changes in collaborator permissions.

### Archive/Wayback Investigator (Optional)
- **Action**: Use `web_extract` to search Wayback Machine for historical snapshots.
- **Reference**: [github-archive-guide.md](./references/github-archive-guide.md)

## 3. Phase 3: Evidence Consolidation
- Use `scripts/evidence-store.py` to record all findings with unique IDs (`EV-XXXX`).
- Every claim in the final report MUST cite at least one evidence ID.

## 4. Phase 4: Hypothesis Formation and Verification
- Formulate hypotheses (e.g., "Actor X force-pushed to hide Malicious Commit Y").
- Verify hypotheses by cross-referencing Git history with Archive/API data.
- **Reference**: [investigation-templates.md](./references/investigation-templates.md)

## 5. Phase 5: Final Report Generation
- Populate the [forensic-report.md](./templates/forensic-report.md) template.
- Ensure the executive summary clear states the verdict (Compromised/Clean/Inconclusive).
- Ensure the timeline is complete and evidence-backed.

## Safety Guidelines
- Do not run suspicious code found in repositories locally.
- Only analyze code using `execute_code` in a sandboxed environment if necessary.
- Redact secrets/API keys found during investigation from the final report.
