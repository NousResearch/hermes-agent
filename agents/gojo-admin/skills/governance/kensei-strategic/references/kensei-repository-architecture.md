# Kensei Dual-Repo Architecture

Established 2026-05-25 after Kensei mistakenly applied customisations to the
wrong repo. This document defines the architecture and workflows for the two
Hermes Agent repositories on Sahil's VPS.

## Repository Roles

| Repo | Path | Role | Tracks | Runs Hermes? |
|---|---|---|---|---|
| **KenseiAgent** | `~/repos/KenseiAgent/` | **SINGLE SOURCE OF TRUTH** | origin→Sahil-SS9/KenseiAgent.git, upstream→NousResearch | ✅ Yes — via `~/.local/bin/hermes` → `.venv/bin/hermes` |
| **hermes-agent** | `~/.hermes/hermes-agent/` | **Vanilla upstream clone** | origin→NousResearch, fork→Sahil-SS9/hermes-agent.git | ❌ No |

## Critical Rule

**ALL customisations, tweaks, new features, and source code changes go into
KenseiAgent.** Never apply source patches to `~/.hermes/hermes-agent/`. That repo
exists only for creating clean PRs to upstream NousResearch.

## Architecture in Detail

### KenseiAgent (`~/repos/KenseiAgent/`)
- Full Hermes Agent source code (from upstream merge on 2026-05-25)
- Custom companion layer: content engine, ops scripts, Postiz self-hosted configs,
  planning docs, profile prompts, .claude skills
- Customisations already applied:
  - Disabled MCP server display in welcome banner (hermes_cli/banner.py)
  - Web search fallback Tavily→Firecrawl on 4xx (tools/web_tools.py)
  - `feature/profile-wizard` branch (PR #31781) — handled separately
- Has `upstream` remote: `https://github.com/NousResearch/hermes-agent.git`

### hermes-agent (`~/.hermes/hermes-agent/`)
- Vanilla checkout of upstream. Should never diverge from `origin/main`.
- Has branches for PR work that can be submitted upstream:
  - `feature/profile-wizard` — PR #31781 already submitted
  - `kensei/main` — integration branch of old customisations (delta reference)
  - `kensei/feat/gbrain-tool` — not merged upstream, kept for revival
  - `kensei/feat/kanban-stale-run-id-guard` — not merged upstream, kept
- Has `fork` remote: `https://github.com/Sahil-SS9/hermes-agent.git`

## Workflows

### A. Pulling upstream changes into KenseiAgent

```
cd ~/repos/KenseiAgent
git fetch upstream main
git merge upstream/main --allow-unrelated-histories
# Resolve conflicts in AGENTS.md, .gitignore, README.md (keep ours)
# Check if customisations conflict with new upstream code
# Repoint symlinks if venv changes needed
# Verify: hermes --version shows Project: /home/kensei/repos/KenseiAgent
```

### B. Moving customisations between repos

When customisations were accidentally applied to the wrong repo:

1. Add the source repo as a remote to KenseiAgent:
   `git remote add hermes-clone /home/kensei/.hermes/hermes-agent`
2. Fetch: `git fetch hermes-clone`
3. Cherry-pick commits: `git cherry-pick <commit-hash>`
4. Remove temp remote: `git remote remove hermes-clone`
5. Push KenseiAgent to origin: `git push origin main`
6. Reset upstream clone if needed: `git reset --hard origin/main`

### C. Creating an upstream PR

1. Ensure upstream clone is on clean `main` matching origin/main
2. Create feature branch: `git checkout -b feat/my-feature origin/main`
3. Make changes, commit, push to `fork` remote
4. Submit PR via `gh pr create --repo NousResearch/hermes-agent`
5. Do NOT port these changes to KenseiAgent unless Sahil explicitly asks

### D. Deploying Hermes from KenseiAgent

After upstream merge or code changes in KenseiAgent:

```
cd ~/repos/KenseiAgent
uv venv --python 3.11
uv pip sync pyproject.toml
uv pip install -e .
ln -sf ~/repos/KenseiAgent/.venv/bin/hermes ~/.local/bin/hermes
ln -sf ~/repos/KenseiAgent/.venv/bin/hermes-acp ~/.local/bin/hermes-acp
ln -sf ~/repos/KenseiAgent/.venv/bin/hermes-agent ~/.local/bin/hermes-agent
hermes --version  # Verify: shows Project: /home/kensei/repos/KenseiAgent
```

## Pitfalls

- **Wrong repo**: The most common mistake. Before making ANY Hermes source change,
  stop and ask: "Is this a customisation (→ KenseiAgent) or an upstream PR contribution
  (→ hermes-agent)?"
- **Unrelated histories**: KenseiAgent and upstream have NO shared git ancestry.
  Always use `--allow-unrelated-histories` when merging upstream.
- **Conflict files**: Only 3 files exist in both repos: AGENTS.md, .gitignore,
  README.md. Always keep KenseiAgent's versions (they contain custom instructions/
  ignore rules).
- **Symlink verification**: After repointing, run `hermes --version` to confirm
  `Project:` points to KenseiAgent. The status output is the fastest sanity check.
- **Venv regeneration**: After pulling upstream, deps may change. Run `uv pip sync
  pyproject.toml` before restarting any gateway services.