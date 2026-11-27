# Hermes Official Submission — Kubrick

This document is the operator checklist for getting Kubrick into Nous Research’s
**official optional skills catalog** while also publishing the community path.

## Two publication lanes

| Lane | Command / path | Trust |
|---|---|---|
| **Official** (PR) | Merge into `NousResearch/hermes-agent` as `optional-skills/creative/kubrick` | `official` |
| **Community** (now) | GitHub repo / tap `scrimshawlife-ctrl/Kubrick` | `community` |

Pursue both: community installability first, official PR for catalog trust.

## Acceptance criteria (official optional skill)

- [x] Bounded purpose: cinematic creative production + continuity + production handoff
- [x] Explicit activation conditions in `description` + **When to Use / When Not to Use**
- [x] No private filesystem paths, API keys, or user-specific assumptions
- [x] Deterministic CLI + fail-closed `NOT_COMPUTABLE` behavior
- [x] Dependencies documented; platforms declared
- [x] Supporting scripts via `${HERMES_SKILL_DIR}`
- [x] Permissive MIT license + real author attribution
- [x] Provenance notes in `SKILL.md`
- [ ] Local Hermes activation test (positive)
- [ ] Local Hermes negative activation test
- [ ] Upstream PR opened against `NousResearch/hermes-agent`

## Frontmatter contract (current)

```yaml
name: kubrick
author: Daniel Meyer / Applied Alchemy Labs
license: MIT
platforms: [linux, macos, windows]
metadata.hermes.category: creative
```

Stable slug: **`kubrick`** — do not rename between releases.

## 1. Validate in-repo

```bash
cd /path/to/Kubrick
python3 scripts/kubrick.py do check --action skill
python3 scripts/kubrick.py do check --action smoke
python3 scripts/validate_hermes_skill.py
```

## 2. Install / sync local Hermes skill

Preferred install location for local trust:

```bash
./install.sh
# or:
rsync -a --delete \
  --exclude '.git/' --exclude '.github/' --exclude 'out/' --exclude '__pycache__/' \
  ./ ~/.hermes/skills/kubrick/
```

Avoid nested paths like `~/.hermes/skills/creative/kubrick/kubrick`.

```bash
hermes skills list | grep -i kubrick
python3 ~/.hermes/skills/kubrick/scripts/kubrick.py do check --action smoke
```

## 3. Activation tests

Positive (should load Kubrick):

```bash
hermes chat --toolsets skills -q \
  "Use the Kubrick skill to create a visual design system for a psychological science-fiction film."
```

Negative (must **not** load Kubrick):

```bash
hermes chat --toolsets skills -q \
  "Calculate the SHA-256 checksum of README.md."
```

## Community layout note

Hermes GitHub taps default to listing skill directories under `skills/`.
This repository ships a hub-discoverable copy at:

```text
skills/kubrick/SKILL.md
```

Root-level files remain the development workspace. After editing root, refresh:

```bash
./scripts/sync_hub_skill.sh
```

## 4. Community publish (no upstream wait)

```bash
# Hub groupings (already in repo root)
# skills.sh.json → grouping title "Cinematic Design"

hermes skills publish . --to github --repo scrimshawlife-ctrl/Kubrick

# Or as a tap:
hermes skills tap add scrimshawlife-ctrl/Kubrick
hermes skills search kubrick --source github
```

## 5. Official PR packaging

```bash
gh repo fork NousResearch/hermes-agent --clone
cd hermes-agent
git checkout -b feat/add-kubrick-skill

# From the Kubrick repo:
/path/to/Kubrick/scripts/package_optional_skill.sh /path/to/hermes-agent

# Verify payload
find optional-skills/creative/kubrick -maxdepth 3 -type f | sort
git diff --check
git status --short

git add optional-skills/creative/kubrick
git commit -m "feat(skills): add Kubrick cinematic design skill"
git push -u origin feat/add-kubrick-skill

gh pr create \
  --repo NousResearch/hermes-agent \
  --title "feat(skills): add Kubrick cinematic design skill" \
  --body-file /path/to/Kubrick/PR_BODY.md
```

After merge:

```bash
hermes skills search kubrick --source official
hermes skills install official/creative/kubrick
hermes skills inspect official/creative/kubrick
```

## Size note

Kubrick is larger than many creative optional skills (corpus + schemas + evals).
That bulk is intentional: deterministic retrieval and continuity need the corpus.
The packaging script strips git/CI/historical planning docs before the PR copy.

If review asks for a thinner first merge, keep:

- `SKILL.md`, `LICENSE`, `README.md`, `QUICKSTART.md`
- `scripts/` (or a documented core subset)
- `schemas/`, `references/` (corpus + patterns), `templates/`, `examples/`
- a reduced `evals/` smoke set

and leave extended historical docs in the standalone GitHub repo.

## Do not ship in the upstream tree

- `.git/`, `.github/`
- local `out/`, caches, secrets
- machine-specific install notes
- OpenClaw-only packaging (lives on branch `openclaw` in the standalone repo)
