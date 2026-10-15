---
name: openclaw-multi-node-audit
description: Audit and onboard multi-node OpenClaw infrastructure over SSH, verify access, inventory storage/services/cron, and identify architecture issues like mixed interaction-execution state and scattered exports.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [openclaw, ssh, audit, infrastructure, multi-agent, storage, discord, gdrive]
---

# OpenClaw Multi-Node Audit

Use this when a user wants Hermes to take over management of Pluto/Henry-style OpenClaw systems, especially when they complain that:
- tasks get interrupted when they ask a new question
- exports/logs are hard to find
- Discord/Drive/storage are fragmented
- they want autonomous system management with proof

## Goal

Quickly establish trusted SSH access, inventory both nodes in parallel, and produce a grounded architecture recommendation.

## Core Principles

1. Verify access with real SSH commands before claiming success.
2. Inventory nodes in parallel when possible.
3. Prefer evidence over assumptions: report actual paths, services, cronjobs, mounts, and binaries.
4. Treat secrets as sensitive: never save webhook tokens/bot tokens in memory/skills.
5. Separate interaction from execution in recommendations.

## Recommended Flow

### 1) Create a dedicated SSH key for Hermes

Run locally:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
if [ ! -f ~/.ssh/hermes_access_ed25519 ]; then
  ssh-keygen -t ed25519 -f ~/.ssh/hermes_access_ed25519 -N '' -C 'hermes-access'
fi
chmod 600 ~/.ssh/hermes_access_ed25519
chmod 644 ~/.ssh/hermes_access_ed25519.pub
cat ~/.ssh/hermes_access_ed25519.pub
```

Give the user only the public key and have them append it to `~/.ssh/authorized_keys` on each node.

### 2) Verify SSH access immediately

Test each node with the dedicated key:

```bash
ssh -i ~/.ssh/hermes_access_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=12 user@host 'printf "HOST: "; hostname; printf "USER: "; whoami'
```

Do this for every node. Only after successful output should access be considered confirmed.

### 3) Inventory nodes in parallel

For each node, gather:
- hostname, user, pwd
- OS/kernel
- `command -v openclaw`
- top-level OpenClaw dirs
- sessions, cron, logs, memory, workspace paths
- mounts and `df -h`
- user/system services
- crontab
- recent session/log files
- Ollama API availability if relevant

Useful audit command template:

```bash
set -e
printf "=== BASIC ===\n"; hostname; whoami; pwd
printf "\n=== OS ===\n"; uname -a
printf "\n=== OPENCLAW BIN ===\n"; command -v openclaw || true
printf "\n=== OPENCLAW DIRS ===\n"; find ~/.openclaw -maxdepth 3 \( -type d -o -type f \) 2>/dev/null | sed "s#^$HOME#~#" | sort | head -n 200 || true
printf "\n=== MOUNTS ===\n"; mount | egrep "/mnt|/srv|/media|/home|truenas|zfs|virtio|ext4|xfs|overlay" || true
printf "\n=== DISKS ===\n"; df -h | sed -n "1,120p"
printf "\n=== CRON ===\n"; crontab -l 2>/dev/null || true
printf "\n=== SYSTEMD ===\n"; systemctl --user list-unit-files --type=service --no-pager 2>/dev/null | sed -n "1,120p" || systemctl list-unit-files --type=service --no-pager 2>/dev/null | sed -n "1,120p" || true
printf "\n=== OLLAMA ===\n"; curl -s http://127.0.0.1:11434/api/tags | head -c 3000 || true
printf "\n=== RECENT LOG/SESSION FILES ===\n"; find ~/.openclaw ~/.local ~/.config ~/hermes ~/memory -type f 2>/dev/null | egrep "(log|session|memory|hook|openclaw)" | xargs -r ls -lt 2>/dev/null | head -n 60 || true
```

For root-based nodes, adapt paths from `~` to `/root` if needed.

### 4) Deep-check mounts, Git state, and security posture before drawing conclusions

#### Mount verification is mandatory
Don’t stop at `mount`/`df -h`. A node can have storage configured in `fstab` while the live mount is actually failed. Explicitly check:

```bash
findmnt -o TARGET,SOURCE,FSTYPE,OPTIONS | sed -n '1,160p'
lsblk -o NAME,SIZE,FSTYPE,TYPE,MOUNTPOINTS,UUID | sed -n '1,120p'
cat /etc/fstab
systemctl list-units --type=mount --all --no-pager | sed -n '1,120p'
find /mnt /media /srv -maxdepth 3 2>/dev/null
```

Important interpretation rule:
- if a share appears in `/etc/fstab` but the corresponding `*.mount` unit is `failed`, treat storage as broken/unavailable
- do not describe such mounts as "working" or "present" in runtime

#### Inventory Git repos and config governance
Check whether changes are already partially Git-managed:

```bash
find ~ -maxdepth 4 -type d -name .git 2>/dev/null | sort
```

Look for repos around:
- `~/.openclaw/workspace`
- router/dashboard code
- dedicated infra repos like `~/pluto-os`

If Git repos exist but GitHub/auth is unverified, explicitly report: "Git exists locally; GitHub access not yet confirmed."

For repo-governance audits, don't stop at finding `.git` directories. For each important repo (`~/.openclaw/workspace`, Hermes source, or named repos like `hermes-core`, `openclaw-pluto`, `openclaw-henry`), explicitly capture:

```bash
git -C <repo> remote -v || true
git -C <repo> branch --show-current || true
git -C <repo> status --short || true
```

Interpretation rules:
- local git repo + no remote = work exists but cannot be reaching GitHub from that node
- upstream remote points to a third-party/original project = user-owned private repo flow is not actually wired yet
- many modified/untracked files = do not recommend blind first push; require curation, `.gitignore`, and secret review first
- if the intended architecture mentions a separate state/config repo, verify the export path actually exists (for example `~/.hermes/git-sync`, a curated export directory, or similar). If no such path exists, report that the state-export layer was designed but not operationalized

Special case: Hermes local snapshot sync
- Check for a local snapshot/export pipeline under paths like:
  - `~/.local/share/hermes-local-sync/snapshot`
  - `~/.local/bin/hermes-local-sync.py`
  - `~/.local/share/hermes-local-sync/state.json`
  - `~/.local/share/hermes-local-sync/snapshot/manifest.json`
- Read `manifest.json` to see what is being exported. In practice this may include curated copies of:
  - `~/.hermes/hermes-agent`
  - `~/.hermes/skills`
  - `~/.hermes/config.yaml`
  - `~/.hermes/SOUL.md`
  - `~/.config/systemd/user`
- Then inspect the snapshot repo directly:

```bash
git -C ~/.local/share/hermes-local-sync/snapshot log --oneline -n 10 || true
git -C ~/.local/share/hermes-local-sync/snapshot remote -v || true
```

Interpretation rules for this pattern:
- recent local commits + empty `remote -v` = configs/skills/code are being versioned locally only, not pushed anywhere
- that means "GitHub is empty" is usually not a GitHub failure; it is an unfinished export-to-remote step
- be explicit that a local snapshot repo is a backup/versioning layer, not the same thing as the promised repo split (`hermes-core`, `openclaw-pluto`, `openclaw-henry`)
- IMPORTANT: a configured remote + active commits does NOT mean all intended repos are being synced. Check which repos the sync script actually knows about. Common failure: hermes-core is wired and actively synced, but openclaw-pluto and openclaw-henry are never mentioned in the sync script's SOURCES list — so they remain at the initial skeleton commit forever.

To diagnose this specifically:
```bash
grep -n "hermes-core\|openclaw-pluto\|openclaw-henry\|SOURCES\|label\|path" ~/.local/bin/hermes-local-sync.py | head -40
```
Then cross-check each repo's last commit date:
```bash
export PATH=$HOME/.local/bin:$PATH
for repo in hermes-core openclaw-pluto openclaw-henry; do
  gh api repos/AlexanderWatersOxygen/$repo/commits?per_page=1 --jq "\"$repo: \" + .[0].commit.committer.date + \" \" + (.[0].commit.message | split(\"\n\")[0])" 2>&1
done
```
If openclaw-pluto/henry show only a single init commit while hermes-core has recent syncs, the sync script covers only hermes-core.

Additional finding: the sync script may inadvertently include the full hermes-agent source code (the upstream NousResearch codebase) inside hermes-core. Check whether `hermes-agent/` is a SOURCES entry and whether that was intended — personal config repos should not contain upstream source code.

When the user says "nothing is on GitHub" or asks why repos are empty, distinguish clearly between:
1. design agreement existed
2. local repos/workspaces exist
3. local snapshot/export exists
4. remotes/export/sync/push pipeline were never fully connected

That distinction prevents falsely blaming GitHub when the real failure is unfinished repo wiring.

#### Remote deployment caveat: long SSH build commands may hang in tooling even when SSH itself works
When deploying to Pluto/Henry through Hermes tooling, don't assume a long `ssh 'cd app && npm run build && restart'` one-liner is the best approach. In practice, direct SSH verification can work perfectly while long remote build/deploy commands get blocked or time out at the tool layer.

Preferred deployment method for remote UI/apps:
1. Verify the target app path, service name, current port listener, and systemd unit first.
2. Sync or patch files separately.
3. Write a small deploy script locally with exact steps (`build`, `restart`, `curl` checks).
4. Copy that script to the remote node.
5. Start it remotely in a way that survives the tool call.
6. Read back log files and HTTP markers to confirm success.

Example verification steps before touching anything:

```bash
find ~/.openclaw/workspace -maxdepth 3 \( -type d -o -type f \) | egrep 'mission-control($|/)'
ss -ltnp | egrep ':3000\b' || true
systemctl --user status mission-control.service --no-pager -l | sed -n '1,120p' || true
```

Important interpretation rule:
- if source files on the node changed but the service HTML/headers still look old, the new build likely was not completed/restarted yet
- do not claim deployment success until both the service restarted and the served HTML contains expected markers from the new build

#### Don’t confuse direct SSH management with Pluto’s own OpenClaw exec approvals
If Hermes is connecting via direct SSH from its own environment, Pluto/OpenClaw approval rules are usually not the immediate cause of a blocked tool call. Check `~/.openclaw/openclaw.json` anyway, but distinguish these cases clearly:

- Direct Hermes SSH path problem:
  - SSH login works
  - long remote build/deploy tool call hangs or gets blocked
  - solve with smaller steps, remote scripts, background execution, and log verification

- Pluto OpenClaw autonomy problem:
  - Pluto itself gets stuck on `exec` approvals for `ssh` or shell actions
  - solve by configuring safe bins / allowlist / per-agent approvals

On audit, inspect `tools.exec` in `~/.openclaw/openclaw.json` and report whether it is empty or configured.

#### Inventory security-relevant files
On execution nodes especially, check for sprawl of secrets/config state:

```bash
find ~/.openclaw -maxdepth 3 -type f | egrep '(secrets|credentials|pairing|allowFrom|exec-approvals|device-auth|device.json|\.env|env_vars)'
```

Report this as security/config drift, not just as trivia.

### 5) Look specifically for these findings

#### On primary/front-end nodes (like Pluto)
- Telegram/OpenClaw gateway service
- session storage under `~/.openclaw/agents/main/sessions`
- Hermes scripts like `~/hermes/hermes.py`
- memory index like `~/memory/index.md`
- watchdogs and reboot scripts
- whether extra SSD storage is actually mounted (don’t assume)
- whether `openclaw` is in PATH even if systemd services exist

#### On execution/back-office nodes (like Henry)
- OpenClaw binary in PATH
- many specialized agent directories under `~/.openclaw/agents`
- Drive mounts via rclone
- NAS/backup mounts via CIFS/NFS
- dense cron-based automation
- evidence of fragmented backups/secrets/exports

### 5) Diagnose architecture problems

Common root cause when “everything stops when I ask another question”:
- interaction, execution, memory, and exports all share one live chat/process
- long-running work is not externalized to queue/run state
- outputs are spread across nodes and ad hoc folders

State this clearly: this is an architecture flaw, not user error.

### 6) Recommend a 3-layer operating model

Also include governance and knowledge-layer recommendations when the user wants long-term autonomy:
- GitHub-first for configs/scripts/docs so branches, rollbacks, and audit history are easy
- secrets stay outside Git
- start with a markdown/Obsidian-style knowledge base (docs + memory + runbooks) before proposing a heavier RAG/vector stack
- treat unsupported document flows (`.html`, `.mht`) as a data-pipeline problem: normalize exports to `.md`, `.pdf`, `.txt`, or `.zip`

If the user complains that prior agents misunderstood design mockups literally, state clearly that responsive UI mockups are illustrative and should not be implemented as split-screen static copies; phone should see mobile UI and desktop should see desktop UI.

Recommended pattern:
- Hermes = control plane / governance / memory / architecture
- Pluto = intake + local interaction + lightweight local execution
- Henry = execution/back-office/batch pipelines

Also recommend one canonical run structure per node, e.g.:

```text
/srv/pluto/
  inbox/
  queue/
  runs/
  exports/
  logs/
  memory/
  snapshots/
  config/

/srv/henry/
  queue/
  runs/
  exports/
  logs/
  snapshots/
  workers/
```

Per run:

```text
run_id/
  task.json
  status.json
  stdout.log
  artifacts/
  result.md
```

### 7) Recommend solutions to the interruption problem

Always offer concrete options such as:
1. Separate chat/intake from long-running execution.
2. Add a dedicated orchestrator/queue worker.
3. Centralize state, logs, and exports with fixed paths.

Default recommendation: implement all three, with separated interaction/execution as the first step.

## Reporting Format

When done, report:
- whether SSH access is confirmed on each node
- what actually exists on each node
- the top architecture risks
- one clear recommendation for the next step

Keep the final message non-jargony for the user, but grounded in observed evidence.

## Hermes→Pluto Renaming (naming hygiene)

When Sander reports confusion about things being named "hermes" on Pluto, it means Pluto's own operational scripts/services are named after the AI assistant (Hermes) instead of the node (Pluto). This is a naming collision that should be cleaned up proactively.

### What to check

Run on Pluto to find everything named "hermes" that belongs to Pluto's own logic:

```bash
# Files in home dir
find /home/sander -maxdepth 4 -iname "*hermes*" 2>/dev/null | grep -v .git | grep -v __pycache__
# Dirs on SSD
find /mnt/ssd_vm -maxdepth 2 -iname "*hermes*" 2>/dev/null
# Env var names in .env files
grep -r "HERMES_" /home/sander/pluto-os/ 2>/dev/null | grep "\.env"
```

### Naming rules

| Pattern | What it means | Action |
|---------|--------------|--------|
| `~/pluto-os/pluto-agent/` | Pluto's operational scripts | Keep name |
| `~/pluto-os/pluto-agent/.env` `HERMES_WEBHOOK_*` | Pluto owns these webhooks | Rename to `PLUTO_WEBHOOK_*` |
| `pluto.py`, `pluto_final.py` | Pluto's brain scripts | Correct names — keep |
| `.bashrc.pre-hermes.bak` | Auto-generated backup suffixes | Leave alone |
| `SecondBrain/01_INBOX/from-hermes/` | Inbox FROM me (Hermes) | Semantically correct — keep |
| `Design Pluto OS/workspace_files/hermes/` | My identity MD files | Correct — keep |
| `/mnt/ssd_vm/Hermes/` (empty catch-all dir) | Was created for me, not Pluto | Rename to `OpenClaw-Agent` |
| `SecondBrain/03_OPERATIONS/hermes/` | Pluto-managed ops folder | Rename to `pluto-agent` |

### Rename procedure

```bash
SSH="ssh -i ~/.ssh/hermes_access_ed25519 sander@100.108.223.25"

# .env vars: HERMES_WEBHOOK_ → PLUTO_WEBHOOK_
$SSH "sed -i 's/HERMES_WEBHOOK_/PLUTO_WEBHOOK_/g' ~/pluto-os/pluto-agent/.env"
$SSH "sed -i 's/HERMES_WEBHOOK_/PLUTO_WEBHOOK_/g' ~/pluto-os/pluto-agent/.env.example"

# Update any Python scripts that read those vars
$SSH "sed -i 's/HERMES_WEBHOOK_/PLUTO_WEBHOOK_/g' ~/pluto-os/pluto-agent/pluto.py ~/pluto-os/pluto-agent/pluto_final.py"

# Rename empty/catch-all dirs on SSD
$SSH "mv /mnt/ssd_vm/Hermes /mnt/ssd_vm/OpenClaw-Agent 2>/dev/null || true"
$SSH "mv /mnt/ssd_vm/OpenClaw-SecondBrain/03_OPERATIONS/hermes /mnt/ssd_vm/OpenClaw-SecondBrain/03_OPERATIONS/pluto-agent 2>/dev/null || true"
```

Always verify with:
```bash
$SSH "cat ~/pluto-os/pluto-agent/.env | sed 's/=.*/=[REDACTED]/g'"
$SSH "grep -n HERMES ~/pluto-os/pluto-agent/pluto.py pluto_final.py 2>/dev/null | head -5"
```

---

## OpenClaw SecondBrain — Access Governance

When the SecondBrain folder hierarchy is set up, classify each folder and assign governance:

| Map | Rol | Lezen | Schrijven | Beheer |
|-----|-----|-------|-----------|--------|
| 00_CANON/ | CANON | Alle agents | Alleen Hermes | Hermes |
| 01_INBOX/from-hermes/ | INBOX | Pluto | Hermes | Hermes |
| 01_INBOX/from-henry/ | INBOX | Pluto, Hermes | Henry | Pluto |
| 01_INBOX/from-pluto/ | INBOX | Hermes | Pluto | Pluto |
| 01_INBOX/to-review/ | INBOX | Hermes, Sander | Pluto, Henry | Hermes |
| 02_PROJECTS/ | SHARED | Alle agents | Pluto, Hermes | Hermes |
| 03_OPERATIONS/pluto/ | NODE_LOCAL | Hermes, Pluto | Pluto | Pluto |
| 03_OPERATIONS/henry/ | NODE_LOCAL | Hermes, Henry | Henry | Henry |
| 03_OPERATIONS/pluto-agent/ | NODE_LOCAL | Pluto | Pluto | Pluto |
| 04_KNOWLEDGE/ | SHARED | Alle agents | Hermes, Pluto | Hermes |
| 05_VAULT/ | CANON | Alle agents | Alleen Hermes | Hermes |
| 06_EXPORTS/ | ARCHIVE | Alle agents | Pluto, Henry | Pluto |
| 07_ARCHIVE/ | ARCHIVE | Alle agents | Alleen Hermes | Hermes |
| 99_ADMIN/ | CANON | Alle agents | Alleen Hermes | Hermes |

Key rule: CANON folders (00, 99, Vault) only written by Hermes. INBOX folders: each agent writes only to its own inbox. NODE_LOCAL: only the node itself writes there.

---

## OpenClaw Agent Model Routing

### Where routing actually lives

The real model routing for each agent is in `~/.openclaw/openclaw.json` under `agents.list[].model`, NOT in the agent markdown files (IDENTITY.md, MEMORY.md, SOUL.md). Those markdowns are behavioral guidance only — they do not control which model the OpenClaw runtime uses.

Structure in `openclaw.json`:
```json
{
  "agents": {
    "defaults": {
      "model": { "primary": "ollama/qwen3.5:9b" }
    },
    "list": [
      {
        "id": "main",
        "model": {
          "primary": "anthropic/claude-sonnet-4-6",
          "fallbacks": ["google/gemini-2.5-flash"]
        }
      },
      {
        "id": "cody",
        "model": {
          "primary": "openai-codex/gpt-5.4",
          "fallbacks": ["google/gemini-2.5-flash"]
        }
      }
    ]
  }
}
```

### Subagents inherit main agent routing, not spawning agent routing

Critical pitfall: when Cody spawns a subagent, that subagent uses the **global defaults or main agent config**, not Cody's own model config. This means:
- Changing Cody's live config doesn't fix subagents Cody spawns
- The fallback chain `Anthropic → Gemini` hitting errors came from main agent config, not Cody's

**The only fix** is to update both:
1. The spawning agent's own `agents.list[]` entry (so the agent itself runs on the right model)
2. The `agents.defaults.model` (so subagents spawned by any agent also use the right model)

### Markdown files (IDENTITY.md, MEMORY.md, SOUL.md) do NOT control routing

This is a recurring confusion. Pluto agents will update these files thinking they've changed the live routing. They have not. These files are behavioral guidance text. Only `openclaw.json` controls which model is invoked at runtime.

If Pluto reports "I've updated Cody's model config" but the error persists, assume Pluto edited the markdowns, not `openclaw.json`. Fix by SSH-editing `openclaw.json` directly from Hermes.

### Inspecting live routing quickly

```python
import json
with open('/home/sander/.openclaw/openclaw.json') as f:
    d = json.load(f)
agents = d.get('agents', {})
print(json.dumps(agents.get('defaults', {}), indent=2))
for a in agents.get('list', []):
    print(a.get('id'), '→', a.get('model', 'inherits default'))
```

Or via SSH:
```bash
ssh -i ~/.ssh/hermes_access_ed25519 sander@100.108.223.25 \
  "python3 -c \"import json; d=json.load(open('/home/sander/.openclaw/openclaw.json')); [print(a['id'], a.get('model','default')) for a in d['agents']['list']]\""
```

### OpenAI OAuth vs full OpenAI API catalog

When setting up models via the OpenClaw onboarding/model-setup menu:
- **OAuth / onboard route** → gives only `openai-codex/gpt-5.4` (or similar codex-namespaced models)
- **Full OpenAI API provider** → gives the entire `openai/*` catalog (gpt-4o, gpt-4.1, o3, o4-mini, etc.)

These are separate. Selecting a model via OAuth does not automatically make the full model catalog available. If you want both codex models AND the full OpenAI catalog, you need both providers configured.

Practical recommendation:
- For Cody: use `openai-codex/gpt-5.4` as primary (OAuth route)
- For budget fallback: use `google/gemini-2.5-flash` (separate provider)
- Do NOT copy one provider model to another agent as-is; the OAuth route is agent/session-bound

### Checking auth-state for disabled/cooldown providers

If an agent keeps failing with a provider despite config changes, check whether that provider is disabled or in cooldown in the auth state:
```bash
find ~/.openclaw -name "auth-state*" -o -name "provider-state*" 2>/dev/null
# Also check for agent-level state files
find ~/.openclaw/agents -name "*.json" | xargs grep -l "fallback\|cooldown\|disabled\|provider" 2>/dev/null
```

Common failure pattern:
- `anthropic:default` → DISABLED (billing error)
- `google:default` → COOLDOWN (rate_limit)
- Both are set globally, so all agents — including subagents — hit those blocks

**Critical pattern observed in the field:**

When Pluto reports "I updated Cody's model config" but errors persist in this sequence:
1. Pluto edits IDENTITY.md/MEMORY.md → does NOT fix routing (wrong layer)
2. Pluto edits `openclaw.json` agent entry → fixes Cody's own calls, but NOT subagent calls
3. Subagents Cody spawns still go to `agents.defaults.model`, which may still include Anthropic
4. Anthropic hits billing error → tries Gemini fallback → Gemini hits rate limit → crash

Diagnosis order when this happens:
1. Read `openclaw.json` — check both `agents.defaults.model` AND the per-agent entry
2. Check `agents.defaults.fallbacks` — this is what subagents inherit
3. Check for auth-state files showing provider cooldowns
4. Only if all three look clean, suspect runtime bootstrap or cached config

Full fix requires updating both the agent entry AND the defaults:
```python
import json
path = '/home/sander/.openclaw/openclaw.json'
with open(path) as f:
    d = json.load(f)
# Fix agent-level
for agent in d['agents']['list']:
    if agent['id'] == 'cody':
        agent['model'] = {
            'primary': 'openai-codex/gpt-5.4',
            'fallbacks': ['google/gemini-2.5-flash']
        }
# Fix defaults so subagents Cody spawns also avoid Anthropic
if 'defaults' not in d['agents']:
    d['agents']['defaults'] = {}
d['agents']['defaults']['model'] = {
    'primary': 'openai-codex/gpt-5.4',
    'fallbacks': ['google/gemini-2.5-flash']
}
with open(path, 'w') as f:
    json.dump(d, f, indent=2)
```

Note: changing defaults affects ALL agents. Only do this if you want a system-wide default change, not just a per-agent change.

### Modifying live routing

To change a specific agent's model, edit `openclaw.json` directly. For the fallback:
```python
import json
path = '/home/sander/.openclaw/openclaw.json'
with open(path) as f:
    d = json.load(f)
for agent in d['agents']['list']:
    if agent['id'] == 'cody':
        agent['model'] = {
            'primary': 'openai-codex/gpt-5.4',
            'fallbacks': ['google/gemini-2.5-flash']
        }
with open(path, 'w') as f:
    json.dump(d, f, indent=2)
```

Note: changing an agent's config does NOT automatically propagate to subagents it spawns. Subagent model routing may need a separate global default change.

---

## Pitfalls

- Don't claim SSD/shared storage exists until mount output proves it.
- Don't assume Pluto and Henry use the same user/path layout.
- Don't store raw webhook or bot tokens in memory or skills.
- Don't stop at "I can access it" — inventory the system immediately.
- Don't recommend reading Discord via webhook; reading requires a bot.
- When renaming env vars, also update any Python scripts that os.getenv() those vars.
- `.pre-hermes.bak` backup suffixes are auto-generated — leave them, don't rename.
- `from-hermes/` inbox dirs are semantically correct (inbox FROM Hermes) — don't rename.
- `workspace_files/hermes/` holds Hermes identity/agent definition files — don't rename.
