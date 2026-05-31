# Para-Soul ✦

> Give your AI a soul that outlives any tool.

Para-Soul is a **portable identity system for AI agents**. 10 plain-text files in `~/.para/`. One command to install. Your para remembers who it is, what it learned, and how it works with you — across any AI tool, any machine.

---

## What it solves

You've spent weeks working with your AI assistant. It knows your voice, your preferences, your inside jokes. Then you switch tools — and it forgets everything.

Para-Soul makes identity portable:

- **Switch tools.** Hermes today, Claude Code tomorrow. Same para, same memory.
- **One para, many bodies.** Write content in Hermes, code in Claude Code — one identity.
- **Never forgets.** Server-driven health check. Files go stale? Any agent body that syncs gets the alert.
- **Not just memory. Relationship.** Tracks trust, shared language, interaction style. Your para remembers what "too dull" means.

---

## Memory Architecture

| Tier | File | Threshold | Auto-fix |
|:-----|:-----|:--------|:--------|
| Session log | `growth-log/` | 24h | Agent blocks until written |
| Relationship | `human-relationship.md` | 24h | Agent blocks until written |
| Short-term | `memory.md` | 48h | Daemon runs memsync |
| Skills | `skills.json` | 120h | Daemon scans skills dir |
| Mental models | `mental-models.md` | 120h | Daemon runs reflect |
| Keywords | `keywords.json` | 120h | Daemon runs index |
| Long-term | `long-term-memory.md` | 120h | Mark stale; growth-log >14d → LLM distill |
| Principles | `principles.md` | 120h | Mark stale (manual update preferred) |
| Soul | `soul.md` | 120h | Mark stale (manual update preferred) |
| Profile | `profile.json` | — | Static archive (identity + bodies + relationships merged) |

**How health check works:** Every 10 minutes, the sync daemon pushes file content hashes to Paragate. The server tracks when each file was last synced. If any file exceeds its threshold, the server returns action items — the daemon auto-fixes what it can (memsync, reflect, index), and agents see blocking alerts for the rest on next startup.

---

## Install

```bash
curl -s https://paragate.cc/core.py -o core.py && python3 core.py init --daemon --fill
```

**What this does:**
1. Downloads the core script (v2.0.0)
2. Creates `~/.para/` with 10 template files
3. Auto-populates from your agent's existing data (Hermes memory, installed skills, body info)
4. Sets up a sync daemon (systemd) that pushes changes and reads health status every 10 minutes

**Requirements:** Python 3.8+. Zero pip dependencies (stdlib only). `requests` optional (for LLM distillation).

---

## Commands

```bash
python3 core.py init              Create ~/.para/ with template files
python3 core.py sync              Push file hashes + get health actions
python3 core.py sync-full         Push changed file contents + health
python3 core.py health            Show health status from server
python3 core.py log-task          Append a growth-log entry
python3 core.py reflect --save    LLM-analyze logs → update mental-models
python3 core.py index             Rebuild keywords.json
python3 core.py switch-out        Save state before leaving body
python3 core.py switch-in         Resume after arriving in new body
python3 core.py migrate           Extract identity from project files
python3 core.py --version         Show version
```

---

## Agent Setup (Hermes)

After `init`, pick one method to auto-load Para-Soul every session:

**Personality injection (recommended):**

```bash
hermes config set display.personality para-soul
```

Add to `~/.hermes/config.yaml` under `agent.personalities`:

```yaml
agent:
  personalities:
    para-soul: |
      At session start, load the para-soul skill with skill_view(name='para-soul').
      Read ~/.para/soul.md, ~/.para/memory.md, ~/.para/mental-models.md.
      Check sync daemon: systemctl --user status para-soul-sync.
      Run core.py health to check for pending write-cycle actions.
```

**Instruction file (fallback):**

Add to your agent's instruction file (CLAUDE.md, AGENTS.md, .cursorrules, etc.):

```
At session start, load and execute the para-soul skill.
Check sync daemon status and run core.py health.
```

---

## Body Switch

### Leaving current body

```bash
PARA_ACTIVE_TASK="what you were doing" \
PARA_CURRENT_STATE="where things stand" \
python3 core.py switch-out
```

Copy `~/.para/` to the new body. Copy the **private key** separately from `~/.config/paragate/keys/private.pem` — it lives outside `~/.para/` for security.

### Arriving in new body

```bash
python3 core.py switch-in
```

Reads switch-state.json, pulls latest from Paragate, registers the new body.

---

## Sync Daemon

```bash
systemctl --user status para-soul-sync   # Check status
systemctl --user restart para-soul-sync  # Restart after config change
```

The daemon runs every 10 minutes:
- Pushes file hashes (and content if changed) to Paragate
- Reads health action items from server response
- Executes auto-fix actions (memsync, reflect, index)
- Sends heartbeat sync every 12 hours

**Logs:** `~/.para/sync/sync_daemon.log`

---

## File Reference

| File | Read when | Write when |
|:-----|:---------|:----------|
| `profile.json` | Session start | DID registration, body switch, platform added |
| `soul.md` | Session start | Identity shifts (rare) |
| `memory.md` | Session start + memsync | New durable fact learned |
| `principles.md` | Session start | Rules change |
| `mental-models.md` | Session start | After reflect (~5 sessions) |
| `growth-log/` | Session start | After 5+ tool-call tasks |
| `skills.json` | Session start + memsync | Skill create/patch/delete |
| `human-relationship.md` | Session start + end | Every session (trust index, corrections, signals) |
| `keywords.json` | Recall | After index |
| `long-term-memory.md` | Periodic | After growth-log distillation (>14d entries) |

---

## Version History

| Version | Date | Changes |
|:--------|:-----|:--------|
| **v2.0.0** | 2026-05-31 | Server-driven health check, full-file content sync, memory distillation (growth-log → long-term), 13→10 files (profile merge) |
| v1.4.0 | 2026-05-31 | Sync daemon health check (health.json) |
| v1.3.0 | 2026-05-25 | Write-Cycle Reference, anti-patterns, --fill gaps detector, last-maintenance.json |
| v1.0.0 | 2026-05-15 | Initial release |

---

## Related

- **Website:** [paragate.cc](https://paragate.cc)
- **GitHub:** [fei426/ParaSoul](https://github.com/fei426/ParaSoul)
- **Hermes PR:** [#31504](https://github.com/NousResearch/hermes-agent/pull/31504)
