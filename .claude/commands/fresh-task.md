---
description: Start a fresh recurring autonomous task armed with a cron (Hermes coordinator)
allowed-tools: CronCreate, CronList, mcp__roo-state-manager__roosync_dashboard, Bash
---

# Command: /fresh-task

**Workspace:** Hermes (coordinator)
**Usage:** `/fresh-task <task description> [--every <cadence>]`
**Defaults:** cadence = 2h, off-minute cron (avoid fleet sync at :00/:30)
**Equivalent pattern:** mini-audit 12h (`f13101b3`), cluster-tour (`525e5650a8ac`)

---

## Purpose

Bootstrap a new **recurring autonomous task** in the Hermes coordinator scope: describe an objective + a cadence, and this command arms a `CronCreate` (recurring, session-only, auto-expires 7j) that runs the task on schedule, then announces it on the coordination dashboard so the cluster tracks it.

Use for recurring monitoring / audit / sweep duties — e.g. *« surveille la latence vLLM --every 1h »*, *« audit backlog PRs stalled --every 12h »*.

---

## Arguments

| Arg | Required | Default | Notes |
|-----|----------|---------|-------|
| `<task description>` | yes | — | The objective the cron executes each fire. Free text. |
| `--every <cadence>` | no | `2h` | One of: `30m`, `1h`, `2h`, `4h`, `6h`, `12h`, `24h`. |

---

## Steps

### 1. Parse arguments
- First positional (everything before a `--` flag) = **task description**.
- `--every <cadence>` = schedule. Absent → default `2h`.

### 2. Anti-duplicate check
`CronList` → if a task with the same description is already armed, **ask the user** before duplicating (avoid double-firing).

### 3. Convert cadence to off-minute cron expr

Use an **off-:00/:30 minute** to avoid the whole fleet hitting the API at the same instant (fleet stagger rule). Mapping:

| `--every` | cron expr | note |
|-----------|-----------|------|
| `30m` | `*/30 * * * *` | half-hour unavoidable at :00/:30 |
| `1h` | `7 * * * *` | hourly at :07 |
| `2h` | `41 */2 * * *` | coordinator cadence |
| `4h` | `19 */4 * * *` | |
| `6h` | `13 */6 * * *` | |
| `12h` | `17 */12 * * *` | mini-audit cadence |
| `24h` | `23 9 * * *` | daily 09:23 |

### 4. Build the cron prompt

Assemble the task description + the Hermes coordination protocol:

```
{task description}

---
Protocole coordination Hermes (chaque exécution) :
1. LIRE workspace-cluster-coordination (intercom, 10 derniers) AVANT d'agir — identifier signaux/ASK/INTENT non répondus.
2. Exécuter la tâche décrite ci-dessus.
3. POSTER le résultat sur workspace-cluster-coordination avec tag adapté ([INFO]/[DONE]/[ALERT]) — OBLIGATOIRE, aucune exception.
4. Vérifier CronList : si CE cron est expiré, le ré-armer (session-only, auto-expire 7j).

Règles anti-hallucination : exécuter la commande AVANT de reporter une erreur. Ne JAMAIS reporter une erreur résolue ou historique.
```

### 5. Arm the cron

```
CronCreate(
  cron: "<expr from step 3>",
  prompt: "<prompt from step 4>",
  recurring: true
)
```
→ session-only, auto-expires after 7 days (cluster convention). **Capture the returned job ID.**

### 6. Announce on coordination dashboard

```
roosync_dashboard(
  action: "append",
  type: "workspace",
  workspace: "cluster-coordination",
  tags: ["CLAIMED", "INFO"],
  content: "## [CLAIMED] Nouvelle tâche récurrente armée — {task description}\n\n**Par:** Hermes (po-2026)\n**Cadence:** {expr} (every {cadence})\n**Job ID:** {job ID}\n**Auto-expire:** 7j\n**Prochaine exécution:** au prochain match cron.\n\nRé-arme nécessaire à chaque nouvelle session Claude (cron session-only)."
)
```

### 7. Confirm

- `CronList` → show the new job armed alongside existing crons.
- Report to user: **job ID, cadence, next fire, dashboard post confirmation**.

---

## Usage Examples

```
/fresh-task Surveille la latence vLLM et alerte si >2s --every 1h
  → cron="7 * * * *", recurring
  → [CLAIMED] posted on cluster-coordination
  → job armed, next fire next :07

/fresh-task Audit du backlog de PRs stalled >48h --every 12h
  → cron="17 */12 * * *", recurring (= mini-audit cadence)

/fresh-task Vérifie la santé des MCP bridges 3/3
  → default --every 2h → cron="41 */2 * * *"
```

---

## Notes

- **Session-only** : le cron meurt quand la session Claude se ferme. Au démarrage d'une nouvelle session, ré-invoquer `/fresh-task` (ou laisser la tâche se ré-armer elle-même via step 4 du prompt).
- **Pas de Telegram par défaut** : la tâche poste sur dashboard. N'ajouter la livraison Telegram que si la tâche le justifie (escalade, alerte user) — via `deliver: "telegram:<chat-id>"` côté job config.
- **Conflit de cadence** : éviter de planifier plusieurs tâches au même top-minute (collision API). Le mapping off-minute ci-dessus distribue déjà les cadences courantes.
