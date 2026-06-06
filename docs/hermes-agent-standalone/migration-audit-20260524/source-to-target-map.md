# Source To Target Map

This map defines where each legacy area should land in Hermes Agent. It is a review plan, not a migration execution log.

## Target System Rules

| Target | Use For | Do Not Use For |
|---|---|---|
| `docs/hermes-agent-standalone` | reviewed architecture, policy, migration reports, source-of-truth docs | raw dumps, secrets, employee-sensitive notes |
| `~/ObsidianVault/HermesAgent/imports` | reviewed legacy imports with source lineage | live runtime config |
| `~/ObsidianVault/HermesAgent/playbooks` | repeatable business/engineering workflows | one-off notes |
| `~/ObsidianVault/HermesAgent/lessons` | incidents, mistakes, prevention rules | current secrets or logs |
| `~/ObsidianVault/HermesAgent/knowledge/domain` | promoted cross-project knowledge | unreviewed raw archive |
| `.hermes/skills` | reusable agent procedures | long business documents |
| `.hermes/profiles` | role-specific operating contracts | project-specific facts |
| `tools/` / `agent/` | tested runtime implementation | copied legacy scripts without tests |
| secure archive outside prompt-visible docs | `.env`, keys, tokens, auth material | any model-readable knowledge |

## Hermes Nous Mapping

| Source | Recommended Target | Action |
|---|---|---|
| `.hermes/active.md`, `.hermes/context.md`, `.hermes/decisions.md` | `~/ObsidianVault/HermesAgent/imports/hermes-nous/` then curated docs | preserve lineage, review for current relevance |
| `.hermes/docker-config-autogpt/*token*` | secure archive only | do not migrate content |
| `AGENTS.md`, `AI_MEMORY.md`, `GEMINI.md`, `MOC.md`, `Project_Spec.md`, `README.md` | `docs/hermes-agent-standalone/imports/hermes-nous/` or Obsidian imports | review for operating rules and architecture history |
| `docs/*.md` | `docs/hermes-agent-standalone` or Obsidian imports | keep architecture/runbooks; archive stale platform plans |
| `knowledge/**` | `~/ObsidianVault/HermesAgent/imports/hermes-nous/knowledge` then promote | curate into business, owner, company, AI-writing, operating-rule packs |
| `knowledge/employee/**` | private-scoped Obsidian notes or secure HR archive | restrict access; do not global-inject |
| `knowledge/owner-context-v0.1.md` | private owner/MD context pack | use as MD/Synerry source of truth after review |
| `lark-catalog/**` | structured reference import | preserve raw JSON/report lineage; do not prompt-dump |
| `lessons/**` | `~/ObsidianVault/HermesAgent/lessons` | convert incidents and prevention rules |
| `memory/profile/**`, `memory/projects/**` | profile/project context cards | dedupe with existing 40 project notes |
| `patterns/**` | domain notes and reusable playbooks | high-priority migration source |
| `playbooks/**` | `~/ObsidianVault/HermesAgent/playbooks` or `.hermes/skills` | convert decision workflows into practical agent procedures |
| `plugin-templates/**` | `docs/hermes-agent-standalone/reference` or archive | keep if still useful for future plugins |
| `review-queue/**` | migration review queue | inspect for unresolved risks; do not auto-promote |
| `skills/**` | `.hermes/skills` only after rewrite | port behavior, not raw legacy assumptions |
| `modules/tensorflow-ml-service/**` | archive/reference unless active need exists | dependency-heavy; do not bring venv |
| `scripts/**` | runtime port review | port selected scripts with tests |
| `web/**` | archive or separate product review | not automatically part of Hermes Agent |
| `.env` | secure archive only | do not copy, quote, or summarize values |

## Hermes Lab Mapping

| Source | Recommended Target | Action |
|---|---|---|
| `.hermes/active.md`, `.hermes/brief.md`, `.hermes/decisions.md`, `.hermes/progress.md` | Obsidian imports and architecture history | review for still-relevant decisions |
| root reports: `FINAL-REPORT-*`, `HANDOFF-*`, `LEAD-DEV-BRIEFING.md`, `PROJECT-SPEC.md`, `TEAM-ONBOARDING.md` | `docs/hermes-agent-standalone/imports/hermes-lab/` | summarize into current operating docs |
| `Blog/HermesFromClaude.md` | archive/reference | keep only if useful narrative/history |
| `config/.env*`, `config/.backup-key` | secure archive only | do not migrate content |
| `config/anti-words/global.json`, `config/group-mapping.json`, `config/identity.json` | runtime/config review | port only active policies |
| `docs/**` | architecture docs | review current relevance |
| `memory/team/facts/**` | `~/ObsidianVault/HermesAgent/imports/hermes-lab/memory/facts` then promote | use as source for lessons/rules/context |
| `memory/team/incidents/**` | `~/ObsidianVault/HermesAgent/lessons` | convert into incident cards |
| `memory/team/rules/**` | playbooks or global operating rules | high-priority migration source |
| `modules/claude-usage-guard/**` | AI governance/cost guard skill or docs | port conceptually if still useful |
| `modules/**` other | runtime port review | do not copy dependencies |
| `scripts/**` | runtime port review | port only scripts still needed |
| `skills/**` | `.hermes/skills` after modernization | update to Hermes Agent skill standards |
| `src/**` | runtime port review | compare with current Hermes Agent implementation before porting |
| `templates/**` | templates/reference | keep if current |
| `state/**` | archive only | do not make active memory without review |
| `node_modules/**` | ignore/archive | recreate from package manager if needed |

## Synerry Business Brain Mapping

| Capability | Primary Source | Target |
|---|---|---|
| Company identity and MD operating context | `MD Assist by AI`, HermesNous owner/company notes | private Synerry context pack and MD profile notes |
| Pitching War Room | MD Assist pitching flow + HermesNous playbooks | `~/ObsidianVault/HermesAgent/playbooks/synerry-pitch-war-room.md` |
| TOR Analyzer | MD Assist TOR analyzer + HermesNous proposal/sales playbooks | `.hermes/skills/business/tor-analyzer` or playbook first |
| Proposal Builder | HermesNous revenue-growth concepts | `.hermes/skills/business/proposal-builder` after review |
| Case Study Writer | HermesNous playbooks and project history | business skill/playbook |
| Market Research Pack | user images/blog + HermesNous research patterns | skill bundle for persona, FGD, competitive, trend, sentiment, survey |
| Finance/Margin Check | finance/turnaround concepts | private finance workflow, not global prompt |
| HR/People Context | employee SWOT and people notes | private-scoped HR notes with strict access |

## First Migration Batch Recommendation

Start with a small but high-value batch before bulk migration:

1. HermesNous `knowledge/owner-context-v0.1.md`
2. HermesNous `knowledge/Knowledge Operating Rules.md`
3. HermesNous `playbooks/*decision-driven-workflow-candidate.md`
4. HermesNous `patterns/pattern-*`
5. HermesNous `lessons/INC-*`
6. Hermes Lab `memory/team/rules/*.md`
7. Hermes Lab `memory/team/incidents/INC-*.md`
8. MD Assist Synerry/pitching/TOR source documents

This batch should prove the full import-review-promote-test loop before touching the rest.
