# Sources SEO : autorité, fraîcheur et conflits

## Document control
- **Scope:** Domaine : SEO
- **Owner:** Analyste de preuves et sources
- **Maturity:** Blocked for external evidence; experimental for process
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Avant usage et à l'échéance de source
- **Linked documents:** ../../support-pole/RESEARCH-PROTOCOL.md, CONTRACTS.md
- **Evidence requirement:** Registre : extrait/date/hiérarchie/conflit/portée
- **Exit / validation criteria:** Aucune assertion externe importante sans récupération
- **Risks / limits / escalation:** Source absente/obsolète : hypothèse ou blocage
- **Change history:** ../../reports/PHASE-0-INVENTORY.md
- **Exclusive responsibility:** Administrer l'autorité et la fraîcheur des sources SEO.

## Statut actuel
**BLOCKED BY NO-SPEND GUARD.** Huit recherches prévues ont été refusées. Aucune source SEO web n'a été récupérée ; une URL candidate n'est donc pas une preuve consultée.

## Hiérarchie lors d'un cycle explicitement autorisé
1. documentation officielle actuelle des moteurs/outils ;
2. normes techniques ;
3. données de première main du propriétaire avec date/droit ;
4. dépôts OSS pour structures, schémas, tests et limites ;
5. secondaire pour contexte, clairement étiqueté.

## Pistes non vérifiées
| Candidate | Usage potentiel | Statut |
|---|---|---|
| `developers.google.com/search/docs/fundamentals/seo-starter-guide` | source primaire candidate | non récupérée |
| `rfc-editor.org/rfc/rfc9309` | norme candidate | non récupérée |
| `github.com/GoogleChrome/lighthouse` | patterns OSS audit/test | non récupérée |
| `github.com/janreges/siteone-crawler` | patterns OSS crawl | non récupérée |

## Observations locales inspectées
| ID | Provenance | Observation limitée | Portée |
|---|---|---|---|
| SRC-LOCAL-01 | `sources/hermes-agent-v2026.8.31/AGENTS.md` lignes 11–14, 71–87 | archive : skills/plugins et validation E2E valorisés | structure seulement |
| SRC-LOCAL-02 | `sources/obsidian-skills/skills/obsidian-markdown/SKILL.md` lignes 10–19 | procédure numérotée et vérification de rendu | skill vérifiable seulement |

Registre réel requis : `SRC-id | claim | URL/path | publisher | published_at | accessed_at | excerpt | freshness | authority | conflicts | reviewer | status`.
