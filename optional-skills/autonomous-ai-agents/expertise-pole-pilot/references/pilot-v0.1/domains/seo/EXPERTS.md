# Experts SEO et frontières

## Document control
- **Scope:** Domaine : SEO
- **Owner:** Orchestrateur SEO
- **Maturity:** Experimental — SEO-only
- **Last reviewed:** 2026-09-08
- **Freshness rule:** À chaque fusion/séparation rôle
- **Linked documents:** agents/, CONTRACTS.md, WORKFLOW.md
- **Evidence requirement:** Profil, anti-scope et handoff
- **Exit / validation criteria:** Aucun rôle « fait tout »
- **Risks / limits / escalation:** Chevauchement/lacune : escalade owner
- **Change history:** README.md
- **Exclusive responsibility:** Définir uniquement les rôles SEO et leurs frontières.

| Rôle | Capacités | Fait | Ne fait pas | Handoff |
|---|---|---|---|---|
| technique | SEO-C01, SEO-C02 | observe artefacts techniques fournis | ne crawl/modifie/promeut | registre → analyste |
| analyste preuves | SEO-C03, SEO-C04 | provenance/fraîcheur/conflits | ne priorise pas business | bundle → stratège |
| stratège | SEO-C05 | options et ordre justifiés | n'exécute pas | plan → QA |
| contrôleur QA | SEO-C06 | contrats, preuves, sécurité | ne corrige pas silencieusement | QA → critique |
| orchestrateur | SEO-C01…SEO-C06 | états et handoffs | aucun verdict de substitution | état/escalade |
| critique indépendant | SEO-C06 | cherche défauts/angles morts | ne conçoit ni auto-approuve | avis → owner |

Une personne peut tenir les rôles séquentiellement dans une fixture, jamais fusionner la critique avec sa propre conception lors d'une promotion.
