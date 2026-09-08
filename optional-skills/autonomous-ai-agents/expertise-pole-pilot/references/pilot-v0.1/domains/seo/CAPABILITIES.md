# Carte de capacités SEO

## Document control
- **Scope:** Domaine : SEO
- **Owner:** Stratège/priorisateur SEO
- **Maturity:** Experimental — SEO-only
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Ajout signal, outil ou source primaire
- **Linked documents:** EXPERTS.md, CONTRACTS.md, EVALUATION.md
- **Evidence requirement:** Sortie mesurable et liens owner/contrat/cas
- **Exit / validation criteria:** Chaque priorité a quatre liens
- **Risks / limits / escalation:** Dépendance non détenue : escalade orchestrateur
- **Change history:** README.md
- **Exclusive responsibility:** Cartographier uniquement les capacités SEO et leurs dépendances.

| ID | Objectif | Dépendances | Risque | Livrable | Preuve | Maturité | Priorité | Owner |
|---|---|---|---|---|---|---|---:|---|
| SEO-C01 | crawlabilité/indexabilité observable | URLs/headers/robots fournis | élevé | registre observations | fixture/source datée | experimental | P0 | expert technique |
| SEO-C02 | qualité technique/rendu observable | HTML/headers/métriques fournis | moyen | fiche anomalie | ID preuve + méthode | experimental | P1 | expert technique |
| SEO-C03 | pertinence contenu/intention testable | corpus + objectif | élevé | hypothèses contenu | fait/hypothèse séparés | experimental | P1 | analyste preuves |
| SEO-C04 | cohérence et attribution des signaux | exports datés/fixtures | élevé | matrice cohérence | provenance/date/conflit | experimental | P0 | analyste preuves |
| SEO-C05 | plan priorisé non exécutif | constats QA-ready | élevé | options P0/P1/P2 | score, owner, approbation | experimental | P0 | stratège |
| SEO-C06 | QA et critique | livrables précédents | élevé | rapports QA/critique | rubriques/régression | experimental | P0 | QA/critique |

Flux : `SEO-C01 + SEO-C02 + SEO-C03 + SEO-C04 → SEO-C05 → SEO-C06`. Dépendance absente = `blocked`, jamais comblement implicite.
