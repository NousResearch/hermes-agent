# Contrats de travail SEO

## Document control
- **Scope:** Domaine : SEO
- **Owner:** Orchestrateur SEO
- **Maturity:** Experimental — SEO-only
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Avant workflow/cas réel
- **Linked documents:** CAPABILITIES.md, WORKFLOW.md, AGENTS.md
- **Evidence requirement:** Entrées, sorties, preuve, exit, escalade
- **Exit / validation criteria:** Handoff accepté seulement si complet
- **Risks / limits / escalation:** Entrée sensible/lacunaire/contradictoire : blocked/needs-human
- **Change history:** README.md
- **Exclusive responsibility:** Rendre exécutoires les contrats de travail SEO.

| Capacité | Entrées | Sortie | Preuve | Exit | Escalade |
|---|---|---|---|---|---|
| SEO-C01 | URLs/robots/headers fournis | registre sans causalité | ligne/fixture | chaque constat sourcé | accès/ambiguïté |
| SEO-C02 | éléments techniques + méthode | fiche symptôme + hypothèse | artefact/méthode | symptôme ≠ recommandation | rendu/crawl absent |
| SEO-C03 | corpus/objectif/audience | hypothèses testables | interne ou missing | hypothèse non classée fait | corpus absent |
| SEO-C04 | exports datés/fixtures | matrice cohérence | date/source/conflit | conflit conservé | enjeu élevé |
| SEO-C05 | constats QA-ready | plan P0/P1/P2 | score/owner/dépendance | approbation/rollback | effet production |
| SEO-C06 | livrables + rubriques | QA/critique | checklist/défaut | pass/rework/blocked | défaut critique |

## Handoff contract
Ajouts obligatoires : `capability_id`, `confidence`, `evidence_status`, `approval_required`, `rollback_hint`, `next_owner`. Un lien/proof absent ou une approbation requise sans demande échoue.

## Recommandation majeure
Tout impact sur indexation, contenu publié, infrastructure, budget ou réputation exige alternative « ne rien changer », incertitudes, réversibilité, owner humain, approbation et vérification post-action ; ce pilote n'exécute jamais l'action.
