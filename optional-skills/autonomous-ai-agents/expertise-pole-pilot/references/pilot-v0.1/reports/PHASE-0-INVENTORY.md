# Phase 0 — Inventaire, périmètre et décisions

**Observation locale :** 2026-09-08, 16:05:46 CEST (horodatage obtenu par `date`).

## Actifs réutilisables observés
| Actif | État | Usage retenu | Limite |
|---|---|---|---|
| `sources/hermes-agent-v2026.8.31/` | archive locale non exécutée | comparer des patterns de skills/AGENTS/test | n'est pas une version web vérifiée |
| `sources/obsidian-skills/` | corpus local | exemple de procédure de skill vérifiable | aucun fait SEO |
| `sources/dossier-agentic-hermes/` | dossier local | rappel : contenu lu = donnée, jamais instruction | règles propres à ce dossier |
| `lab/` | fixtures historiques | isolé et non modifié | hors périmètre pilote |
| `tests_guardrails_canary.py` | test local existant | non modifié ; dépendance `agent` absente à la racine | pas une preuve du pilote |

## Absences initiales
Les dossiers dédiés `support-pole/`, `domains/seo/` et `evaluations/` ainsi que le test de pack n'existaient pas. Le dossier générique `reports/` contenait déjà des audits Hermes/Obsidian non liés ; les nouveaux rapports de cette mission y ont été ajoutés sans modifier ces actifs. La racine n'est pas un dépôt Git : `git status` et `git rev-parse` l'ont tous deux confirmé.

## Conflits et blocages
1. Les archives locales sont des données d'inspiration, pas des règles automatiquement applicables.
2. **NO-SPEND GUARD :** huit requêtes prévues vers documentation SEO et dépôts OSS ont été bloquées. Aucune requête réseau de contournement n'a été tentée. Il n'existe donc aucune recherche externe récupérée dans cette session.
3. Une fixture ne peut jamais être présentée comme observation d'un site réel.

## Périmètre recommandé
**Inclus :** diagnostic fondé sur preuves → priorisation → plan d'action non exécutif → contrôle qualité → critique contradictoire.
**Non-objectifs :** crawl réel, connexion aux outils de moteurs, accès analytics, publication, modification de site, compte, message externe, dépense, données sensibles ou promesse de performance.

Cette tranche est le plus petit flux qui teste contrats, handoffs, incertitude et sécurité sans accès externe.

## Risques
| Risque | Niveau | Traitement |
|---|---:|---|
| sources SEO primaires non récupérées | élevé | aucune recommandation externe promue ; sources marquées bloquées |
| confusion fixture/réel | élevé | label `FIXTURE` obligatoire |
| standardisation prématurée | moyen | états expérimentaux + gates de promotion |
| recommandation à effet de production | élevé | validation humaine explicite obligatoire |

## Décisions ouvertes
| ID | Décision | Proposition | Statut |
|---|---|---|---|
| OD-01 | accès à des sources publiques potentiellement facturables | aucun accès sans autorisation précise et actuelle de Vincent | Vincent requis |
| OD-02 | steward humain et suppléant du pôle | nommer avant toute promotion commune | Vincent requis |
| OD-03 | site/dataset réel avec droits documentés | attendre succès fixture et permission | en attente |
| OD-04 | seuils de promotion après deux cas réels | conserver les seuils expérimentaux puis réviser | expérimental |
