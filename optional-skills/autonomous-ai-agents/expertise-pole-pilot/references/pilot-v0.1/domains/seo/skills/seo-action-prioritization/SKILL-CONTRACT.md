---
name: seo-action-prioritization
description: Prioritize verified SEO actions without executing changes.
version: 0.1.0
author: Vincent HERON, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  domain: seo
  status: experimental
---

# SEO action prioritization

## Trigger
Plusieurs findings QA-ready doivent devenir un plan non exécutif répété. Ne pas utiliser pour claims non vérifiés ou autoriser une modification.

## Prerequisites
Bundle vérifié, owner, contraintes métier déclarées ou gap, option réversible, contrat `SEO-C05`.

## Procedure
1. Retourner `partial|conflicted|missing` sauf action collecte explicitement. **Check:** tous candidats QA-ready.
2. Scorer qualitativement gravité, confiance, dépendance/effort, réversibilité, valeur déclarée. **Check:** aucun score n'est prédiction.
3. Rédiger P0/P1/P2 : EV-ID, owner, dépendance, approbation, test, rollback. **Check:** P0 a décision humaine.
4. Inclure collecter preuve/ne rien changer si confiance basse. **Check:** incertitude visible.
5. Handoff QA. **Check:** action traçable au contrat/preuve.

## Pitfalls
Score ≠ prévision ; “quick win” sans preuve interdit. Robots/canonicals/indexing/templates/contenu/budget sont propositions humaines seulement.

## Verification evidence
QA vérifie liens, approbation et rollback ; critique attaque l'option la plus risquée. Audit local vérifie les sections.

## Do not use when
Collecte, observation technique brute, rédaction, tâche unique, ou action externe/production.
