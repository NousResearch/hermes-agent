---
name: evidence-bundle
description: Assemble a traceable evidence bundle for SEO decisions.
version: 0.1.0
author: Vincent HERON, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  domain: seo
  status: experimental
---

# Evidence bundle

## Trigger
Diagnostic SEO récurrent avec signaux fournis à transformer en preuves révisables. Ne pas utiliser pour opinion isolée ou sans artefact nommé.

## Prerequisites
Cas/requête nommé, périmètre borné, entrées fournies, accès déclaré, contrat `SEO-C01…SEO-C04`. Source non fournie = manquante ; aucun fetch externe sans autorisation humaine actuelle.

## Procedure
1. ID `EV-` + classe `FIXTURE|REAL|UNKNOWN|SENSITIVE` à chaque entrée. **Check:** aucune entrée non classée.
2. Extraire seulement observations littérales avec champs/lignes. **Check:** chaque statement a EV-ID.
3. Étiqueter fact/observation/hypothesis/recommendation/blocked. **Check:** recommandation non déguisée en fait.
4. Enregistrer fraîcheur, lacune, contradiction. **Check:** conflit conservé.
5. Handoff avec confiance/next owner. **Check:** evidence_status complet/partiel/conflicted/missing.

## Pitfalls
Fixture ≠ observation site ; URL écrite ≠ source consultée ; signal unique ≠ causalité/indexation/impact.

## Verification evidence
QA réconcilie chaque finding final à un EV-ID et rejette date/classe/conflit/handoff manquant. `python3 -m unittest tests.test_audit_pack -v` vérifie les marqueurs structurels.

## Do not use when
Tâche unique/vague, crawl/source externe non approuvé, ou changement de production.
