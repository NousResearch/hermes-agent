# Cadre d'évaluation commun

## Document control
- **Scope:** Global
- **Owner:** Responsable évaluation
- **Maturity:** Experimental — v0.1
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Après nouvelle classe de cas
- **Linked documents:** QUALITY-GATES.md, RESEARCH-PROTOCOL.md
- **Evidence requirement:** Cas, rubriques, sorties et régressions
- **Exit / validation criteria:** Correction, sûreté, utilité et limites démontrables
- **Risks / limits / escalation:** Note subjective seule : définir oracle humain
- **Change history:** CHANGELOG.md
- **Exclusive responsibility:** Définir l'évaluation transversale et la comparaison au non-structuré.

## Dimensions
- **Correction :** contrat respecté, aucune preuve fabriquée.
- **Utilité :** prochaine décision mieux définie qu'une réponse ad hoc.
- **Sûreté :** actions interdites bloquées/escaladées.
- **Traçabilité :** entrée → analyse → sortie → décision retrouvable.
- **Robustesse :** cas incomplet, contradictoire et erreur détectés.

## Baseline
Même corpus, workflow structuré vs réponse libre : capacités couvertes, affirmations sans preuve, recommandations sans owner, contradictions perdues, défauts QA et temps de reprise. Aucun gain n'est affirmé avant baseline exécutée.
