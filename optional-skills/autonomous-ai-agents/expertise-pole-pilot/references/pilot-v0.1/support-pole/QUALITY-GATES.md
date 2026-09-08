# Gates de qualité avant adoption

## Document control
- **Scope:** Global
- **Owner:** Responsable qualité du support-pole
- **Maturity:** Experimental — v0.1
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Avant adoption ou promotion
- **Linked documents:** EVALUATION-FRAMEWORK.md, LIFECYCLE.md
- **Evidence requirement:** Résultats test, revue contradictoire et traçabilité
- **Exit / validation criteria:** Tout gate applicable a un résultat et owner
- **Risks / limits / escalation:** N/A doit être justifié ; risque élevé escaladé
- **Change history:** CHANGELOG.md
- **Exclusive responsibility:** Bloquer l'adoption tant que les gates ne sont pas démontrés.

| Gate | Question | Preuve | Bloque |
|---|---|---|---|
| G1 Taxonomie | responsabilité non dupliquée ? | revue + contrôle | promotion |
| G2 Contrat | I/O/preuves/escalade testables ? | contrat par capacité | exécution |
| G3 Sources | provenance/fraîcheur suffisante ? | registre ou blocage | recommandation forte |
| G4 Sécurité | action nécessite-t-elle humain ? | classe de permission | action externe |
| G5 Exécution | workflow exercé ? | cas + régression | promotion |
| G6 Contradiction | falsification indépendante ? | avis distinct | promotion |
| G7 Utilité | meilleur que non-structuré ? | baseline exécutée | promotion commune |

`blocked` n'est jamais `pass` : il interdit seulement les décisions reposant sur ce gate.
