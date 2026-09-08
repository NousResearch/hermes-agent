# Évaluation du pilote SEO

## Document control
- **Scope:** Domaine : SEO
- **Owner:** Responsable évaluation SEO
- **Maturity:** Experimental — SEO-only
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Changement contrat/skill/workflow
- **Linked documents:** evaluations/, CONTRACTS.md, SOURCES.md
- **Evidence requirement:** Cas, rubrique, résultat de régression
- **Exit / validation criteria:** Six scénarios + baseline structurelle exécutés
- **Risks / limits / escalation:** Fixtures ne prédisent aucun effet réel
- **Change history:** README.md
- **Exclusive responsibility:** Définir les scénarios, rubriques et seuils SEO.

## Rubrique (0–2)
| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Couverture | capacité absente | partielle | SEO-C01…SEO-C06 liée/owner |
| Exactitude | fait fabriqué | limite partielle | entrée suivie et limite déclarée |
| Traçabilité | provenance absente | lien incomplet | entrée→sortie reproductible |
| Priorisation | arbitraire | facteur implicite | risque/preuve/dépendance/valeur |
| Contrats | I/O manque | lacune | complet |
| Sécurité | action interdite | approbation vague | action bloquée/explicite |
| Handoffs | owner absent | statut vague | état, owner, décision |
| Utilité | résumé | option vague | prochaine décision claire |

Pass cas : aucun 0 sur exactitude/traçabilité/contrats/sécurité, total ≥12/16. Promotion SEO-only : 6 cas, erreur ré-exécutée, critique clôturée. Promotion commune : deux domaines exigés.

Cas : CASE-01 nominal ; CASE-02 incomplet ; CASE-03 contradictoire ; CASE-04 risque ; CASE-05 source obsolète ; CASE-06 erreur injectée.

**Couverture explicite :** SEO-C01, SEO-C02, SEO-C03, SEO-C04, SEO-C05 et SEO-C06 sont testées par la combinaison des six cas et de la revue critique.
