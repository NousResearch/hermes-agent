# Cycle de vie des artefacts

## Document control
- **Scope:** Global
- **Owner:** Support-pole steward
- **Maturity:** Experimental — v0.1
- **Last reviewed:** 2026-09-08
- **Freshness rule:** À chaque transition
- **Linked documents:** QUALITY-GATES.md, GOVERNANCE.md
- **Evidence requirement:** Gate, critique, décision et version
- **Exit / validation criteria:** État affirmé seulement avec preuve de gate
- **Risks / limits / escalation:** Promotion hâtive ou archive perdue : escalade
- **Change history:** CHANGELOG.md
- **Exclusive responsibility:** Administrer le cycle de vie de chaque artefact.

## États
`proposed → experimental → validated-domain → promoted-common`; alternatives : `deprecated → archived` et `quarantined`.

| Transition | Conditions | Décideur |
|---|---|---|
| proposed → experimental | contrat, owner, risque, test initial | owner domaine |
| experimental → validated-domain | 2 cas, régression verte, critique clôturée | owner + critique |
| validated-domain → promoted-common | preuve sur 2 domaines, aucune dépendance métier cachée | Vincent + steward |
| any → quarantined | risque élevé ou preuve invalide | critique / steward |
| deprecated → archived | migration ou lien de remplacement | owner |

Toute modification indique motif, impact, tests touchés, état avant/après et revue suivante.
