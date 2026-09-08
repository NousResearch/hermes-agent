# Workflow — evidence-to-action

## Trigger
`CASE-nn` ou demande bornée, données nommées, plan SEO non exécutif demandé.

## Output
Registre preuves, findings, action-plan, QA, critical-review, état final ; chaque pièce porte capability/handoff ID.

## States and failure paths
| State | Owner | Entry | Exit / failure |
|---|---|---|---|
| intake | orchestrateur | demande | blocked si scope/permission absent |
| evidence-ready | technique+preuves | inputs classifiés | conflicted/blocked si preuve incomplet |
| analysed | experts | observations/hypothèses | rework si contrat manque |
| prioritised | stratège | candidats QA-ready | needs-human si risque |
| qa-ready | QA | plan lié | rework si source/approval manque |
| criticised | critique | QA disponible | quarantined si critique persiste |
| accepted | owner | gates pass | jamais exécution production |

## Handoffs
`H-seo-01` orchestrateur→experts ; `H-seo-02` experts→stratège ; `H-seo-03` stratège→QA ; `H-seo-04` QA→critique ; `H-seo-05` critique→orchestrateur/Vincent.

Un handoff sans roles, status, capability, evidence, confidence, risk ou next_owner est rejeté ; needs-human ne devient jamais accepted sans demande humaine résolue.
