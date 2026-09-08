# Orchestration du pilote SEO

## Document control
- **Scope:** Domaine : SEO
- **Owner:** Orchestrateur SEO
- **Maturity:** Experimental — SEO-only
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Après cas, handoff raté ou changement rôle
- **Linked documents:** workflows/evidence-to-action.md, CONTRACTS.md, EVALUATION.md
- **Evidence requirement:** Journal d'état, handoffs, issue QA/critique
- **Exit / validation criteria:** Transition = contrat + owner
- **Risks / limits / escalation:** État ambigu/risque : blocked ou needs-human
- **Change history:** README.md
- **Exclusive responsibility:** Orchestrer le flux SEO, ses états et ses décisions.

## Trigger
Demande diagnostic SEO avec données explicitement fournies ou `CASE-nn` fixture.

## Output
Bundle de preuves, constats, plan non exécutif, QA, critique, décision d'état.

## States and failure paths
`intake → evidence-ready → analysed → prioritised → qa-ready → criticised → accepted` ; échecs `blocked`, `conflicted`, `needs-human`, `rework`, `quarantined`. Un état d'échec ne devient pas accepted sans handoff nouveau.

## Handoffs
1. orchestrateur → technique/analyste : entrées + SEO-C01…SEO-C04 ;
2. experts → stratège : preuves/confiance/conflits ;
3. stratège → QA : plan/score/owner/approbation ;
4. QA → critique : paquet et rubriques sans verdict imposé ;
5. critique → orchestrateur/Vincent : défauts/décision.

Détail : `workflows/evidence-to-action.md`.
