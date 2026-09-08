# Orchestration transversale

## Document control
- **Scope:** Global
- **Owner:** Orchestrateur du support-pole
- **Maturity:** Experimental — v0.1
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Après incident de handoff
- **Linked documents:** AGENTS.md, QUALITY-GATES.md, SECURITY-AND-PERMISSIONS.md
- **Evidence requirement:** Handoffs, état et escalade
- **Exit / validation criteria:** Chaque livrable a son owner suivant
- **Risks / limits / escalation:** Boucle, conflit de rôle, état silencieux : escalade
- **Change history:** CHANGELOG.md
- **Exclusive responsibility:** Décrire la coordination transversale et les handoffs communs.

## Flux
`intake → evidence-ready → analysis-ready → decision-ready → QA → critique → accepted | rework | blocked | needs-human`.

L'orchestrateur vérifie états et contrats ; il ne réécrit pas un jugement métier. Le producteur n'approuve pas seul un livrable à risque. Un blocage nomme manque, impact, rôle habilité et question exacte. Les conflits conservent les lectures concurrentes et leurs preuves.

## Escalade obligatoire
Accès/secret, dépense, effet externe, production, base de preuve insuffisante pour une recommandation majeure, conflit d'autorité ou risque élevé.
