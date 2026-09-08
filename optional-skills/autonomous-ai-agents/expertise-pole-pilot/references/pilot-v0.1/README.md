# Pôle support d’expertises — pilote SEO

Ce répertoire contient un **socle expérimental auditable** et un pilote SEO à fixtures, conçu pour la chaîne : preuve → priorisation → plan non exécutif → QA → critique.

## Points d’entrée

- [`support-pole/README.md`](support-pole/README.md) — gouvernance, taxonomie, cycle de vie, sécurité et templates communs.
- [`domains/seo/README.md`](domains/seo/README.md) — pilote SEO, neuf documents canoniques, profils, skills, workflow et évaluations.
- [`reports/TRACEABILITY-MATRIX.md`](reports/TRACEABILITY-MATRIX.md) — exigence → preuve → statut.
- [`reports/MATURITY-DECISION.md`](reports/MATURITY-DECISION.md) — ce qui n’est **pas** promu et les gates bloquants.
- [`reports/OPEN-DECISIONS.md`](reports/OPEN-DECISIONS.md) — décisions demandant Vincent.
- [`domains/seo/evaluations/runs/RUN-001.md`](domains/seo/evaluations/runs/RUN-001.md) — preuve de régression locale.

## Vérification locale

```bash
python3 -m unittest discover -s tests -p 'test_audit_pack.py' -v
```

La suite ne consulte pas le réseau et ne déclenche aucune action externe. Elle contrôle le contrat structurel du pack ; elle ne prouve ni le comportement de moteurs de recherche ni le gain métier.

## Gates bloquants assumés

La recherche SEO externe et le sous-agent de critique indépendant ont été bloqués par le NO-SPEND GUARD. Aucun résultat de recherche, aucune comparaison externe et aucun avis indépendant n’ont été simulés. Voir `reports/RESEARCH-COMPARISON-STATUS.md` et `reports/INDEPENDENT-REVIEW.md`.

## Note de packaging Hermes

Le pack source nomme les procédures réutilisables `SKILL.md`. Dans cette copie de référence, les trois fichiers sont nommés `SKILL-CONTRACT.md` afin que le générateur de documentation Hermes ne les découvre pas comme des skills installables imbriqués. Leur contenu reste une spécification de skill ; ils sont **not loadable** depuis ce répertoire de référence. Le skill installable est uniquement `optional-skills/autonomous-ai-agents/expertise-pole-pilot/SKILL.md`.
