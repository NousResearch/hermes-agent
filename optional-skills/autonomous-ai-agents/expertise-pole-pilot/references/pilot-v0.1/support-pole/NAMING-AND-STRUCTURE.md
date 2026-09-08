# Nommage et structure

## Document control
- **Scope:** Global
- **Owner:** Curateur documentaire
- **Maturity:** Experimental — v0.1
- **Last reviewed:** 2026-09-08
- **Freshness rule:** Semestrielle ou avant un nouveau domaine
- **Linked documents:** DOCUMENT-TAXONOMY.md, templates/
- **Evidence requirement:** Chemin stable, ID unique et liens relatifs
- **Exit / validation criteria:** Convention respectée ou exception décidée
- **Risks / limits / escalation:** Rigidité prématurée : exception tracée
- **Change history:** CHANGELOG.md
- **Exclusive responsibility:** Normaliser les noms, emplacements et identifiants.

## Arborescence
```text
support-pole/             règles et templates communs
domains/<slug>/           pack d'un domaine
  agents/ skills/ workflows/ templates/ evaluations/
reports/                  preuves transversales
evaluations/ + tests/     contrôles locaux
```

## Conventions
- minuscules-kebab-case, sauf les neuf documents canoniques en majuscules ;
- IDs : `<DOMAIN>-Cnn`, `H-<domain>-nn`, `OD-nn`, `SRC-nn`, `CASE-nn`, `F-nn` ;
- liens relatifs seulement ;
- profils `agents/<role>.md`, skills `skills/<skill>/SKILL.md`, workflows `workflows/<workflow>.md`.
