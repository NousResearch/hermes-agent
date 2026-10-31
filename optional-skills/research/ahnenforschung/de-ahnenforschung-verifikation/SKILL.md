---
name: de-ahnenforschung-verifikation
description: Use when du deine newly akzeptierten Personen sowohl technisch (Datenbank‑Integrität, DSGVO‑Leck) als auch fachlich (GPS‑5, QUAY‑Bewertung, Widerspruchs‑Erkennung) prüfen willst – bevor sie in den Baum oder Export gehen.
category: research
version: 1.0.0
author: Schrauberhirn (NousResearch Discord), Hermes Agent
license: MIT
platforms:
- windows
- linux
- macos
metadata:
  hermes:
    tags:
    - Verification
    - Genealogy
    related_skills:
    - genealogy-shared
---

# genealogy-verification

QUERSCHNITT‑Skill. Prüft deine newly akzeptierten Personen sowohl technisch (Datenbank‑Integrität, DSGVO‑Leck) als auch fachlich (GPS‑5, QUAY‑Bewertung, Widerspruchs‑Erkennung) – bevor sie in den Baum oder Export gehen.

## Wann laden
Nach jedem erfolgreichen Capture‑Schritt, nachdem neue Facts in die DB geschrieben wurden, bevor sie in den Baum (tree-archive) oder Export (gedcom_export) gehen.

## Technische Prüfung (Datenbank‑Integrität)
- `PRAGMA foreign_keys=ON;` – sicherstellen, dass alle CHIL‑ und FAM‑Links gültige Person‑IDs referenzieren.
- `SELECT COUNT(*) FROM person WHERE confidence NOT IN ('belegt','unbestaetigt','Vermutung');` – ungültige confidence‑Werte erkennen.
- DSGVO‑Leck‑Check: Keine Person mit `birth_date < heute - 100 Jahre` darf `address` oder `phone` enthalten (falls solche Felder existieren).

## Fachliche Prüfung (GPS‑5, QUAY‑Bewertung, Widerspruch)
- **GPS‑5** (Genealogical Proof Standard):  
  1. Gründliche Recherche  
  2. Vollständige Quellenangaben  
  3. Analyse und Korrelation  
  4. Widerspruchslösung  
  5. Schlussfolgerung bei schriftlicher Beweislage  
  Der Skill stellt sicher, dass jedes neue Fact mindestens zwei unabhängige Quellen hat (oder eine Originalquelle mit hoher Qualität) und dass keine widersprüchlichen Facts unbehandelt bleiben.
- **QUAY‑Bewertung** (Qualität der Quellen):  
  Jede Quelle erhält ein Score von 0–3 nach Herkunft (Originalregister → 3, indexierte Transkription → 2, sekundäre Quelle → 1, unsichere Internetquelle → 0). Der Skill warnt, wenn ein Fact nur aus Quellen mit Score ≤1 stammt.
- **Widerspruchs‑Erkennung**:  
  Bei neu hinzugefügten Facts wird geprüft, ob es bestehende Facts mit gleicher Person, Typ und Datum aber unterschiedlichem Wert gibt. Bei Konflikt wird das Fact in die `review_queue` gestellt und der Benutzer aufgefordert, zu entscheiden.

## Skripte
- scripts/verify_integrity.py (Datenbank‑Checks + GPS‑5)
- scripts/verify_sources.py (QUAY‑Bewertung)
- scripts/verify_conflicts.py (Widerspruchs‑Erkennung)

## Verifikation
- Nach jedem Lauf sollte `review_queue` nur Facts enthalten, die noch keiner Entscheidung unterliegen.
- Die DB sollte keine Duplikate bei `(person_id, fact_type, fact_value)` aufweisen.

## Pitfalls
- Über‑Vertrauen in automatisierte Quellen: Auch hochqualifizierte Transkriptionen können Fehler enthalten – immer menschliche Zweitprüfung bei umstrittenen Facts.
- Bei sehr alten Akten (vor 1500) sind Kirchenbuchlücken normal; das Skill setzt keine Facts, wenn die Quelle fehlt oder mehrdeutig ist.