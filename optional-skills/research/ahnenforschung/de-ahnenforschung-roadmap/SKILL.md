---
name: de-ahnenforschung-roadmap
description: Use when du nicht weißt, welcher genealogy‑Skill für eine konkrete Aufgabe zuständig ist. Gibt den zu ladenden Skill (EBENE 0‑4 + Unter‑Skills) zurück.
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
    - Roadmap
    - Genealogy
    related_skills:
    - genealogy-shared
---

# genealogy-roadmap (Meta-Router)

Bei jeder genealogy-Aufgabe hier nachschlagen: WELCHER Skill greift? Vermeidet, dass
veraltete/redundante Skills geladen werden. Konsolidiert 2026-07-25.

## ROUTING-TABELLE

| Aufgabe | Skill (EBENE) | Unter-Skill bei Bedarf |
|---------|---------------|------------------------|
| Kirchenbuch lesen/deuten (Latein, Status) | genealogy-kirchenbuch-deutung (0) | — |
| Scan transkribieren (Ollama lokal) | genealogy-transcription (1) | transkribus-api-setup |
| GEDCOM importieren, normalisieren, Daten erfassen | genealogy-data-capture (1) | — |
| "Wo sind die Kirchenbuch-Archive?" | genealogy-gemeinde-forschung (2) | genealogy-source-research |
| Quellenlogik, Sperrfristen, Archion/Matricula | genealogy-source-research (2) | genealogy-churchbook-portals |
| Nachnamen weltweit / Herkunft / Bedeutung | genealogy-name-research (2) | genealogy-name-distribution (VERALTET) |
| Autonomer FS/OFB-Abgleich, Kandidaten suchen | genealogy-agent-search (2) | genealogy-pool-ingest |
| GEDCOM 7.0 Export, Review-Report, Backup | genealogy-tree-archive (3) | — |
| Stammbaum, Timeline, Geo-Map, Dashboard | genealogy-view (4) | — |
| Religion/Konfession/Orte aus Original | genealogy-shared (GRUNDSATZ) + Original-GEDCOM parsen | — |
| Technik: GEDCOM-reparse, camelCase, OBJE, SVG, Playwright | genealogy-data-pipeline (Klassen) | — |
| Quellenprüfung, GPS-5, QUAY, Widerspruch | genealogy-verification (QUERSCHNITT) | — |

## REGELN
1. Immer genealogy-shared laden, wenn ein anderer genealogy-Skill greift (Schema/confidence/review_queue).
2. Original-GEDCOM vor DB (siehe shared-Grundsatz).
3. Veraltet: genealogy-name-distribution (→ name-research), genealogy-original-gedcom-workflow (→ shared).
4. Unter-Skills nur bei konkretem Bedarf laden, nicht pauschal.

## SPIELFOLGE (typischer Forschungslauf)
capture(1) → transcription(1) → source-research(2)+churchbook-portals →
agent-search(2)+pool-ingest → review_queue (HITL) → data-capture(1) übernimmt →
tree-archive(3) export → view(4) darstellen.
Kirchenbuch-Deutung(0) sitzt vor transcription.