---
name: de-ahnenforschung-export
description: Use when du akzeptierte Kandidaten aus der review_queue als GEDCOM 7.0 exportieren und den täglichen Review‑Report versenden möchtest. Sichert außerdem DSGVO‑konform und legt Backups an.
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
    - Tree
    - Archive
    - Genealogy
    related_skills:
    - genealogy-shared
---

# genealogy-tree-archive

EBENE 3. Das Ablage- und Export-Element. Nimmt akzeptierte review_queue-Eintraege, schreibt
sie als belegten Fakt in die DB und exportiert den Kanon als GEDCOM. Sendet den Report.

## Wann laden
Beim Exportieren des Baums (Gramps), beim Erzeugen des taeglichen Agent-Reports, beim
Einbinden in das lokale Backup-Konzept (lab-backup).

## GEDCOM 7.0 (Kanon)
- Export nach D:\\Ahnenforschung\\gedcom\\families.ged
- Jede INDI/FAM traegt SOUR (Quelle) + confidence als NOTE/_CONF
- Gramps-kompatibel (Import/Export getestet)

## Ablage (D:\\Ahnenforschung\\) 
- gedcom/families.ged    Kanon
- db/working.sqlite      Arbeits-DB
- docs/                  Scans/Urkunden (referenziert, nicht als Blob)
- reports/               Agent-Reports (Review-Queue)
- cache/                 Transkriptions-Cache

## DSGVO
- Verstorbene: unkritisch
- Lebende: geschuetzt -> KEINE Cloud-Sync, KEINE Personendaten an externe LLM
- Transkription lokal via Ollama (genealogy-transcription)
- Backup: in D:\\backups einbinden (lab-backup), verschluesselt

## Report
- scripts/report.py liest review_queue (pending) + confidence-Verteilung
- Ausgabe an User (Discord/Telegram via Hermes deliver)

## Skripte
- scripts/gedcom_export.py (DB -> families.ged)
- scripts/report.py (Review-Queue + Konfidenz -> Report)

## Verifikation
- gedcom_export.py -> families.ged valid (Gramps laedt ohne Fehler)
- report.py -> Liste der offenen Kandidaten

## Pitfalls
- GEDCOM ohne Quelle = wertlos: jedes FACT braucht SOUR.
- Lebende nicht exportieren/publizieren.
- Backup vor jedem GEDCOM-Re-Export (db/ ist Single Source of Truth).