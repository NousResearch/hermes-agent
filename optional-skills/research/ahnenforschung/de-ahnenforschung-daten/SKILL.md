---
name: de-ahnenforschung-daten
description: Use when du rohe Scans, Urkunden oder GEDCOM‑Dateien hast und sie in das normale SQLite‑Schema überführen willst. Erfasst und normalisiert Daten, schreibt nur in working.sqlite.
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
    - Data
    - Capture
    - Genealogy
    related_skills:
    - genealogy-shared
---

# genealogy-data-capture

EBENE 1 des Ahnenforschungs-Skillsets. Nimmt rohe Daten auf und normalisiert sie in das
gemeinsame SQLite-Schema (siehe genealogy-shared/schema.sql). Schreibt NUR in die DB, nie
selbststaendig in einen Baum/Export.

## Wann laden
Beim Anlegen neuer Personen/Quellen, beim Import eines GEDCOM (Gramps/FamilySearch),
beim Erfassen von Urkunden-Scans, wenn Datum/Ort normalisiert werden muessen.

## Datenmodell (Spine)
- Jeder Fakt traegt Pflicht-Feld `confidence`: belegt | unbestaetigt | Vermutung
- Quelle je Fakt: Typ, Signatur/Archiv-Ref, URL/Link zum Scan, Erhoben-Datum

## Schritte
1. DB initialisieren (einmalig): `python genealogy/genealogy-shared/scripts/init_db.py`
2. Scan/Urkunde nach `D:\\Ahnenforschung\\docs\\` legen, Dateiname = Signatur.
3. person/family/source/fact in DB schreiben (Skript scripts/capture.py oder direkt SQL).
4. Quellenlink auf docs/-Datei oder externen Record setzen.
5. confidence je Fakt setzen - ohne Quelle = 'Vermutung'.

## Normalisierung (Pflicht)
- Datum -> ISO 8601 (YYYY-MM-DD, unbekannt: YYYY bzw. YYYY-MM)
- Ort   -> "Ort, Region, Land" (z.B. Schnaittach, Bayern, DE)
- sex   -> M | F | U
- Kein Raten: unbekannte Felder LEER lassen, nicht '?'.

## Skripte
- scripts/capture.py (Person/Quelle einfuegen, Normalisierung)
- genealogy/genealogy-shared/scripts/init_db.py (DB anlegen)

## Verifikation
- `SELECT COUNT(*) FROM person;` zeigt eingetragene Personen.
- `SELECT id, given, surname, confidence FROM person LIMIT 5;` Stichprobe.

## Pitfalls
- Keine Personendaten Lebender in Cloud/LLM - alles lokal auf D:.
- GEDCOM-Import: Nur Fakten mit Quelle uebernehmen, Rest als 'Vermutung' markieren.
- Nicht zwei Personen mit gleicher ID anlegen.