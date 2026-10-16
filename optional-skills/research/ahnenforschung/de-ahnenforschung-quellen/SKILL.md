---
name: de-ahnenforschung-quellen
description: Use when du wissen musst, welche Quellen für eine bestimmte Zeit/Region zugänglich sind und welche Sperrfristen gelten. Liefert DE‑spezifische Quellenlogik und Such‑Strategie.
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
    - Source
    - Research
    - Genealogy
    related_skills:
    - genealogy-shared
---

# genealogy-source-research

EBENE 2. Gibt die Recherche-Richtung vor: welche Quelle fuer welche Zeit/Region, wo die
rechtlichen Grenzen sind, wie man ToS-konform sucht. Treibt genealogy-agent-search an.

## Wann laden
Vor jeder Suche: welche Archive/Register fuer die Zielperson zustaendig, welche Frist gilt,
wie man vorgeht (Standesamt-Antrag vs. Kirchenbuch-Scan).

## DE-Sperrfristen (BGB)
- Geburtenregister: 110 Jahre
- Eheregister:    80 Jahre
- Sterberegister:  30 Jahre
- Danach: frei einsehbar beim Standesamt. Davor: nur mit berechtigtem Interesse
  (Abstammung) + eigene Geburtsurkunde.

## Quellen nach Epoche
- ab 1874/76 (Preussen): Zivilehe Pflicht -> Standesamt + Kirchenbuch parallel
- vor 1874: FAST NUR Kirchenbuecher (Taufe/Trauung/Begraebnis ab ~1550 regional)
- Ev: Archion (archion.de, kostenpflichtig), Matricula (kath./teils ev, gratis)
- Kath: Bistumsarchive, Matricula
- Frei sofort: FamilySearch, CompGen (OFBs/Wiki), Geneanet

## Such-Strategie
1. Bekannte Person + Datum + Ort -> Standesamt/KB der Gemeinde
2. Region+Zeit -> OFB (Ortsfamilienbuch) auf CompGen pruefen
3. ueberregionale Luecken -> FamilySearch full-text + Matricula
4. Archion nur gezielt, einzeln (kein Bulk)

## ToS-Regeln (harter Stopp)
- Archion: KEIN Scraping/Bulk-Download (Vertrag)
- Matricula: keine Massen-Downloads
- FamilySearch API: non-commercial, persoenliche Nutzung, Rate-Limits beachten

## Skripte
- scripts/lookup_plan.py (gibt zu Region/Zeit die zustaendigen Quellen + Frist aus)

## UNTER-SKILL: genealogy-churchbook-portals
Portal-Praxis (Matricula/Archion/FamilySearch) ist ausgelagert nach
genealogy-churchbook-portals (konfessionelles Routing, Arcanum-JS-Blocker,
FamilySearch-WAF, PowerShell-Extract für Pfarrei-Listen). Bei konkretem
Portal-Zugriff DIESEN laden, nicht hier neu erfinden.

## Verifikation
- lookup_plan.py liefert fuer "Bayern 1840" -> Kirchenbuch+Archion+Matricula, kein Standesamt (Frist).

## Pitfalls
- Nicht von falschen Eltern ausgehen: erst Vater+Mutter+Taufort belegen.
- Standesamt-Antrag braucht oft Wohnort-Nachweis des Antragstellers.