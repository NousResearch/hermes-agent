---
name: de-ahnenforschung-interpretation
description: Use when du ein transkribiertes Kirchenbuchblatt hast und dessen fachliche Interpretation benötigst. Liefert Latein‑Glossar, Status‑Marker, Kurrent‑Hilfen und Mapping zu GEDCOM.
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
    - Kirchenbuch
    - Deutung
    - Genealogy
    related_skills:
    - genealogy-shared
---

# genealogy-kirchenbuch-deutung

EBENE 0. Nimmt ein transkribiertes Kirchenbuch (aus Vision/Ollama/Transkribus) und
DEUTET es fachlich: Welcher Bucher-Typ? Welche Marker = Taufe/Sterben/Mischbuch?
Welche Latein-Begriffe? Wie nach GEDCOM mappe?

## Wann laden
Bei jedem Kirchenbuch-Scan, jeder Transkription, jedem lateinischen Kirchenbuch-Zitat.
Nicht nur bei Mehlinger — alle Pfarreien.

## 1. BUCHTYPEN (Typologie)
- Taufbuch (Taufregister, baptizatus): Kind, Eltern, Paten, Taufdatum, Wohnort
- Traubuch (Copulationsregister, Ehebuch): Brautleute, Eltern, Zeugen, Ort
- Sterbebuch (Totenbuch): Verstorbener, Alter, Todesursache, Begräbnis
- MISCHBUCH: vor ~1750 oft Taufen+STERBEFÄLLE im selben Band (siehe "vixit annis")
- Konfirmationsregister, Kommunikantenlisten: oft nur Namen+Jahr

## 2. STATUS-MARKER (entscheidend)
- infans = Kind/Säugling (Taufe)
- puella = Mädchen, virgo = Jungfrau, adolescens = Jüngling (Taufe/Geschlecht)
- vixit annis X = "lebte X Jahre" → TODESEINTRAG, nicht Taufe!
- sepultus / tumulatus = begraben (Sterbeeintrag)
- suscepit / susceptrix / compater / commater = Pate / Patin
- baptizatus / elutus = getauft
- conjunx / copulatus cum = verheiratet mit (Trauung)
- vidua / viduus = Witwe / Witwer
- illegitimus / ex illegitimo thoro / spurius = unehelich (wichtig für Forschung!)

## 3. LATEIN-GLOSSAR (kompakt, Vollversion siehe references/glossar.md)
Verwandtschaft:
  filius/filia = Sohn/Tochter; parentes = Eltern; conjuges = Eheleute
  patrini = Paten; testes = Zeugen; avus/avia = Großvater/Großmutter
  frater/soror = Bruder/Schwester; avunculus = Muttersbruder (Onkel)
  amita = Vaters Schwester; socer/socrus = Schwiegervater/-mutter
  vitricus = Stiefvater; gener = Schwiegersohn
Beruf/Stand:
  agricola = Bauer; faber ferrarius = Schmied; sutor = Schuster; textor = Weber
  carpentarius = Wagner; colonus = Kolonist/Bauer; auriga = Kutscher
  aedituus = Küster; pastor = Pfarrer; capellanus = Hilfsgeistlicher
Todesursachen (Sterbebuch!):
  apoplexia = Schlaganfall; variolae = Pocken; dysenteria = Ruhr
  febris = Fieber; tabes/tuberculosis = Schwindsucht; asthma = Asthma
  aquis suffocatus = ertrunken; violenta = gewaltsam; subito mortuus = plötzlich
Datierung:
  feria II-VI = Montag-Freitag; ??? bis ???
  Xbris = 7bris/8bris/9bris/10bris = Sep/Okt/Nov/Dez (Kanzlei-Kürzel!)
  eodem die natus et baptizatus = am selben Tag geboren und getauft
  annus et dies obitus = Sterbedatum

## 4. KURRENT / SÜTTERLIN HILFEN
- Transkribus (transkribus.org): KI-HTR, trainiertes "Kurrent (German)"-Modell.
  Free-Tier ~2000 Seiten, dann Abo. Cloud → nur historische Scans (DSGVO-frei).
- readcoop.eu: kommerziell, selbe Engine.
- eScriptorium + Kraken (open-source HTR): eigenes Modell trainieren, läuft lokal/CPU,
  DSGVO-sicher, präzise bei eigenem Buch. Aufwand lohnt sich ab ~50 Seiten.
- Lokal auf LAB (i5-10500, 24GB RAM, KEINE GPU — Intel UHD 630 von Ollama nicht
  genutzt): gemma3:12b-it (oder :27b ~18GB) = bestes lokales Vision-OCR,
  Kurrent bleibt aber fehlerhaft. llama3.2-vision:11b schwächer.
- kurrentschrift-lernen.de: selbst lesen lernen (Mensch > jedes Modell bei Kurrent)
- GENEREELL: Cloud-Vision halluziniert bei Kurrent systematisch (Bsp: 1704 vs 1667
  verwechselt). IMMER Mensch gegenliest. Nie Nachnamen erfinden.

## 5. MAPPING KIRCHE → GEDCOM
- taufe → INDI @Ix@ + CHIL-Link zu Eltern + BIRT.DATE + (BIRT.PLAC)
- eltern → FAM @Fx@ (HUSB/WIFE) + CHIL
- paten → ASSO @Ix@ (TYPE godparent) oder NOTE
- trauung → FAM @Fx@ MARR.DATE/PLAC
- sterbefall → DEAT.DATE + DEAT.CAUSE (NOTE) + BURI.DATE
- unehelich → flag in NOTE "illegitim", Vater ggf. unbekannt

## 6. INTERPRETATIONS-REGELN
1. Vor 1800: Kirchenbücher durchweg LATEIN. Danach Deutsch.
2. Nachname oft nur bei Erstnennung oder gar nicht pro Eintrag → nicht erfinden.
3. "vixit annis" = Toter, nicht Täufling. Mischbuch erkennen!
4. Wiederverheiratung vor 1800: Witwenstatus oft weggelassen → Begräbnisliste prüfen.
5. X-Abkürzung bei Vornamen: X phorus = Christophorus, X ian = Christian,
   X ina = Christina (GenWiki-Regel).
6. Ortsnamen latinisiert: Alba Regia = Stuhlweißenburg, Strigoniensis = Esztergom.

## 7. QUELLEN (für Skill-Search / Nachschlag)
- GenWiki "Lesen von Kirchenbuchdaten": https://genwiki.genealogy.net/Lesen_von_Kirchenbuchdaten
- Dresdner Verein Genealogie "Kirchenbuchlatein" (PDF): https://www.dresdner-verein-fuer-genealogie.de/.../DVG-Tipps-02-Kirchenbuchlatein.pdf
- Rechenberg Historia Latein-Glossar: https://rechenberg-historia.de/lateinische-begriffe-kirchenbuecher/
- Krumhermersdorf Latein: https://www.krumhermersdorf.de/literatur/latein.htm
- Transkribus Kurrent: https://www.transkribus.org/de/kurrentschrift-uebersetzen
- kurrentschrift-lernen.de

## Verifikation
- Transkript gegen Scan durch Menschen prüfen (Pflicht bei Kurrent).
- review_queue-Eintrag mit confidence, erst nach Accept in DB.
- Kein Schreiben in Kernbaum ohne Human-in-the-loop.