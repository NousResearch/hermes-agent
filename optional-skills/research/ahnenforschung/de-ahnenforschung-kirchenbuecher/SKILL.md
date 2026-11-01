---
name: de-ahnenforschung-kirchenbuecher
description: Use when 
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
    - Churchbook
    - Portals
    - Genealogy
    related_skills:
    - genealogy-shared
---

# genealogy-churchbook-portals

EBENE 2. Ergänzt `genealogy-source-research` um die **praktische Portal-Ebene**:
welches Portal für welche Konfession, wie man rankommt (und wo es technisch
blockt), wie man Pfarrei-Listen maschinell extrahiert.

## Wann laden
Bei jeder Kirchenbuch-Suche nach einer konkreten Pfarrei / einem Namen in DE.
Vorher klären: katholisch oder evangelisch? (bestimmt das Portal).

## Konfessionelles Routing (Franken, übertragbar)
- **Katholisch** → matricula-online.eu, Diözese Bamberg oder Eichstätt.
  - Pfade: `/de/deutschland/bamberg/` , `/de/deutschland/eichstaett/`
  - ACHTUNG: `eichstaett-bistum` (mit -bistum) → 404/503. Nur `eichstaett`.
  - Index-Bände (alphabetische Namensregister) = schnellster Einstieg:
    `M11/73` Taufen-Index, `M11/74` Ehen-Index, `M11/75` Sterben-Index.
- **Evangelisch-lutherisch** → archion.de (ELKB) und/oder FamilySearch
  Collection "Germany, Lutheran Baptisms, Marriages, and Burials, 1500-1971"
  (Coll-ID **3015626**).

## Technische Blocker (wichtig, sonst Zeitverschwendung)
1. **matricula = Arcanum-JS-Viewer — ABER Bilder direkt erreichbar (NEU 2026-07-26).**
   - Der Katalog-HTML (curl/web_extract) liefert KEINE Bild-URL und die matricula-`?pg=NN`
     Direkt-URLs → HTTP 404 (Arcanum blockt serverseitig). Das bleibt wahr.
   - **ABER:** Der zugrundeliegende Bildserver `hosted-images.matricula-online.eu`
     ist OHNE Login/CSRF direkt erreichbar (HTTP 200). Der Viewer lädt darüber.
   - **URL-Struktur (reverse-engineered, Schnaittach St. Kunigund, Diözese Bamberg):**
     `http://hosted-images.matricula-online.eu/images/matricula/DE-AEB/images/AEB_Schnaittach/{num:04d}_Schnaittach_Bd.{X}_{K}/{num:04d}_Schnaittach_Bd.{X}_{K}_{BLATT}.jpg`
     - `DE-AEB` = Diözese Bamberg. `AEB_Schnaittach` = Pfarrei.
     - `X` = Band-Präfix (M1, M2, ... M11). `K` = Kapitel innerhalb des Bandes.
     - `num` = fortlaufende Pfarrei-Bandnummer = `BASIS[X] + K` (SCHNAITTACH-SPEZIFISCH!)
     - `{BLATT}` = Blattnummer `0000`–`0012` (404 = Ende).
     - Beispiel (Johann @I39@ *1875, M11/73 K20 Blatt 0004):
       `.../0293_Schnaittach_Bd.11_20/0293_Schnaittach_Bd.11_20_0004.jpg` (num=0273+20=0293)
   - **BASIS-Map (Schnaittach, extrahiert 2026-07-26):** M1=0, M2=18, M3=38, M4=57,
     M5=81, M11=273. M6–M10 via Auto-Search im Skript (Bereich zwischen Nachbar-BASIS).
     NICHT linear, nicht aus Kapitel ableitbar.
   - **pg ↔ Kapitel ist NICHT 1:1!** `?pg=206` (matricula) → Kapitel 20 (nicht 206).
     Kapitel kommt aus dem Blatt-Link `XX_000N` im DOM (XX = K). Erschließen via
     Browser-Log (`performance.getEntriesByType('resource')` nach `hosted-images`
     filtern, BASE64-decodiert) oder Auto-Search im Skript.
   - **PITFALL — aggressives Probing vermeiden:** Ein 25er-Loop mit SSL-Bypass
     (`ssl.CERT_NONE`) wurde vom USER GE ,BLOCKT. Nutze nur gezielte EINZEL-Requests
     (1–2 pro Band), nie einen breiten Rate-Loop. Besser: Mapping über Browser-Log.
   - **PITFALL — `#register-header` = ALLE Bände, nicht nur M11!** Die Übersichtsseite
     (Anker `#register-header`) listet **75 Matriken** (M1/1 … M??/75), nicht nur
     M11/73-75. M11 deckt nur 1564–1929 der Spätzeit. Frühe Taufen/Heiraten/Sterben
     stecken in M1–M10 (z.B. M5/37-40 Taufen 1698–1758, M6/62, M7/65). Beim Voll-Download
     IMMER die Register-Übersicht auswerten, nicht nur die bekannten Index-Bände.
   - **Ordner-Konvention pro Band-Signatur (User-Korrektur):** Speichern nach
     `M{X}_{Y}/` (z.B. `M11_73/`, `M1_4/`), NICHT nach Registertyp gemischt
     (`M11_73_taufen/`). Dateien: `K{Kapitel}_{Blatt}.jpg` + `_alt_uploads/` für
     manuell hochgeladene `clip_*`-Scans. Pro Zielordner eine `NAMENSKONVENTION.md`
     mit Mapping Kapitel→Band→Registertyp + Matricula-Lizenz (CC BY-NC-ND 2.0).
   - **NEVER extrapolate the num-logic to other towns.** Die `BASIS`-Tabelle
     (num = BASIS[Band] + Kapitel) ist SCHNAITTACH-SPEZIFISCH. Andere Pfarreien
     haben eigene `Bd`-Präfixe und `num`-Basen — diese erst über Browser-Log
     erschließen, nicht annehmen.
   - **Kanonisches Skript + num-Map:** Skill **`matricula-scrape`** (NEU 2026-07-26)
     enthält das re-runable Download-Skript (Resume + Auto-Num-Search) und die
     exakte Schnaittach-BASIS-Map. Diesen Skill für Voll-Downloads laden.
     (Altes `scripts/download_matricula_schnaittach.py` hier ist veraltet —
     nutze matricula-scrape stattdessen.)
   - **Konsequenz:** Bilder können automatisiert heruntergeladen werden (curl/Python),
     nicht nur interaktiv. Danach Kurrent-Transkription via Vision (schmale Prompts)
     oder Transkribus/ScriptLing (siehe references).
   - Browser-Screenshot des Viewers erfasst das Modal oft nicht — nutze den
     hosted-images-Direktdownload statt Screenshot.
2. **familysearch = WAF/Incapsula.**
   - Collection 3015626 ist richtig, aber **ohne Login keine Treffer einsehbar**.
   - curl wie Browser (ohne Account) liefern keine Ergebnisliste.
   - → familysearch-Suche braucht eingeloggten Account im Browser.
3. **archion = Volltext nur für hochgeladene Bände.**
   - Schwarzenbruck-Beispiel: nur Bestattungen 1960+ online, Taufen/Trauungen
     NOCH NICHT digitalisiert. Volltextsuche nach historischen Personen oft erfolglos.
   - Originale ggf. beim Landeskirchlichen Archiv ELKB ([EMAIL]) anfragen.

## Trick: Pfarrei-Liste aus matricula extrahieren
`search_files` schlägt auf der UTF-8-matricula-Seite fehl (Index-Problem).
Stattdessen gecachte markdown (aus web_extract) mit PowerShell parsen:
```powershell
$lines = Get-Content $cacheMd
$longest = $lines | Sort-Object Length -Descending | Select-Object -First 1
$links = [regex]::Matches($longest, '\\[([^\\]]+)\\]\\((https://data\\.matricula-online\\.eu/[^)]+)\\)')
foreach ($m in $links) { if ($m.Groups[1].Value -match 'schwarz|ochen|schnaittach') {
  \"$($m.Groups[1].Value) -> $($m.Groups[2].Value)\" } }
```
Die Ortsliste ist eine einzige sehr lange Zeile im gecachten Markdown.

## Ablauf für neue Franken-Orte
1. Konfession klären.
2. kath: matricula-Diözese via PowerShell-Extract nach Pfarrei durchsuchen.
3. Index-Band (falls vorhanden) als Einstieg nutzen.
4. ev: archion prüfen (was online?), sonst familysearch-Login oder LAELKB-Anfrage.

## Verwandt
- genealogy-source-research (Fristen, OFB, Standesamt) — diese Skill ergänzt nur die Portal-Praxis.

## references/
- references/franken-kirchenbuecher-portale.md — verifizierte Portal-URLs, Signatur-Beispiele, PowerShell-Snippet, archion/familysearch-Status.
- references/matricula_direct_image_url.md — NEU: hosted-images-Backend URL-Muster, pg→Bd-Mapping, Browser-Log-Abruf, PITFALL (kein SSL-Bypass-Loop).
- references/schnaittach-signatures.md — Signatur-Karte Schnaittach St. Kunigund (M11/73 Taufen-Index etc.), reverse-engineered Session 2026-07-25. PITFALL: Direkt-URL `?pg=17` liefert "Not Found" (Arcanum-JS), nur interaktiv blaettern.
- references/schnaittach_matricula_workflow.md — Scan-Ablage (Band-Ordner), idempotentes review_queue-Ingest, USER-OVERRIDE-Regel (pg/Blatt des Users = Wahrheit), verifizierte Heiratsdaten M11/74, DB-Kernfamilie. Session 2026-07-25.

- **SESSION 2026-07-26 LEARNINGS (Automatisierung + Hypothesen-Ebene)**
  - **Taufen = Geburten.** User: matricula nennt sie "Geburten", nicht "Taufen".
    DB/review_queue/hyp_*-Tabellen: Typ-Wert "Taufe" -> "Geburt" umbenennen.
    Berichte entsprechend betiteln.
  - **Hypothesen-Ebene statt person.** User wollte KEINE person-Datensätze
    anlegen (Verwandtschaft unsicher, könnte sich ändern). Stattdessen drei
    eigene Tabellen `hyp_geburst / hyp_sterben / hyp_ehe` (idempotent aus
    review_queue gefüllt, nur gültige JSON, korrupte Forschungs-Notizen
    überspringen). Querverknüpfung erst wenn matricula-belegt.
  - **Quellen-Verknüpfung via `fact`-Tabelle.** Schema: `fact(person_id,\n    type, value, source_id, confidence)`. `source`-Tabelle hat\n    (id,type,title,signature,url,retrieved,confidence). person hat KEINE\n    source-Spalte — Quellen hängen an fact. Neue Quelle: INSERT source,\n    dann INSERT fact mit source_id. Alle als confidence="Vermutung" (Kurrent).
  - **Vollautomatischer Download möglich:** hosted-images-Backend (siehe\n    Block 1 oben). Kein Browser-Scraping nötig. Mapping pg→Bd über Browser-Log.
  - **Transkribus/ScriptLing/omastagebuch** als Kurrent-Hilfsmittel notiert\n    (User-Links). Vision bleibt "Vermutung", nie Fakt.
- **SESSION 2026-07-25 LEARNINGS (matricula Scan-Auswertung)**
  - **KEIN Typ "Steuerbuch".** User-Korrektur: "steuerbuch gibt es nicht, das sind\n    alles kirchenbuecher\". Auch Taler/Gulden-Seiten sind Kirchenbuch-Nebeneintraege
    -> Typ "Verzeichnis".
  - **USER OVERRIDE.** User pg/Blatt-Angabe = Wahrheit, Vision-Kurrent nur Hypothese\n    (confidence=Vermutung). Vision niemals gegen explizite User-Angabe durchsetzen.
  - **Narrow vision prompts** (volle Transkription -> Timeout). Schmal: "Kurz:\n    Registertyp? Jahr? Mehlinger/Meier/Mueller-Zeilen (Nachname,Vorname,Datum)\".
  - **Dateiname:** `clip_20260725_130436_1.png` hat Zeitstempel+Lfd-Nr;
    `int(fn.replace('clip_20260725_','').replace('.png',''))` crasht am '_1'.
    Nutzen `parts=fn[:-4].split('_'); num=int(parts[-1])`.
  - **Schreibvariante:** alt "Mellinger" = oft "Mehlinger". Zusammenfassen.
  - **DB-Kern vs FS-Pool:** ~300 "Mehlinger" in DB sind Auswanderer (nicht
    Schnaittach). child_link-Spalten: family_id, person_id, role.
  - **Konfession in Scan versteckt:** Kopfzeile (Buchtyp+Ort) bei Scans fast
    IMMER abgeschnitten — nur JAHR als große Zahl oben sichtbar. Ort "Schnaittach"
    stand nur auf Bild 60 ("Geburt Schnaittach"). Nicht raten; User fragt gezielt
    nach Überschriften -> schmale Vision-Prompts "Nur Überschrift oben wörtlich?".
  - **ZWEI gleichnamige Conrad-Taufen:** Conrad ~1801 (Scan 62, Familie Conrad\n    u.Maria, 24 Juni) vs Conrad *1806 (pg=169 Blatt 16_0008, Kopfzeile\n    "Geborne anno 1806."). Bild 63 zeigt dieselbe 1806-Seite ("Hübsch Conrad\n    2.Dec" ist ANDERER Nachname, nicht Mehlinger). NIEMALS die beiden Conrad
    mergen. KI liest "1806" oft fälschlich als "Schreibfehler 1866" -> User
    widerspricht, User = Wahrheit.
  - **Chronologie-Methode:** Scans (review_queue) + DB-Kern (+ User pg/Blatt)
    cross-checken. Heirat -~25-35J = Geburtsjahr-Schaetzung; gleiche Vorname in
    aufeinanderfolgenden Jahrzehnten = VERSCHIEDENE Personen (Andreas 1794/1828,
    And.Friedrich 1799/1817/1878). Vollst. verifizierte Kette Wolfgang/Wilhelm
    1757 -> Josef 1948 liegt in references/schnaittach_matricula_workflow.md.