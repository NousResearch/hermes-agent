---
name: de-ahnenforschung-geschichte
description: Use when du die Kirchenbuch‑Archive für die Geburtsorte deiner Personen wissen möchtest. Mappt Orte auf zuständige kath./ev. Archive und leitet Konfession nur als Hinweis ab.
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
    - Gemeinde
    - Forschung
    - Genealogy
    related_skills:
    - genealogy-shared
---

# genealogy-gemeinde-forschung

EBENE 2. Nimmt die Geburtsorte aller Personen in working.sqlite, clustert sie nach
Region, mappt jede Region auf die zustaendigen Kirchenbuch-Archive (kath./ev.).
Konfession wird aus Regional-Standard ABGELEITET (nicht aus Primärquelle), klar markiert.

## Wann laden
Wenn der User "Gemeindebücher", "wo sind die KB", "welche Archive", "Konfession",
"Religion der Ahnen" fragt. Oder als Research-Track neben Pool-Matching.

## KRITISCHE REGEL
- religion-Spalte in DB ist bei ALLEN Datensaetzen LEER (Stand 2026-07-25, 348 Personen).
- Konfession NIEMALS blind aus Region raten und in DB schreiben.
- Ableitung nur als HINWEIS im Report (gray/INFO), nicht als fact.
- Pro Person Einzelrecherche (Diözesan-/Landeskirchen-Übersicht) → dann belegen.

## REGIONEN + ARCHIVE (Kurzform, Vollversion: docs/gemeinde_forschung.md)
1. Bayern (Franken/Oberpfalz): LAELKB Nürnberg (ev.), Matricula Bistum Regensburg/
   Eichstätt/Bamberg (kath.), Archion
2. Hessen/Rheinhessen/Pfalz (GRÖSSTE GRUPPE ~46): EKHN Zentralarchiv Darmstadt (ev.),
   Dom- u. Diözesanarchiv Mainz (kath.), Hessisches Landesarchiv Wiesbaden
3. Elsaß-Lothringen (~24): Archion (ev.), Matricula/FamilySearch (kath.), ahnen-forscher
4. Luxemburg (~21, kath.): Matricula Online LU
5. USA Diaspora (~42): FamilySearch, Ancestry, Find-a-Grave
6. Wolgadeutsche Russland (~7): Archiv Saratow (Engels), FamilySearch Wiki, BKDR
7. Baden (~3): Matricula Erzbistum Freiburg, Badische Landeskirche
8. Preußen/Schlesien (~1): Matricula Polen, FamilySearch Oppeln

## METHODIK
1. birth_place normalisieren (Bundesland/Region extrahieren).
2. Cluster zählen (siehe execute_code unten).
3. Pro Cluster Archiv zuordnen (Format: kath.=X, ev.=Y).
4. "Germany" ohne Ort (61 Personen) = NICHT zuordenbar → FS-Detail öffnen, Ort holen.
5. Report schreiben nach docs/gemeinde_forschung.md.

## CODE-SNIPPET (Region-Cluster)
```python
import sqlite3
from collections import defaultdict, Counter
DB=r"D:\\Ahnenforschung\\db\\working.sqlite"
c=sqlite3.connect(DB)
# region() Funktion wie in Session 2026-07-25 (siehe docs/gemeinde_forschung.md)
rows=c.execute("SELECT birth_place,COUNT(*) FROM person WHERE birth_place!='' GROUP BY 1").fetchall()
reg=defaultdict(lambda: defaultdict(int))
for bp,n in rows: reg[region(bp)][bp]+=n
for r,places in sorted(reg.items(), key=lambda x:-sum(x[1].values())):
    print(r, sum(places.values()))
```

## VERIFIKATION
- Region-Count stimmt mit DB (348 Personen, ~280 pool + ~49 kern + kb-19).
- Archive-URLs sind die recherchierten (session 2026-07-25, Web-Search bestätigt).
- Konfession IMMER als Ableitung markieren, nie als belegt.

## PITFALLS
- "Germany" (61×) ist wertlos für Ortsrecherche → FS-Detailseiten nach Ort durchsuchen.
- Elsaß-Lothringen: ev. KB oft bei FamilySearch/Archion, kath. bei Matricula/FR.
- Wolgadeutsche: KB oft in Russland verstreut, Digitalisate rar → FS + lokale Vereine.