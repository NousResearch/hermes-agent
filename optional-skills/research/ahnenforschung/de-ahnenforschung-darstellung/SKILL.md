---
name: de-ahnenforschung-darstellung
description: "DE-Darstellung: Stammbaum, Timeline, Geo-Map, Konfidenz-Dashboard für DE-Ahnenforschung."
category: research
version: 1.0.0
author: Schrauberhirn
license: MIT
platforms:
- windows
---

# DE-Ahnenforschung — Darstellung (EBENE 4)

## Interactive Family Tree Dashboard

### Datei: `D:\\Ahnenforschung\\tree_de.html`

Das Dashboard ist eine **Single-File HTML-Anwendung** — kein Server nötig.

### Tabs

| Tab | Inhalt |
|---|---|
| 📊 Übersicht | Dashboard, Stat-Karten, Charts, „Zufälliger Ahnen-Spotlight“ |
| 🌲 Stammbaum | Visueller Stammbaum mit Zoom und Surname-Filter |
| 👥 Personen | Listenansicht mit Live-Suche, Geschlechtsfilter |
| 📅 Zeitstrahl | Chronologische Events (Geburt, Tod, Heirat) mit Sortierung |
| 🔔 Tages-Hinweise | „An diesem Tag“ — Geburtstage, Todesfälle, Hochzeiten |

### Features

1. **Zurück zum Dashboard**: Klick auf `🏛️ Family Explorer` Header
2. **Klickbare Stat-Karten**: Alle Karten (Personen, Familien, Orte, etc.) öffnen Listen/Sidebars
3. **Sidebar-Navigation**: Personen/Klick öffnet Sidebar (nicht Modal) in tree, people, timeline Tabs
4. **Two-Column-Tree-Navigation**: Surname-Buttons mit Live-Filter + Zählern
5. **Sortierung**: Nachname → Vorname alphabetisch (localeCompare, sensitivity: 'base')
6. **Verifizierungs-Kenner**:  
   - 🟢 Grün (✓): verifiziert (`confidence = 'belegt'`)  
   - 🟡 Gelb (?): nicht verifiziert (`confidence != 'belegt'`)
7. **Zeitstrahl-Sortierung**: Auf-/Absteigend umschalten via Button
8. **Todesdatum in Timeline**: Bei Geburts-Events mit Todesdatum

### Deutsche Übersetzungen

| Englisch | Deutsch |
|---|---|
| males · females | männlich · weiblich |
| Born/Died/Married | Geboren/Gestorben/Geheiratet |
| Parents/Spouse/Children | Eltern/Ehepartner/Kinder |
| All | Alle |
| Most Children | Meist Kinder |

### Datenmodell-Hinweis
19 Personen (5,5%) haben leeren Nachnamen — typisches 17. Jahrhundert Kirchenbuch-Muster.
Diese erscheinen in der Sortierung am Ende.

## Related

[[de-ahnenforschung-roadmap]]
[[de-ahnenforschung-daten]]