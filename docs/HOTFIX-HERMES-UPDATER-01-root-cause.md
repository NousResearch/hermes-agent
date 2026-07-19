# HOTFIX-HERMES-UPDATER-01 — Root Cause Analysis

**Status:** Analyse abgeschlossen, Patch in Vorbereitung (noch nicht committet/gepusht).
**Betroffener Code:** `hermes_cli/main.py`
**Symptom:** Nach `hermes update` wurde die Web-UI (`web/`) nicht neu gebaut, obwohl sich Quellen/Abhängigkeiten geändert hatten. Die Plattform lieferte veraltetes/inkonsistentes UI aus.

## Auslöser (Phase-5-Reparatur, bereits durchgeführt)

Die folgenden Schritte haben den akuten Defekt behoben:

1. `npm cache verify` — korrupten/brockigen npm-Cache repariert.
2. Veralteten Skip-Cache entfernt — ein zwischen Stationen persistiertes
   Skip-Marker/Sentinel, das den Web-Build fälschlich als "nicht nötig"
   einstufte.
3. Unvollständige `web/node_modules` entfernt — nach abgebrochenem/
   fehlgeschlagenem Install lagen Teil-Artefakte vor, die `_web_ui_build_needed`
   täuschten.
4. `npm ci --workspace web --include=dev` — deterministischer, vollständiger Dependenz-Install.
5. `npm run build -w web` — sauberer Rebuild des Dashboards.
6. Dashboard, Gateway und Plattformen validiert — Lauffähigkeit bestätigt.

## Root Cause

Der Updater selbst besitzt **keine explizite "Web-UI muss neu gebaut werden"-Garantie**
für den Fall, dass sich `web/`-Quellen oder der Workspace-Lockfile geändert haben.

Die einzige Stelle, die den Web-Build überspringt, ist `_web_ui_build_needed()`
(`hermes_cli/main.py`, ~Zeile 4669). Sie entscheidet **ausschließlich über mtime-Vergleich**:

- Sentinel = `hermes_cli/web_dist/.vite/manifest.json`, Fallback `web_dist/index.html`.
- Wenn `web_dist/sentinel.mtime >= max(mtime aller web/ Quelldateien, package*.json,
  vite.config.*, <workspace-root>/package-lock.json)`, wird der Build übersprungen.

Diese Logik reicht in folgenden Situationen nicht aus und führt zum "Skip trotz Änderung":

1. **Git-Checkout verschiebt mtime in die Vergangenheit.** Ein `git pull`/`reset`
   setzt die mtime von Quelldateien oft auf die Checkout-Zeit (oder älter), während
   `web_dist/index.html` eine frische mtime aus einem früheren Build behält.
   Ergebnis: `web_dist` wirkt "neuer" → Build wird übersprungen, obwohl Inhalte neu sind.
   (Dasselbe mtime-Problem war bei `_tui_need_npm_install` bereits bekannt und wurde
   dort auf Content-Vergleich umgestellt — der Web-Pfad wurde nicht nachgezogen.)

2. **Fehlender `.vite/manifest.json`-Sentinel.** Vite schreibt `manifest.json` nur,
   wenn `build.manifest = true` in der Config gesetzt ist. Ist das nicht der Fall,
   fällt `_web_ui_build_needed` auf `web_dist/index.html` zurück. Wenn `index.html`
   durch einen vorherigen (ggf. fehlerhaften) Build existiert, aber keine vollständige
   Übereinstimmung mit den aktuellen Quellen mehr besteht, greift der mtime-Skip erneut.

3. **Nur-Lockfile-Änderung ohne Quell-mtime-Änderung.** Wenn sich
   `<workspace-root>/package-lock.json` ändert (npm-Dependenzen), aber die `web/`-Quellen
   selbst nicht angefasst werden, hängt das Ergebnis allein vom mtime des Lockfiles ab.
   Ein Lockfile-Rewrite durch npm (plattform-spezifisch, non-deterministisch) kann eine
   ältere mtime erzeugen als `web_dist` → Skip, obwohl Dependenz-Baum neu ist.

4. **Keine explizite Force-Invalidierung im Update-Flow.** `_cmd_update_impl`
   (~Zeile 10111) ruft nach dem Pull `node_failures = _update_node_dependencies()`
   und dann `_build_web_ui(PROJECT_ROOT / "web")` auf. `_update_node_dependencies`
   prüft zwar `_npm_lockfile_changed()` (Content-Hash über Lockfile + package.json),
   aber dieser Entscheidungswert wird **nicht** an `_build_web_ui`/`_web_ui_build_needed`
   weitergereicht. Der Web-Build folgt somit einer eigenen, schwächeren mtime-Heuristik.

### Zusammenfassung

Der Defekt ist eine **Heuristik-Lücke**: Der Updater verlässt sich beim Web-UI-Skip auf
mtime, die durch git-Checkout und npm-Rewrites unzuverlässig werden. Es fehlt eine
deterministische (Content-basierte) "Neu bauen erzwingen"-Entscheidung, die an
Lockfile-/Quelländerungen gekoppelt ist — analog zur bereits korrigierten TUI-Logik.

## Geplante Patches (Phase 7)

### Patch 1 — `_web_ui_build_needed` robust machen
- Signatur erhält optionales `force: bool`-Argument.
- Bei `force=True` wird immer `True` geliefert (Bypass der mtime-Heuristik).
- Optional: Content-Hash-Vergleich der `web/`-Quellen gegen einen im
  `web_dist/.web_build_stamp.json` abgelegten Hash (Vorbild: `_desktop_build_needed`
  / `desktop-build-stamp.json`), um den Skip deterministisch statt über mtime zu treffen.

### Patch 2 — Update-Flow koppelt Web-Build an Änderungserkennung
- In `_cmd_update_impl` nach dem Pull: Wenn `_npm_lockfile_changed` (`True`) **oder**
  sich `web/`-Quellen seit dem letzten Web-Build-Hash geändert haben, wird
  `_build_web_ui(..., force=True)` aufgerufen.
- Schreiben eines `web_dist/.web_build_stamp.json` nach erfolgreichem Build
  (Hash der berücksichtigten Quellen/Lockfiles), damit zukünftige Skips korrekt sind.
- Die bestehende `npm ci --workspace web --include=dev` + `npm run build -w web`
  Sequenz aus Phase 5 wird zur kanonischen Update-Route.

## Validierung (lokal, vor PR)

- [x] `python3 -m py_compile hermes_cli/main.py` → SYNTAX_OK
- [x] Unit-Test (`docs/test_hotfix_updater_01.py`, 9 Checks) grün:
  - `needed(force=True)` → immer `True`
  - kein Stamp + frischer dist → liefert `bool` (mtime-Fallthrough)
  - Stamp schreiben → existiert, hat `contentHash` + `builtAt`
  - `needed(force=False)` mit passendem Stamp → `False`
  - **Quelländerung nach Stamp → `True`** (der ursprüngliche Defekt; hier
    gefixt: Stamp-existenz + Hash-Ungleich liefert sofort `True`, kein
    mtime-Fallthrough mehr)
  - neuer Stamp nach Änderung → wieder `False`
  - fehlender dist → `True`
- [x] Ein während des Tests gefundener Logikfehler (Stamp-mismatch fiel auf
  mtime-Heuristik zurück) wurde korrigiert und erneut grün validiert.
- [ ] Noch nicht committet/gepusht (Warte auf separaten Wartungs-PR).
