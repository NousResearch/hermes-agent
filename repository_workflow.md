# repository workflow

Stand: 2026-05-02

Dieses Repository ist lokal so konfiguriert, dass der eigene Fork der normale
Arbeits-Remote ist und das Original-Repository nur als Upstream-Quelle dient.

## remotes

```text
origin    git@github.com:dimto13/hermes.git
upstream  https://github.com/NousResearch/hermes-agent.git
```

`origin` ist der eigene Fork. Normale Pulls und Pushes laufen gegen diesen Fork.

`upstream` ist das Original-Repository. Es ist nur fuer Fetch/Pull gedacht. Push
zu `upstream` ist lokal absichtlich deaktiviert:

```text
upstream  DISABLED (push)
```

## status pruefen

```bash
git remote -v
git status --short --branch
git branch -vv
```

## eigene aenderungen in den fork pushen

```bash
git status --short
git add <dateien>
git commit -m "kurze beschreibung"
git push origin main
```

Wenn auf einem Feature-Branch gearbeitet wird:

```bash
git switch -c <branch-name>
git add <dateien>
git commit -m "kurze beschreibung"
git push -u origin <branch-name>
```

## fork vom original aktualisieren

Vorher pruefen, ob lokale Aenderungen offen sind:

```bash
git status --short
```

Upstream holen:

```bash
git fetch upstream
```

Aktuellen Branch mit dem Original-`main` aktualisieren:

```bash
git pull upstream main
```

Danach den eigenen Fork ebenfalls aktualisieren:

```bash
git push origin main
```

## alternative: fork-main explizit setzen

Wenn der lokale `main` exakt auf den aktuellen Stand aus dem Original gebracht
werden soll, zuerst lokale Aenderungen committen oder bewusst stashen. Danach:

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

Keine lokalen Aenderungen verwerfen, solange sie nicht gesichert sind.

## wichtige regel

Nicht direkt gegen `upstream` pushen. Das ist absichtlich blockiert.

Richtig:

```bash
git push origin main
```

Nicht verwenden:

```bash
git push upstream main
```

## lokale besonderheiten

Aktuell gibt es lokale, projektbezogene Aenderungen fuer die CocoIndex/Hermes
Dokumentensuche:

```text
plugins/cocoindex_docs/
skills/cocoindex
```

Ausserdem existiert eine lokale Aenderung in:

```text
ui-tui/package-lock.json
```

Diese Datei wurde nicht im Rahmen der CocoIndex-Integration angepasst und sollte
vor einem Commit separat geprueft werden.
