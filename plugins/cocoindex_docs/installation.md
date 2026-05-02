# CocoIndex Documents - Reproduzierbare Installation

Stand: 2026-05-02  
Referenzsystem: `station_192_168_178_121`  
Projekt: `/home/tobi/skripte/hermes-agent`

Diese Notiz dokumentiert die konkreten Installations- und Rollout-Erkenntnisse
fuer die Hermes/CocoIndex-Dokumentensuche. Ziel ist, die Einrichtung auf anderen
PCs schneller und reproduzierbar durchfuehren zu koennen.

## Zielbild

```text
/home/tobi/skripte/hermes-agent
  -> Hermes Agent
  -> plugins/cocoindex_docs/
  -> MCP-Server fuer Dokumentensuche

/home/tobi/skripte/cocoindex
  -> CocoIndex Checkout
  -> editable install in der Hermes-venv

/home/tobi/Dokumente
  -> Quelle der zu indexierenden Dokumente

/home/tobi/.local/state/hermes/cocoindex
  -> CocoIndex-State

/home/tobi/.local/share/hermes/cocoindex/documents.sqlite3
  -> lokaler SQLite-Suchindex
```

## Gepruefte Versionen auf dem Referenzsystem

- Ubuntu `22.04.5 LTS`
- Hermes Agent `v0.12.0`
- Hermes Python: `3.11.15`
- Hermes venv: `/home/tobi/skripte/hermes-agent/venv`
- CocoIndex: editable install aus `/home/tobi/skripte/cocoindex`
- CocoIndex-Version: `999.0.0`
- MCP SDK: `1.27.0`
- sentence-transformers: `5.4.1`
- pypdf: `6.10.2`
- sqlite-vec: `0.1.9`
- watchfiles: `1.1.1`

## Systemvoraussetzungen

Minimal:

```bash
sudo apt update
sudo apt install -y git build-essential python3-venv
```

Empfohlen, wenn CocoIndex oder native Python-Pakete aus Source gebaut werden:

```bash
sudo apt install -y pkg-config libssl-dev rustc cargo
```

Optional fuer manuelle Datenbankinspektion:

```bash
sudo apt install -y sqlite3
```

Python muss fuer CocoIndex mindestens `3.11` sein. Auf dem Referenzsystem nutzt
Hermes bereits eine eigene Python-`3.11.15`-venv. Diese venv ist die bevorzugte
Laufzeitumgebung.

## Hermes-venv verwenden

```bash
cd /home/tobi/skripte/hermes-agent
source /home/tobi/skripte/hermes-agent/venv/bin/activate
python --version
```

Erwartet:

```text
Python 3.11.x
```

## CocoIndex installieren

Wenn CocoIndex als lokaler Checkout vorhanden ist:

```bash
cd /home/tobi/skripte/hermes-agent
source /home/tobi/skripte/hermes-agent/venv/bin/activate
python -m pip install -e "/home/tobi/skripte/cocoindex[sentence_transformers,sqlite]"
python -m pip install pypdf mcp
```

Falls `sqlite-vec` nicht durch das CocoIndex-Extra installiert wurde:

```bash
python -m pip install sqlite-vec
```

Pruefung:

```bash
/home/tobi/skripte/hermes-agent/venv/bin/python - <<'PY'
import importlib.util

for module in [
    "cocoindex",
    "mcp",
    "pypdf",
    "sentence_transformers",
    "sqlite_vec",
    "watchfiles",
]:
    print(module, bool(importlib.util.find_spec(module)))
PY
```

## Warum SQLite statt LanceDB?

LanceDB wurde auf dem Referenzsystem bewusst nicht verwendet.

Der Rechner hat eine Sandy-Bridge-CPU:

```text
Intel Core i7-2600K
```

Auf diesem System beendet bereits `import lancedb` den Python-Prozess mit:

```text
Illegal instruction (core dumped)
```

Ursache ist sehr wahrscheinlich ein CPU-Instruktionssatz-Konflikt in einem
Binary-Wheel aus LanceDB/PyArrow/Lance. Deshalb verwendet dieses Plugin SQLite.

Auf neueren Rechnern kann LanceDB erneut geprueft werden:

```bash
/home/tobi/skripte/hermes-agent/venv/bin/python - <<'PY'
import lancedb
print("lancedb import ok")
PY
```

Wenn dieser Test fehlschlaegt, SQLite verwenden.

## SQLite-Vector-Support pruefen

```bash
/home/tobi/skripte/hermes-agent/venv/bin/python - <<'PY'
import sqlite3
import sqlite_vec

conn = sqlite3.connect(":memory:")
conn.enable_load_extension(True)
sqlite_vec.load(conn)
print(conn.execute("select vec_version()").fetchone()[0])
PY
```

Erwartet:

```text
v0.1.9
```

## Plugin-Dateien

Das Plugin liegt unter:

```text
/home/tobi/skripte/hermes-agent/plugins/cocoindex_docs/
```

Wichtige Dateien:

```text
config.py       Pfade, Tabellenname, Include-/Exclude-Regeln, Modellname
extractors.py   Text-, Code-, PDF-Extraktion und Pfad-/Topic-Injektion
pipeline.py     CocoIndex-App
search.py       SQLite-basierte semantische Suche
mcp_server.py   MCP-stdio-Server fuer Hermes
readme.md       Kurze Bedienhinweise
```

## Laufzeitpfade

Standardwerte:

```text
HERMES_DOCS_SOURCE_DIR=/home/tobi/Dokumente
COCOINDEX_DB=/home/tobi/.local/state/hermes/cocoindex/cocoindex.db
HERMES_DOCS_SQLITE_DB=/home/tobi/.local/share/hermes/cocoindex/documents.sqlite3
```

Die Werte koennen per Environment ueberschrieben werden.

## CocoIndex-App pruefen

```bash
cd /home/tobi/skripte/hermes-agent
COCOINDEX_DB=/home/tobi/.local/state/hermes/cocoindex/cocoindex.db \
  /home/tobi/skripte/hermes-agent/venv/bin/cocoindex ls \
  plugins/cocoindex_docs/pipeline.py
```

Erwartet ist die App:

```text
HermesDocumentsKnowledge
```

## Initialen Indexlauf starten

Einmaliger Catch-up-Lauf:

```bash
cd /home/tobi/skripte/hermes-agent
COCOINDEX_DB=/home/tobi/.local/state/hermes/cocoindex/cocoindex.db \
HERMES_COCOINDEX_VERBOSE=1 \
  /home/tobi/skripte/hermes-agent/venv/bin/cocoindex update \
  plugins/cocoindex_docs/pipeline.py -f
```

Live-Modus:

```bash
cd /home/tobi/skripte/hermes-agent
COCOINDEX_DB=/home/tobi/.local/state/hermes/cocoindex/cocoindex.db \
HERMES_COCOINDEX_VERBOSE=1 \
  /home/tobi/skripte/hermes-agent/venv/bin/cocoindex update \
  plugins/cocoindex_docs/pipeline.py -f -L
```

## systemd-User-Service

Service-Datei:

```text
/home/tobi/.config/systemd/user/hermes-cocoindex-docs.service
```

Aktivieren:

```bash
systemctl --user daemon-reload
systemctl --user enable --now hermes-cocoindex-docs.service
```

Status:

```bash
systemctl --user status hermes-cocoindex-docs.service
```

Logs:

```bash
journalctl --user -u hermes-cocoindex-docs.service -f
```

## MCP-Server in Hermes registrieren

```bash
/home/tobi/.local/bin/hermes mcp add cocoindex_docs \
  --command /home/tobi/skripte/hermes-agent/venv/bin/python \
  --args /home/tobi/skripte/hermes-agent/plugins/cocoindex_docs/mcp_server.py
```

Pruefen:

```bash
/home/tobi/.local/bin/hermes mcp list
/home/tobi/.local/bin/hermes mcp test cocoindex_docs
```

Wenn Hermes Gateway oder Telegram bereits laeuft, Gateway danach neu starten,
damit das neue MCP-Tool geladen wird:

```bash
systemctl --user restart hermes-gateway.service
```

Die Telegram-Verknuepfung selbst muss dabei nicht neu eingerichtet werden.

## Direkter Suchtest

```bash
cd /home/tobi/skripte/hermes-agent/plugins/cocoindex_docs
/home/tobi/skripte/hermes-agent/venv/bin/python - <<'PY'
from search import search_documents

result = search_documents(
    "Amplify REST API authorization rules",
    topic="aws",
    limit=3,
)
print(result["ok"])
for item in result["results"]:
    print(item["score"], item["relative_path"], item["topic"], item["file_type"])
PY
```

## Datenbank pruefen

```bash
cd /home/tobi/skripte/hermes-agent/plugins/cocoindex_docs
/home/tobi/skripte/hermes-agent/venv/bin/python - <<'PY'
from config import SQLITE_DB_PATH, TABLE_NAME
import sqlite3

conn = sqlite3.connect(SQLITE_DB_PATH)
try:
    print(conn.execute(f"select count(*) from {TABLE_NAME}").fetchone()[0])
    print(
        conn.execute(
            f"select relative_path, topic, file_type from {TABLE_NAME} limit 5"
        ).fetchall()
    )
finally:
    conn.close()
PY
```

## Wichtige Include-/Exclude-Erkenntnisse

`~/Dokumente` enthaelt nicht nur Dokumente, sondern auch venvs, Python-Artefakte,
SVN-Daten und grosse Medien-/Bilddateien. Deshalb sind harte Excludes wichtig.

Besonders wichtig:

```text
**/.svn/**
**/__pycache__/**
**/site-packages/**
**/venv/**
**/venv-*/**
**/env/**
**/env-*/**
**/node_modules/**
**/*.pyc
**/*.so
**/*.mov
**/*.mp4
**/*.psd
**/*.bmp
```

Diese Regel wurde nachgeschaerft, weil auf dem Referenzsystem ein Ordner
`venv-lukas` zuerst mit erfasst wurde.

## Reset / Neuaufbau

Service stoppen:

```bash
systemctl --user stop hermes-cocoindex-docs.service
```

Index-State und SQLite-Ziel entfernen:

```bash
rm -rf /home/tobi/.local/state/hermes/cocoindex/cocoindex.db
rm -f /home/tobi/.local/share/hermes/cocoindex/documents.sqlite3
rm -f /home/tobi/.local/share/hermes/cocoindex/documents.sqlite3-*
```

Service wieder starten:

```bash
systemctl --user start hermes-cocoindex-docs.service
```

Wichtig: CocoIndex-State und SQLite-Ziel muessen zusammenpassen. Wenn nur die
SQLite-Datei geloescht wird, kann CocoIndex Dateien als unveraendert betrachten,
obwohl die Zieldaten fehlen.

## Bekannte Stolperstellen

- LanceDB kann auf aelteren CPUs mit `Illegal instruction` crashen.
- Der erste SentenceTransformer-Lauf laedt bzw. initialisiert das Modell und ist langsamer.
- Ohne `HF_TOKEN` erscheinen Hugging-Face-Warnungen; das Setup funktioniert trotzdem.
- PDF-Extraktion mit `pypdf` funktioniert fuer Text-PDFs, nicht fuer gescannte PDFs/OCR.
- Der initiale Catch-up kann bei vielen PDFs mehrere Minuten dauern.
- Hermes Gateway muss nach neuer MCP-Registrierung neu gestartet werden.

## Minimaler Rollout-Ablauf

1. Hermes installieren und venv pruefen.
2. CocoIndex-Checkout bereitstellen.
3. CocoIndex editable in Hermes-venv installieren.
4. `sqlite-vec`, `sentence-transformers`, `pypdf`, `mcp` pruefen.
5. `plugins/cocoindex_docs/` kopieren.
6. `cocoindex ls` ausfuehren.
7. systemd-User-Service installieren und starten.
8. MCP-Server mit `hermes mcp add cocoindex_docs ...` registrieren.
9. `hermes mcp test cocoindex_docs` ausfuehren.
10. Hermes Gateway neu starten.
11. Direkten Suchtest und Telegram-Test durchfuehren.
