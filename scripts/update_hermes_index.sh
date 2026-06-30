#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"

ACCOUNTS=("dimto-finanzen" "getflattobias" "tobias-freudling")

for ACCOUNT in "${ACCOUNTS[@]}"; do
    DEST_DIR="$HOME/Dokumente/hermes-dokuments/emails/$ACCOUNT"
    echo "Erstelle Zielverzeichnis $DEST_DIR"
    mkdir -p "$DEST_DIR"

    echo "Synchronisiere Account: $ACCOUNT"
    if ! himalaya folder list -a "$ACCOUNT" >/dev/null 2>&1; then
        echo "Warnung: Zugriff auf Account $ACCOUNT fehlgeschlagen (Passwort/GPG evtl. noch nicht eingerichtet). Überspringe..."
        continue
    fi

    while IFS= read -r FOLDER; do
        if [ -z "$FOLDER" ]; then
            continue
        fi
        echo "Prüfe Ordner: $FOLDER"
        mkdir -p "$DEST_DIR/$FOLDER"
        
        IDS=$(himalaya -o json envelope list -a "$ACCOUNT" -f "$FOLDER" --page-size 1000 2>/dev/null | python3 -c '
import json
import sys

try:
    envelopes = json.load(sys.stdin)
except json.JSONDecodeError:
    envelopes = []
for envelope in envelopes:
    message_id = envelope.get("id")
    if message_id:
        print(message_id)
')
        
        for ID in $IDS; do
            if [ ! -f "$DEST_DIR/$FOLDER/$ID.eml" ]; then
                echo "  Exportiere ID $ID in Ordner $FOLDER..."
                himalaya message export -a "$ACCOUNT" -f "$FOLDER" --full "$ID" -d "$DEST_DIR/$FOLDER/$ID.eml" >/dev/null 2>&1
            fi
        done
    done < <(himalaya -o json folder list -a "$ACCOUNT" | python3 -c '
import json
import sys

try:
    folders = json.load(sys.stdin)
except json.JSONDecodeError:
    folders = []
for folder in folders:
    name = folder.get("name")
    if name:
        print(name)
')
done

echo "Alle E-Mails wurden synchronisiert."

echo "E-Mail-Sync abgeschlossen."
echo "CocoIndex wird nicht durch dieses Skript gestartet."
echo "Die Indexierung übernimmt hermes-cocoindex-docs.service."
echo "Status prüfen mit: systemctl --user status hermes-cocoindex-docs.service --no-pager"
