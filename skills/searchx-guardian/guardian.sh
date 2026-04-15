#!/bin/bash
# SearchX Guardian - Health Monitor for SearXNG

URL="http://localhost:8888"
CONTAINER_NAME="searxng"

echo "[$(date)] Checking SearXNG health..."

# Check if the service returns 200 OK
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$URL")

if [ "$STATUS_CODE" -eq 200 ]; then
    echo "[$(date)] Status: 200 OK. SearchX is healthy."
else
    echo "[$(date)] Status: $STATUS_CODE. ERROR detected. Attempting restart..."
    docker restart "$CONTAINER_NAME"
    if [ $? -eq 0 ]; then
        echo "[$(date)] Successfully restarted $CONTAINER_NAME."
    else
        echo "[$(date)] Failed to restart $CONTAINER_NAME."
    fi
fi
