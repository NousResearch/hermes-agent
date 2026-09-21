#!/usr/bin/env bash
set -euo pipefail
stamp="$(TZ=Europe/London date '+%d/%m/%y %H:%M:%S')"
out="/tmp/hermes-discord-cron-html-attachment-probe.html"
cat > "$out" <<'HTML'
<!doctype html>
<html>
<head><meta charset="utf-8"><title>Discord HTML Attachment Probe</title></head>
<body><h1>Discord HTML Attachment Probe</h1><p>If you can download/open this from Discord, cron MEDIA HTML attachments are working.</p></body>
</html>
HTML
printf '✅ HTML attachment probe · %s\n' "$stamp"
printf 'checked · 1 file · should be downloadable in Discord\n\n'
printf '• This should appear as an uploaded `.html` file, not a visible VPS path.\n\n'
printf 'MEDIA:%s\n' "$out"
