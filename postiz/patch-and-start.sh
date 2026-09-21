#!/bin/sh
set -eu

# Persistent local startup patch for Sahil's self-hosted Postiz container.
# 1. Keep LinkedIn personal-profile OAuth usable on self-hosted Postiz.
# 2. Disable /api/copilot/chat at nginx so the UI cannot trigger 30s upstream timeouts
#    while stock Postiz lacks proven OpenAI-compatible provider routing.

sed -i 's/&prompt=none//' \
  /app/apps/backend/dist/libraries/nestjs-libraries/src/integrations/social/linkedin.provider.js \
  /app/apps/backend/dist/libraries/nestjs-libraries/src/integrations/social/linkedin.page.provider.js \
  /app/apps/orchestrator/dist/libraries/nestjs-libraries/src/integrations/social/linkedin.page.provider.js 2>/dev/null || true

sed -i '/this.scopes = \[/,/\];/c\        this.scopes = [\n            "openid",\n            "profile",\n            "w_member_social",\n        ];' \
  /app/apps/backend/dist/libraries/nestjs-libraries/src/integrations/social/linkedin.provider.js \
  /app/apps/backend/dist/libraries/nestjs-libraries/src/integrations/social/linkedin.page.provider.js \
  /app/apps/orchestrator/dist/libraries/nestjs-libraries/src/integrations/social/linkedin.page.provider.js 2>/dev/null || true

python3 - <<'PY'
from pathlib import Path
p = Path('/etc/nginx/nginx.conf')
s = p.read_text()
block = '''        location = /api/copilot/chat {
            default_type application/json;
            return 503 '{"error":"Postiz Copilot disabled on this self-hosted instance until OpenAI-compatible provider routing is supported"}';
        }

'''
if 'location = /api/copilot/chat' not in s:
    s = s.replace('        location /api/ {', block + '        location /api/ {')
    p.write_text(s)
PY

nginx
exec pnpm run pm2
