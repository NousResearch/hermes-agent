# Doni Recall-like MVP

Lokalni prototip: capture, summary, search, Q&A + web UI.

Sada ima i **Rekol lane**:
- Telegram tekst unos (`/api/add_telegram`)
- YouTube URL unos (`/api/add_youtube`)
- automatski export svake bilješke u Obsidian vault (`Inbox/Rekol`, canonical)

## Run
```bash
cd /mnt/d/HermesAgent/app/recall-like-mvp
python3 app.py
```

App će biti na: `http://127.0.0.1:8099`

## Pristup s telefona (LAN)
1) Pokreni app da sluša na svim interfejsima:
```bash
export DONI_RECALL_HOST=0.0.0.0
export DONI_RECALL_PORT=8099
python3 app.py
```
2) Na računalu uzmi lokalni IP (npr. `192.168.1.50`):
```bash
hostname -I
```
3) Na telefonu (isti Wi‑Fi) otvori:
- `http://<LAN_IP>:8099`

Napomena za OAuth: provider callback mora odgovarati URL-u koji koristiš. Za pristup samo unutar kućnog Wi‑Fi-ja može i LAN URL; za pristup izvan kuće treba javni HTTPS URL (tunnel/reverse proxy + domena).

## API
- `POST /api/add_text` `{"title":"...","text":"..."}`
- `POST /api/add_url` `{"url":"https://..."}`
- `POST /api/add_youtube` `{"url":"https://www.youtube.com/watch?v=..."}`
- `POST /api/add_telegram` `{"title":"...","text":"...","chat":"telegram","author":"..."}`
- `POST /api/add_file` `{"path":"/mnt/d/.../note.md","title":"optional"}` (txt/md/log)
- `GET /api/notes?limit=N`
- `GET /api/search?q=...`
- `GET /api/stats`
- `POST /api/ask` `{"question":"..."}`

## Obsidian auto-sync
Po defaultu svaka nova bilješka ide i u markdown file pod:
- `~/Documents/Obsidian Vault/Inbox/Rekol/`

Možeš promijeniti putanje:
```bash
export OBSIDIAN_VAULT_PATH="/putanja/do/tvog/vaulta"
export DONI_REKOL_OBSIDIAN_DIR="Inbox/Rekol"
```

## Optional auth (API key)
Ako želiš zaštitu API-ja, pokreni app s env var key-em:
```bash
export DONI_RECALL_API_KEY="<set-your-api-key>"
python3 app.py
```
Tada svi `/api/*` endpointi traže header: `X-API-Key: <set-your-api-key>`.

## OAuth2 / OIDC (novo)
App sada podržava OAuth login (Authorization Code) preko bilo kojeg OIDC providera.

Obavezni env varovi:
```bash
export DONI_RECALL_OAUTH_ENABLED=true
export DONI_RECALL_OAUTH_CLIENT_ID="..."
export DONI_RECALL_OAUTH_CLIENT_SECRET="..."
export DONI_RECALL_OAUTH_AUTHORIZE_URL="https://provider.example.com/oauth2/authorize"
export DONI_RECALL_OAUTH_TOKEN_URL="https://provider.example.com/oauth2/token"
export DONI_RECALL_OAUTH_USERINFO_URL="https://provider.example.com/oauth2/userinfo"
export DONI_RECALL_OAUTH_REDIRECT_URI="http://127.0.0.1:8099/oauth/callback"
# optional:
export DONI_RECALL_OAUTH_SCOPE="openid email profile"
export DONI_RECALL_OAUTH_ALLOWED_EMAILS="me@firma.com,admin@firma.com"
export DONI_RECALL_SESSION_SECRET="<set-strong-session-secret>"
```

Flow:
- otvori `http://127.0.0.1:8099/` → redirect na `/login`
- login kod providera
- callback na `/oauth/callback`
- logout na `/logout`

Napomena: API key i OAuth mogu raditi zajedno (API key bypass za automatizaciju).

## Brzi test
```bash
curl -s -X POST http://127.0.0.1:8099/api/add_text \
  -H 'content-type: application/json' \
  -d '{"title":"Test","text":"Recall-like app sprema bilješke i omogućava pretragu."}'

curl -s 'http://127.0.0.1:8099/api/search?q=bilješke'

curl -s -X POST http://127.0.0.1:8099/api/ask \
  -H 'content-type: application/json' \
  -d '{"question":"Što je smisao ove aplikacije?"}'
```
