# MoeMail REST API

The same workflow without MCP. Every request sends `X-API-Key: $MOEMAIL_API_KEY`.
Replace `https://moemail.app` with your own deployment when self-hosting.

```bash
H="X-API-Key: $MOEMAIL_API_KEY"
BASE=https://moemail.app

# 1. Pick a domain and create a mailbox (expiryTime in ms: 3600000 = 1 hour)
DOMAIN=$(curl -s "$BASE/api/config" -H "$H" | jq -r '.emailDomains | split(",")[0]')
BOX=$(curl -s -X POST "$BASE/api/emails/generate" -H "$H" -H 'content-type: application/json' \
  -d "{\"expiryTime\":3600000,\"domain\":\"$DOMAIN\"}")
ID=$(echo "$BOX" | jq -r .id); echo "$BOX" | jq -r .email

# 2. Wait for mail (long-poll up to 50 s). since=0 also returns mail that
#    already arrived; pass the returned cursor as since on the next call.
curl -s "$BASE/api/emails/$ID/wait?since=0&timeout=50" -H "$H" | jq '.messages[-1], .cursor'

# 3. Read a message in full
curl -s "$BASE/api/emails/$ID/<messageId>" -H "$H" | jq .message

# 4. Delete the mailbox
curl -s -X DELETE "$BASE/api/emails/$ID" -H "$H"
```
