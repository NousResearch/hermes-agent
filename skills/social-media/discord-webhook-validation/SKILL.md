---
name: discord-webhook-validation
description: Safely validate a Discord webhook by fetching its metadata first, confirm whether it can post, and avoid storing leaked webhook secrets.
---

# Discord webhook validation

Use this when a user shares a Discord webhook and wants to know whether the setup is correct.

## Goals
- Verify whether the webhook is valid without sending a message yet.
- Distinguish clearly between posting capability and read capability.
- Treat the webhook URL/token as sensitive and avoid persisting it.

## Procedure
1. **Do not store the raw webhook URL or token in memory.**
   - Webhook URLs are secrets.
   - If the user pasted one in chat, treat it as potentially leaked.

2. **Validate by GET request first, not POST.**
   - Use a simple HTTP GET against the webhook URL.
   - A valid webhook usually returns HTTP 200 and JSON metadata including:
     - `id`
     - `name`
     - `channel_id`
     - `guild_id`
     - `type`
   - This confirms the webhook exists and can be used for posting to that bound channel.

3. **Explain the result in plain language.**
   - If GET returns 200: tell the user the webhook is valid and identify the webhook name/channel/server IDs if available.
   - If unauthorized / not found: tell the user the webhook is invalid, deleted, malformed, or revoked.

4. **Clarify the capability boundary.**
   - Discord webhook = can post to a specific channel.
   - Discord webhook != read existing channel messages.
   - Reading requires a bot/integration with channel permissions.

5. **Recommend secret rotation after exposure.**
   - If a webhook was pasted into chat, advise the user to regenerate/rotate it in Discord.
   - Explain briefly that anyone with the URL can post to that channel.

6. **Only send a test message if the user explicitly wants that.**
   - Default to non-posting validation first.
   - If posting, make the message clearly labeled as a test.

## Example validation command
Use terminal or equivalent HTTP tooling:

```bash
python - <<'PY'
import urllib.request
url='https://discord.com/api/webhooks/...'
req=urllib.request.Request(url, headers={'User-Agent':'Hermes'})
with urllib.request.urlopen(req, timeout=20) as r:
    print(r.status)
    print(r.read().decode('utf-8','replace'))
PY
```

## Response template
- “Ja — deze webhook werkt.”
- Mention `name`, `guild_id`, and `channel_id` if returned.
- Add: “Belangrijk: deze URL is een geheim; omdat hij gedeeld is, raad ik aan hem te regenereren.”
- Add: “Met deze webhook kan ik posten, niet lezen.”

## Pitfalls
- Do not claim read access from webhook validation.
- Do not save raw webhook secrets in memory or skills.
- Do not send a test post unless the user asked for it.
