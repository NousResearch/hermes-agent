# Twilio Voice Setup (Hermes Phone MVP)

## 1) Secrets in `.hermes/.env`

Add these values:

```bash
TWILIO_AUTH_TOKEN=YOUR_TWILIO_AUTH_TOKEN
VOICE_HOST=0.0.0.0
VOICE_PORT=8091
VOICE_HERMES_TIMEOUT_SEC=35
TWILIO_TTS_VOICE_DE=Polly.Vicki
TWILIO_TTS_VOICE_TR=Polly.Filiz
TWILIO_TTS_VOICE_EN=Polly.Joanna
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_REALTIME_MODEL=gpt-4o-realtime-preview
OPENAI_REALTIME_VOICE=alloy
PUBLIC_WSS_BASE=https://YOUR_PUBLIC_DOMAIN

# Optional Vapi guest-intake pilot. Do not commit real values.
VAPI_WEBHOOK_BEARER_TOKEN=YOUR_VAPI_CUSTOM_CREDENTIAL_TOKEN
# or:
VAPI_WEBHOOK_HMAC_SECRET=YOUR_VAPI_CUSTOM_CREDENTIAL_HMAC_SECRET
VAPI_GUEST_ASSISTANT_ID=YOUR_FIXED_GUEST_ASSISTANT_ID
VAPI_GUEST_INTAKE_DIR=/Users/appleserver/.hermes/reports/vapi-guest-intake
```

Note:
- If a value was shared in chat and looks like a secret/token, put it in `TWILIO_AUTH_TOKEN` locally and do not commit it.
- If this token was posted in chat, rotate it once after setup.

## 2) Start service

```bash
/Users/appleserver/hermes-system/core/scripts/run_voice_gateway.sh
```

Health check:

```bash
curl http://127.0.0.1:8091/health
```

## 3) Public URL for Twilio webhook

Twilio needs a public HTTPS endpoint.

Option A: reverse proxy (recommended)
- `https://your-domain/twilio/voice` -> `http://127.0.0.1:8091/twilio/voice`

Option B: temporary tunnel
- `ngrok http 8091`

## 4) Twilio Console

For your phone number:
- Voice webhook URL (Phase 1 classic STT): `https://YOUR_PUBLIC_URL/twilio/voice`
- Voice webhook URL (Phase 2 realtime/Jarvis): `https://YOUR_PUBLIC_URL/twilio/voice-realtime`
- Method: `POST`

Use `/twilio/voice` for productive hotel business first. It routes the caller through Hermes and keeps the existing workflow rules.
Use `/twilio/voice-realtime` only as a low-latency frontdesk until a secured tool bridge exists. Realtime mode must not claim live HotelRunner checks, bookings, finance access, or system administration.

## 5) Test flow

1. Call the number.
2. Hermes greets and asks your intent.
3. Say: booking request / room availability / price.
4. Hermes responds and asks follow-up.

Realtime test (Phase 2):
1. Set webhook to `/twilio/voice-realtime`.
2. Call the number.
3. Twilio opens websocket to `wss://YOUR_PUBLIC_DOMAIN/twilio/stream`.
4. Audio streams bidirectional via OpenAI Realtime.

## 6) Production hardening (next step)

- Add per-caller rate limiting.
- Add explicit escalation to team member on confidence drop.
- Move from speech gather to Twilio media stream + OpenAI Realtime for lower latency.

## 7) Vapi guest-intake pilot

The gateway also exposes a deliberately narrow Vapi webhook:

```text
POST /vapi/events
```

Use this only for guest communication experiments. Keep Twilio as the number and routing owner, and point only a test number or reversible forwarding path at Vapi.

Security defaults:
- Without `VAPI_WEBHOOK_BEARER_TOKEN` or `VAPI_WEBHOOK_HMAC_SECRET`, `/vapi/events` returns `403`.
- `VAPI_WEBHOOK_AUTH_DISABLED=1` is accepted only for localhost tests.
- Vapi Custom Credentials should send either a bearer token or an HMAC SHA-256 signature.

Allowed pilot tools:
- `create_guest_intake`: records a guest-intake handoff for Hermes/Paperclip review.
- `build_booking_link`: read-only HotelRunner booking-link helper when arrival and departure dates are known.
- `request_human_followup`: records a callback/follow-up request.
- `transfer_to_reception`: records or routes a transfer request; use only when the caller asks or policy allows it.

Forbidden in the pilot:
- Booking creation, cancellation, price changes, payment handling, refunds, internal team data, WhatsApp IDs, admin links, logs, secrets, or raw transcript dumps into the Vault.

Vapi assistant boundary:
- Vapi may talk to guests and collect facts.
- Hermes decides what can happen.
- Paperclip is the durable work queue.
- Humans approve sensitive actions.
