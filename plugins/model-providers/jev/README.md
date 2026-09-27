# TypeSafe (Jev) provider

Registers TypeSafe Jev for **credential discovery only** — `hermes setup`,
`hermes auth`, and `hermes doctor` know about `TYPESAFE_API_KEY`. This follows
the same judgment-only posture as OMP's `/login typesafe` / `TypeSafeJudge`:
Jev is **not** listed in `hermes model` or the session chat picker
(`api_mode: systemone`).

This is **not** a chat-completions backend. TypeSafe's API is
`POST /v1/systemone` (model `jev-latest`). Install the community plugin
[typesafe-skill-router](https://github.com/DECRUX9812/typesafe-skill-router)
for skill routing, or call System One from your own hook. Do not set
`model.provider: jev` as the session chat model.

## Auth

Create a key at https://console.typesafe.ai/settings/keys and put it in the
profile `.env` as `TYPESAFE_API_KEY`. Optional override: `TYPESAFE_BASE_URL`
(default `https://api.typesafe.ai/v1`).

`hermes doctor` skips a `/v1/models` probe (`supports_health_check=False`).
The picker catalog is the static id `jev-latest`.
