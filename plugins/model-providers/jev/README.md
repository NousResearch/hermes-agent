# TypeSafe (Jev) provider

Registers TypeSafe Jev in Hermes's model-provider list so setup, `hermes auth`,
`hermes doctor`, and `hermes model` know about it.

This is **not** a chat-completions backend. TypeSafe's API is
`POST /v1/systemone` (model `jev-latest`). The working consumer is the
community plugin [typesafe-skill-router](https://github.com/DECRUX9812/typesafe-skill-router).
Do not set `model.provider: jev` as the session chat model.

## Auth

Create a key at https://console.typesafe.ai/settings/keys and put it in the
profile `.env` as `TYPESAFE_API_KEY`. Optional override: `TYPESAFE_BASE_URL`
(default `https://api.typesafe.ai/v1`).

`hermes doctor` skips a `/v1/models` probe (`supports_health_check=False`).
The picker catalog is the static id `jev-latest`.
