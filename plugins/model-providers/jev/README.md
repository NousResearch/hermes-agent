# Jev (TypeSafe) provider

Registers the `jev` provider (`typesafe`, `typesafe-ai` aliases) so `hermes setup`,
`hermes auth`, `hermes doctor`, and `hermes model` auto-wire `TYPESAFE_API_KEY`.

Jev is TypeSafe's System One advisory model (`jev-latest`). Its API is
`POST https://api.typesafe.ai/v1/systemone`, not chat completions, and TypeSafe
publishes no `/v1/models` catalog, so the model list is static and health
checks are skipped.

This is NOT a chat backend. Do not set `jev-latest` as a session, aux, or
compression model. The live consumer is the out-of-tree
[typesafe-skill-router](https://github.com/DECRUX9812/typesafe-skill-router)
plugin, which calls System One directly and stays fail-open.
