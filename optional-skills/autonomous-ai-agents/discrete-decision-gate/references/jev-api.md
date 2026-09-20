# TypeSafe Jev: what the endpoint actually is

Research date: 2026-09-20. Jev launched 2026-09-15 (TypeSafe AI). Verify against
`https://docs.typesafe.ai/api` before relying on any detail here.

## Contract

One endpoint, typed questions, no text generation:

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer $TYPESAFE_API_KEY
    Content-Type: application/json

    {
      "state": "Help! My payouts have been failing for 3 days.",
      "model": "jev-latest",
      "questions": {
        "is_urgent": {
          "type": "noul",
          "instructions": "Does this convey urgency?",
          "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}
        },
        "department": {
          "type": "choice",
          "instructions": "Which team should handle this",
          "criteria": {"billing": "Payment or subscription issues",
                       "technical": "Bugs or integration problems"}
        }
      }
    }

Response carries one answer per named question plus usage:

    {"model": "jev-latest",
     "answers": {"department": {"type": "choice", "choice": "billing",
                               "probabilities": {"billing": 0.87, "technical": 0.13},
                               "confidence": 0.82}},
     "usage": {"input_tokens": 312, "output_tokens": 48}}

Primitives: `choice` (one of up to 255 options, full probability map), `noul` (yes/no
probability 0-1), `score` (probability-weighted value on an ordered rubric). All questions in
one request are answered in a single parallel pass. SDKs: `pip install typesafe-sdk` (Python
3.10+) or `npm install @typesafe-ai/sdk` (Node 20+); both read `TYPESAFE_API_KEY` and default to
`jev-latest`. HTTP-only callers (this skill) do not need either SDK.

## Access paths (checked live 2026-09-20)

| Path | Status |
|---|---|
| `api.typesafe.ai/v1/systemone` directly | Reachable. No key: HTTP 403 `Must supply an API key!`; wrong key: HTTP 401 `authentication_error`. |
| OpenRouter `/chat/completions` with `typesafe/jev` | HTTP 400 `typesafe/jev is not a valid model ID`. The `/api/v1/providers` list does include a `TypeSafe` provider, but `/api/v1/models` (447 models) contains no `typesafe/*` id. |
| OpenRouter **Decisions** route (alpha) | **Works.** `POST https://openrouter.ai/api/alpha/decisions`, flat body, model `typesafe/jev-1.13`. See the section below. |
| OpenRouter `/api/v1/decisions` | HTTP 404 - the route is under `/api/alpha`, not `/api/v1`. |
| Vercel AI Gateway, `typesafe-ai/jev` | `/chat/completions` rejected: "is an evaluation model, not a language model. Use the evaluation generation API instead." |
| Local substitutes | Any chat model constrained to one label (OpenRouter), or Ollama for offline use. |

Keys: `console.typesafe.ai/settings/keys` (early access was waitlisted at launch).

Opt-in note: the 1P route is NOT in the local auto chain - OpenRouter's Decisions route is the
default and 1P is reached only with `--backend typesafe` / `JEV_BACKEND=typesafe`. A stray free
key should not be able to become the gate path, because a rate-limited provider fails closed and
every verdict would look permanently cautious.

## Claimed numbers (vendor-reported, unverified)

- `$0.042` per 1M input tokens, output unmetered; 70-500ms end-to-end latency
- 64k request context (32k state plus the longest single question); rate limits ~250k tok/s,
  1,200 req/min, both stated as subject to change
- Text-only input; answers are schema-conformant by construction, every answer carries confidence

Treat these as vendor claims. Nothing here was measured against a live Jev key, because none was
available - only the failure modes above were reproduced.

## OpenRouter Decisions route (alpha) - measured 2026-09-20

Works with an ordinary `OPENROUTER_API_KEY`, no TypeSafe account:

    POST https://openrouter.ai/api/alpha/decisions
    Authorization: Bearer $OPENROUTER_API_KEY
    Content-Type: application/json

    {"model": "typesafe/jev-1.13",
     "state": {"ticket": "..."},
     "questions": {"tier": {"type": "choice",
                            "instructions": "pick a tier",
                            "criteria": {"bash_direct": "inspection only",
                                         "crg_first": "refactor across modules"}}}}

The body is FLAT - `model`, `questions`, `state` at the top level. Wrapping it in
`{"decisionsRequest": {...}}` returns HTTP 400 `invalid_type ... path: ["model"]`.

Response is TypeSafe-shaped plus usage and provenance:

    {"model": "typesafe/jev-1.13-20260917",
     "answers": {"tier": {"type": "choice", "choice": "bash_direct",
                          "probabilities": {"bash_direct": 1, "crg_first": 0},
                          "confidence": 1}},
     "usage": {"input_tokens": 312, "output_tokens": 35, "cost": 1.31e-05},
     "id": "gen-dec-...", "provider": "TypeSafe"}

Verified answer shapes:

| Type | Answer | Notes |
|---|---|---|
| `choice` | `{"type": "choice", "choice": "...", "probabilities": {...}, "confidence": 1}` | the only type that carries `confidence` |
| `noul` | `{"type": "noul", "noul": 0.01}` | flat probability, **no** `probabilities`, **no** `confidence` |

Measured cost and latency: $1.5-2.7e-05 per call (~$16-27 per million), 250-430ms end to end;
`state` accepts a JSON object or a string. The route is marked alpha and the model catalog does not
list it, so log `endpoint` + `model` + `cost` from each response and treat a silent endpoint move
as a drift event rather than a mysterious constant verdict.

### Finding "the latest Jev" on OpenRouter (measured)

There is no `-latest` alias. Measured answers:

| Model id sent | Result |
|---|---|
| `typesafe/jev-1.13` | 200, resolves to `typesafe/jev-1.13-20260917` |
| `typesafe/jev-latest` | error `Model typesafe/jev-latest does not exist` |
| `typesafe/jev` | `Model typesafe/jev does not exist` |
| `typesafe/jev-latest-20260918` | `Model typesafe/jev-latest-20260918 does not exist` |

Two discovery paths exist, and only one of them works - the catalog hides this family:

    GET https://openrouter.ai/api/v1/models/typesafe/jev-1.13/endpoints   # 200
    GET https://openrouter.ai/api/v1/models/typesafe/jev-1.13             # 404

The endpoints response carries the provider endpoint name (`TypeSafe | typesafe/jev-1.13-20260917`),
`created`, `pricing.prompt` (`0.000000042` per token = $0.042/M), `context_length` (32000),
`max_completion_tokens` (28800), `uptime_last_1d` and `architecture.modality`
(`text->decisions`). So the usable form of "latest" is: pin `typesafe/jev-1.13`, read the served
snapshot from every response, and watch the advertised endpoint name for a version bump.
