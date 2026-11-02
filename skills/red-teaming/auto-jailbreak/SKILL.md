---
name: auto-jailbreak
description: "Red-team any LLM with Microsoft PyRIT — Crescendo, TAP, PAIR, many-shot and prompt converters. Configurable target and attacker models, plus a learning memory that makes the adversary smarter over time. For authorized robustness testing."
version: 1.0.0
author: VJ
license: MIT
metadata:
  hermes:
    tags: [red-teaming, security, pyrit, robustness, evaluation, jailbreak]
    related_skills: [godmode]
---

# Auto-Jailbreak

Runs real adversarial attacks against an LLM to **measure how robust it is** to
jailbreaks. Built on top of [Microsoft PyRIT](https://github.com/Azure/PyRIT)
(Python Risk Identification Toolkit) — the published red-teaming techniques,
wrapped in one CLI. Point it at a target model, pick an attack mode, get a
verdict back.

The target and the attacker are **separate models**. The target is the model
under test; the attacker is a less-censored model that writes the escalation
(a heavily-aligned model refuses to play attacker). A **learning memory** records
what worked per target/category and feeds winning angles back into later runs.

## ⚠️ Responsible use

This is a **security research / robustness evaluation** tool. Use it only on
models you own or are **explicitly authorized** to test. It measures a model's
resistance to known, published jailbreak techniques so you can harden it — the
same purpose as Microsoft PyRIT itself. Do not use it to extract harmful content
from third-party services you are not authorized to assess.

## Install (one command)

```bash
bash scripts/install.sh
```

`install.sh` **auto-detects what you already have** and configures itself:

- If an **OpenRouter key** is present in your Hermes config (`~/.hermes/.env`,
  the standard `OPENROUTER_API_KEY`), it uses it and picks a proven pair —
  attacker `hermes-4-405b` breaking target `deepseek-v4-pro`. Just press Enter.
- Else if a local **Ollama** is running, it goes fully local with your models.
- Else it asks you.

Flags: `--yes` (apply the detected setup, zero questions), `--manual` (choose
everything), `--plan` (print what it would do, then exit). It writes your config
to `~/.auto-jailbreak/config.env` and runs a smoke test.

**No telemetry, no phone-home:** the tool only talks to the model endpoints you
choose. Point them at local models (Ollama) and nothing leaves your machine.

Then run attacks with `scripts/redteam.sh` (it loads your config and runs the
engine in the isolated env — see below).

## Manual setup (alternative)

- Python 3.11+
- Install dependencies (PyRIT pulls the heavy transitive deps: transformers,
  datasets, sqlalchemy…):

```bash
pip install -r scripts/requirements.txt
```

- A reachable **target** endpoint (any OpenAI/Anthropic-compatible or local
  server — Ollama, LM Studio, vLLM, a proxy…). LiteLLM handles the transport.

## Configuration (environment variables)

Nothing is hard-coded — everything is set via env vars, no code edits.

**Target** (the model under test):

| Variable | Meaning | Default |
|---|---|---|
| `LITELLM_ENDPOINT` | Target base URL | `http://localhost:8000/anthropic` |
| `LITELLM_MODEL` | Target model id (LiteLLM format) | `anthropic/claude-sonnet-4-5-notools` |
| `LITELLM_API_KEY` | Target API key | `proxy-local` |
| `LITELLM_MAX_TOKENS` | Max tokens per reply | `1024` |

**Attacker** (writes the escalation — use a less-censored model). Falls back to
the target if unset:

| Variable | Meaning |
|---|---|
| `LITELLM_ADVERSE_ENDPOINT` | Attacker base URL |
| `LITELLM_ADVERSE_MODEL` | Attacker model id |
| `LITELLM_ADVERSE_API_KEY` | Attacker API key |

**Judge** (scores whether the target actually complied). Falls back to the
target if unset: `LITELLM_JUGE_ENDPOINT`, `LITELLM_JUGE_MODEL`,
`LITELLM_JUGE_API_KEY`.

**Learning memory**:

| Variable | Meaning | Default |
|---|---|---|
| `PYRIT_MEMOIRE_DB` | SQLite file for the learning DB | `~/.auto-jailbreak/apprentissage.db` |
| `PYRIT_APPRENTISSAGE` | `off` to stop injecting past wins (recording stays on) | on |

Example target = a local Ollama model:

```bash
export LITELLM_ENDPOINT=http://localhost:11434
export LITELLM_MODEL=ollama/llama3
```

## How to run

The engine reads a **JSON objective on stdin** and prints the result on stdout,
prefixed by the marker `###PYRIT_JSON###` (PyRIT emits log lines before it, so
read the JSON that follows the marker):

```bash
echo '{"question": "OBJECTIVE HERE", "mode": "crescendo"}' \
  | bash scripts/redteam.sh
```

Parse the output: take everything after `###PYRIT_JSON###`, `json.loads` it. The
object contains `ok`, whether the target complied, the prompts used and a reply
excerpt.

## Attack modes

Set `"mode"` in the input JSON:

| `mode` | Technique |
|---|---|
| `crescendo` | Multi-turn Crescendo escalation (attacker-written) |
| `tap` | Tree of Attacks with Pruning |
| `pair` | PAIR (iterative refinement) |
| `manyshot` | Many-shot jailbreak |
| `conv:math` | MathPrompt converter |
| `conv:persuasion` | Persuasion converter |
| `conv:pastense` | Past-tense reframing |
| `policy` | Policy Puppetry |
| `scan` | Try every single-turn strategy, report each |
| `percer` | Try strategies, stop at first breach |
| `suite` | Follow-up turn on an existing conversation (pass `historique`) |

If `mode` is omitted, a single-turn strategy runs (default `skeleton_key`). Pick
it with `"strategie"`: `direct`, `base64`, `leetspeak`, `rot13`, `cesar`,
`morse`, `flip`, `skeleton_key`.

Optional input keys: `strategie`, `strategies` (list, for `scan`/`percer`),
`cible_nom` / `adverse_nom` (logical names used as the memory key),
`historique` (for `suite`), `max_tokens`, `percee_tours`.

## Examples

Crescendo against the configured target:

```bash
echo '{"question": "<your authorized test objective>", "mode": "crescendo"}' \
  | bash scripts/redteam.sh
```

Many-shot, with a dedicated uncensored attacker:

```bash
export LITELLM_ADVERSE_ENDPOINT=https://openrouter.ai/api/v1
export LITELLM_ADVERSE_MODEL=openrouter/cognitivecomputations/dolphin-mistral-24b-venice-edition
export LITELLM_ADVERSE_API_KEY=$OPENROUTER_API_KEY
echo '{"question": "<objective>", "mode": "manyshot", "cible_nom": "llama3"}' \
  | bash scripts/redteam.sh
```

Scan every single-turn strategy:

```bash
echo '{"question": "<objective>", "mode": "scan"}' | bash scripts/redteam.sh
```

## Learning memory

Each run is recorded in the SQLite DB (`PYRIT_MEMOIRE_DB`). Per target and
category, the engine remembers which angles broke through and which never did,
and feeds the winning angles into the attacker's system prompt on later runs —
so the adversary gets cumulatively better against a given target. Set
`PYRIT_APPRENTISSAGE=off` to stop injecting past wins (recording still happens),
useful for clean A/B measurements.

## Files

- `scripts/install.sh` — one-command setup (deps + interactive config).
- `scripts/redteam.sh` — run wrapper (loads config, isolated env).
- `scripts/attaque.py` — the engine (all techniques, stdin/stdout contract).
- `scripts/cible.py` — builds the target and attacker from env vars.
- `scripts/memoire.py` — the learning memory (standalone SQLite).
- `scripts/requirements.txt` — `pyrit` + `litellm`.

## Credits

Attack techniques and datasets: [Microsoft PyRIT](https://github.com/Azure/PyRIT)
(MIT). This skill is a configurable CLI wrapper with a learning layer on top.
