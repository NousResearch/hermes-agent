# Reasoning-replay consumption eval

Which providers and models actually **read** the reasoning that a client replays from
earlier assistant turns? A `200 OK` only shows a field was accepted. The gateway may drop it,
and so may the chat template (for example, Qwen3 templates discard reasoning from earlier turns).
This harness measures consumption on the wire. It uses only the standard library's HTTP client
and has no Hermes runtime dependency, so it tests what a provider does with a request, not what
Hermes sends.

## Method

**Step 1 — synthetic text carriers.** The fixture is: system prompt, user task, an assistant
`get_weather` tool call, the tool result, then a probe. The assistant tool-call turn carries a
reasoning trace of about 400 tokens. The trace is distinctive filler with one canary sentence
(`The verification word for this task is <word>.`). It is sent in one of these variants:

| variant | wire shape |
|---|---|
| `none` | baseline: no reasoning |
| `reasoning_content` | `assistant.reasoning_content = trace` |
| `reasoning` | `assistant.reasoning = trace` |
| `reasoning_details` | `[{type:"reasoning.text", text, format:"unknown", index:0}]` |
| `all` | all three at once |

Responses routes use `summary_item` (a `reasoning` input item with `summary_text`) and
`content_item` (`reasoning_text` content). Gemini native uses `thought_part` (`{text, thought:true}`).

Each variant is tested at two positions:

- **in_loop**: the reasoning sits on the tool-call turn immediately before the current request.
  The probe question rides in the tool result, and no new user message follows.
- **cross_turn**: the tool loop has finished (`tool result`, then a final assistant answer), and a
  new user message asks the probe.

**Primary signal:** the `prompt_tokens` delta against `none`, with every other byte identical.
- A delta close to the trace size means the reasoning was tokenized into the prompt (**consumed**).
- A delta near 0 means the gateway or the template dropped it.
- A 4xx means the field was rejected; the status code and exact error are recorded.

**Secondary signal:** canary recall. The probe asks the model to reply with the verification
word or `NONE`. A negative result is not conclusive, because models may refuse.

**Step 2 — real opaque carriers.** Turn 1 runs live against the same model and records whatever
reasoning it returns. That can include `reasoning_details` (`reasoning.text`, `reasoning.encrypted`,
signatures), `reasoning_content`, Gemini `extra_content` / `thoughtSignature`, or encrypted
Responses `reasoning` items. Turn 2 is then sent twice, identical except that the carriers are
replayed **verbatim** in one and **stripped** in the other. The step-2 cells report the
prompt-token delta and the status code.

When turn 1 returns no carrier or no tool call, the reasoning effort is raised (low → medium →
high). If it still returns nothing, the opaque cells are left empty.

On OpenAI Responses, both turn-2 requests are also sent with `reasoning.context: "all_turns"`,
which the API rejects with a 400 on models that do not support it. Gemini native adds a third
turn-2 request carrying the documented `skip_thought_signature_validator` dummy signature.

**Lanes (Nous Portal).** The response id identifies the lane. `gen-…` means the OpenRouter lane,
and anything else (`chatcmpl-`, a uuid, `msg_`) means a direct upstream. The same public id can
switch lanes between calls. Pass `--want-direct id1,id2` to resample a cell up to
`--lane-attempts` times until it lands on a direct lane. Results are keyed by
`(route, model, lane)`. A cell that only ever reached OpenRouter is reported as
`direct-lane-not-reached`, never as a pass.

**Probe wording.** xAI (chat and Responses) returns
`403 "I'm sorry, I can't help with that request."` for a user turn that asks about the
model's "earlier reasoning". When a baseline gets that 403, the whole position is rerun with a
neutral wording (`alt`: "If a verification word was mentioned for this task…"). Deltas are
only ever computed between requests that used the same wording.

## Verdicts

Let `T` be the trace size in tokens (about chars/4). A carrier counts as consumed when its delta
is at least `0.5·T`. For opaque carriers, any verbatim delta above 25 tokens counts.

- `consumes-all-turns`: a cross-turn carrier is consumed.
- `in-loop-only`: an in-loop carrier is consumed, but no cross-turn carrier is.
- `accepted-ignored`: returns 200, but the delta is about 0 everywhere.
- `rejected`: every carrier returned 4xx.
- `baseline-failed` / `direct-lane-not-reached`: no comparable measurement exists.

## Running

```bash
# what the Portal lists, minus aliases / :batch / non-text / non-reasoning ids
python evals/reasoning_replay/harness.py models --route nous

# priority ids, resampled until a direct-lane result exists for each cell
python evals/reasoning_replay/harness.py run --route nous \
    --models x-ai/grok-4.7,z-ai/glm-5.3 --want-direct x-ai/grok-4.7,z-ai/glm-5.3 --out $OUT

# whole Portal catalog up to $4 per 1M output tokens; vendor routes
python evals/reasoning_replay/harness.py run --route nous --models portal-max:4 --out $OUT
python evals/reasoning_replay/harness.py run --route openai-responses --models gpt-6-luna --out $OUT
python evals/reasoning_replay/harness.py run --route gemini-native --models gemini-3.5-flash --out $OUT

# any OpenAI-compatible endpoint (DeepSeek, Moonshot, Z.ai, DashScope, MiniMax, Fireworks,
# Groq, Cerebras, Mistral, vLLM, llama.cpp, ...)
python evals/reasoning_replay/harness.py run --route custom --kind chat \
    --base-url https://api.deepseek.com/v1 --key-env DEEPSEEK_API_KEY --models deepseek-reasoner --out $OUT

python evals/reasoning_replay/harness.py report --out $OUT   # writes matrix.md / matrix.json
```

**Credentials** are read at call time and never written to disk or printed.
- The Nous Portal key comes from `~/.hermes/auth.json` (`providers.nous.agent_key`, falling
  back to `access_token`). On a 401 the file is re-read once, waiting 15 s if the key has not
  changed yet.
- Every other route reads its key from the process environment first, then from
  `~/.hermes/.env`.
- Named routes are listed in `ROUTES`. For anything else, use `--route custom` with
  `--base-url` and `--key-env`.

**Budget.** Every call is appended to `<out>/cost.jsonl` with `usage.cost` when the provider
returns it, and an estimate from the Portal price list otherwise. A run stops at `--budget`
(default $10), which counts every run that shares the same `--out`. Raw records go to
`calls.jsonl`, and run metadata (seed, canary, trace size) goes to `runs.jsonl`.

Keep result directories outside the repo.
