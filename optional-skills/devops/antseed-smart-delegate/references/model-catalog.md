# AntSeed Model Catalog

172 models available through the AntSeed P2P network, 54 curator-verified. Listed below by category.

## Flagship (Chat + Reasoning)

| Model ID | Best For |
|----------|----------|
| `claude-opus-4-7` | Complex analysis, research |
| `claude-sonnet-4-6` | Balanced chat/code |
| `deepseek-v4-flash` | Fast general, **default** |
| `deepseek-v4-pro` | Deep reasoning |
| `deepseek-r1` | Advanced reasoning |
| `gemini-3-flash` | Fast multimodal |
| `gemini-3.1-pro` | Advanced multimodal |
| `gpt-5.5-pro` | High-quality general |
| `gpt-5.4` | Reliable general |
| `gpt-5.4-pro` | Enhanced quality |
| `gpt-5.4-mini` | Fast lightweight |
| `grok-41-fast` | Fast, up-to-date |
| `kimi-k2.6` | Long context |
| `kimi-k2-thinking` | Deep thinking |

## Coding

`gpt-5.3-codex`, `gpt-5.3-codex-spark`, `qwen3-coder-480b`, `qwen3-coder-480b-turbo`, `qwen3-coder-next-80b`, `mistralai/devstral-2-123b-instruct-2512`

## Minimax + Step

`minimax-m2.5`, `minimax-m2.7`, `minimax-m2.7-highspeed`, `minimax-highspeed`, `stepfun-ai/step-3.5-flash`

**Use `minimax-m2.7` for auxiliary slots (title_generation, compression)** — good quality, low cost, no streaming needed.

## GLM

`glm-5.1`, `glm-5v-turbo` (vision), `glm-4.7`, `glm-4.7-flash`, `glm-4-5-free`

## Qwen

`qwen3-235b-instruct`, `qwen3-235b-thinking`, `qwen3.5-9b`, `qwen3-vl-235b-a22b` (vision), `qwen3-next-80b`, `qwen3-5-397b-a17b`

## Small / Cheap

`gpt-4o-mini`, `gemma-4-31b`, `mistral-small-3.2`, `deepseek-v3.2`

## Free

`gpt-oss-120b-free`, `nemotron-120b-free`

## Open / Llama

`llama-3.3-70b`

## Reasoning

`arcee-trinity-large-thinking`

## Uncensored

`venice-uncensored`, `gemma-4-uncensored`

## E2EE (Privacy)

`e2ee-glm-4-7-flash-p`, `e2ee-glm-5-1`, `e2ee-gpt-oss-120b-p`, `e2ee-qwen3-5-122b-a10b`, `e2ee-qwen3-vl-30b-a3b-p`

## Specialist

`grok-4-20-multi-agent`, `mistralai/magistral-small-2506`, `mistralai/mistral-small-4-119b-2603`

---

## Selection Logic by Task Type

| Task Type | Recommended Category | Example Model |
|-----------|--------------------|---------------|
| `research` | Flagship (deep) | `claude-opus-4-7`, `deepseek-r1` |
| `code` | Coding or Flagship | `qwen3-coder-480b`, `gpt-5.3-codex` |
| `vision` | Flagship vision or GLM-5v | `gemini-3-flash`, `glm-5v-turbo` |
| `chat` | Flagship fast | `deepseek-v4-flash`, `gpt-5.4-mini` |
| `cheap` | Free or Small/Cheap | `gpt-oss-120b-free`, `gpt-4o-mini` |
| `any` | Best scored | Top-ranked by algorithm |

`best-peer.sh` scores candidates by: free bonus (+20), tag match (+10), protocol preference (chat_completions > responses), price penalty.