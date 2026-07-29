# Reasoning, Self-Reflection & Prompt Customization

## When to Use
User wants to enable reasoning/thinking mode, configure self-reflection, customize prompts, adjust generation parameters (max tokens, temperature, citations), or understand thinking budget options.

## Process
1. Detect the deployment mode (Docker / Helm / Library). Docker: edit the active env file. Helm: edit `values.yaml`. Library: edit `notebooks/config.yaml`
2. Read the relevant source doc for the specific feature
3. Apply env vars to the active config or edit prompt files, restart RAG server
4. Prompt changes require `--build` flag (Docker); env var changes only need restart
5. Verify: test with a query and check for reasoning output or changed behavior

## Decision Table

| Goal | Source Doc | Key Action |
|------|-----------|------------|
| Enable reasoning (Nemotron 1.5) | `docs/enable-nemotron-thinking.md` | Edit `prompt.yaml`: `/no_think` → `/think`, set temperature |
| Enable reasoning (Nano 30B) | `docs/enable-nemotron-thinking.md` | `ENABLE_NEMOTRON_3_NANO_THINKING=true` |
| Self-reflection | `docs/self-reflection.md` | `ENABLE_REFLECTION=true`, set thresholds |
| Prompt customization | `docs/prompt-customization.md` | `PROMPT_CONFIG_FILE=/path/to/custom.yaml` or edit prompt.yaml |
| Generation parameters | `docs/llm-params.md` | `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`, `ENABLE_CITATIONS` |
| Per-request overrides | `docs/llm-params.md` | `temperature`, `top_p`, `max_tokens`, `stop` in API payload |

## Agent-Specific Notes

- Prompt changes need `--build` flag on restart; env var changes do not
- Self-reflection: streaming not supported during groundedness checks
- Self-reflection uses same LLM by default; override with `REFLECTION_LLM`, `REFLECTION_LLM_SERVERURL`, `REFLECTION_LLM_APIKEY`
- Helm: only on-premises reflection is supported
- GPU requirements for reflection: see `docs/self-reflection.md` for optimal GPU configurations
- Debug reflection: set `LOGLEVEL=INFO` to observe iteration counts
- `FILTER_THINK_TOKENS=false` to see full reasoning output (filtered by default)
- 18 prompt templates available in `prompt.yaml` — custom file only overrides specified keys

### Reasoning Model Comparison

| Model | Control | Thinking Budget | Output Format |
|-------|---------|-----------------|---------------|
| Nemotron 1.5 | System prompt (`/think`) | None | `<think>` tags (filtered by default) |
| Nemotron-3-Nano 9B | System prompt (`/think`) | `min_thinking_tokens` + `max_thinking_tokens` | `reasoning_content` field |
| Nemotron-3-Nano 30B | `ENABLE_NEMOTRON_3_NANO_THINKING` env var | `max_thinking_tokens` only | `reasoning_content` field |

### Thinking Budget Recommendations

| Range | Use Case |
|-------|----------|
| 1024–4096 | Faster responses for simpler questions |
| 8192–16384 | More thorough reasoning for complex queries |

## Notebooks
- `notebooks/retriever_api_usage.ipynb` — end-to-end query examples showing generation behavior

## Source Documentation
- `docs/enable-nemotron-thinking.md` — Reasoning mode for all Nemotron models
- `docs/self-reflection.md` — Self-reflection configuration and thresholds
- `docs/prompt-customization.md` — Prompt template catalog and customization
- `docs/llm-params.md` — Generation parameters (temperature, max tokens, etc.)
