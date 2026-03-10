---
name: tinker
description: Run LLM post-training on Tinker with CPU-side orchestration and remote GPU execution. Use when preparing or launching SFT, DPO, or PPO-style runs, checkpointing, sampling checkpoints, or resuming long-running jobs through Hermes Research Agent.
version: 1.0.0
author: Hermes Research Agent
license: MIT
metadata:
  hermes:
    tags: [tinker, training, sft, dpo, ppo, checkpoints, lora]
---

# Tinker

Tinker is the only compute backend in Hermes Research Agent v1.

## Requirements

- `TINKER_API_KEY` must be set.
- Use `tinker_posttrain` for lifecycle management instead of ad hoc shell commands when possible.

## Supported methods

- `sft`: dataset rows need `prompt` and `completion`
- `dpo`: dataset rows need `prompt`, `chosen`, and `rejected`
- `ppo`: dataset rows need `prompt`, `completion`, token-level `logprobs`, and either `advantage` or `reward`

## Default operating pattern

1. Validate the config with `tinker_posttrain(action="validate_config", ...)`.
2. Start the run with `tinker_posttrain(action="start_run", ...)`.
3. Let `research_loop(action="monitor_run", ...)` manage long-running polling and resumption.
4. Use `tinker_posttrain(action="sample_checkpoint", ...)` for quick qualitative checks.
5. Use `tinker_posttrain(action="download_checkpoint", ...)` only when the project needs local checkpoint artifacts.

## Run hygiene

- Always tie the run to an experiment id and a written hypothesis.
- Save checkpoints regularly.
- Record the outcome with `research_state(action="record_result", ...)`.
- If approval mode is active, expect `start_run`, `resume_run`, and `stop_run` to require an approval record.
