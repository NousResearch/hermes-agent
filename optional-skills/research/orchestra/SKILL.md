---
name: orchestra-research
description: Gateway to 94 research engineering skills from Orchestra.
version: 1.0.0
platforms: [linux, macos]
author: Orchestra Research
license: MIT
metadata:
  hermes:
    tags: [Research, Engineering, ML, Training, Inference]
    category: research
---

# Orchestra Research Gateway

Use when asked about model architecture, distributed training (FSDP2, DeepSpeed), post-training (GRPO, DPO), high-throughput inference (vLLM, SGLang), mechanistic interpretability, or any research engineering workflow.

This skill is a gateway to the **Orchestra Research Skills** library. Instead of bundling 94 skills, it indexes them across 21 domains and fetches what you need on demand.

## Sources

Original repository: `https://github.com/NousResearch/orchestra-research-skills`
94 SKILL.md files across 21 research domains.

## How to fetch and use a skill

1. Identify the domain from the index below.
2. Use the `terminal` tool to clone the repo:
   `git clone --depth 1 https://github.com/NousResearch/orchestra-research-skills /tmp/orchestra-skills`
3. Use the `read_file` tool to inspect the skill:
   Path: `/tmp/orchestra-skills/skills/<skill-name>/SKILL.md`
4. Apply the fetched patterns to the current task.

## Skill Index

| Domain | Skills |
|--------|--------|
| 01-model-architecture | nanogpt, mamba, moe-training, rwkv, transformer-lens |
| 02-tokenization | huggingface-tokenizers, sentencepiece |
| 03-fine-tuning | axolotl, llama-factory, peft, unsloth |
| 04-mechanistic-interpretability | nnsight, pyvene, saelens |
| 05-data-processing | nemo-curator, ray-data |
| 06-post-training | grpo-rl-training, knowledge-distillation, model-merging, openrlhf, simpo, trl-fine-tuning |
| 07-safety-alignment | constitutional-ai, llamaguard, prompt-guard |
| 08-distributed-training | accelerate, deepspeed, megatron-core, pytorch-fsdp2, torchforge, torchtitan |
| 09-infrastructure | chroma, faiss, pinecone, qdrant, sentence-transformers |
| 10-optimization | awq, bitsandbytes, flash-attention, gguf, gptq, hqq, tensorrt-llm, speculative-decoding |
| 11-inference | litgpt, llama-cpp, openvla-oft, sglang, vllm |
| 12-training-tools | mlflow, ml-training-recipes, pytorch-lightning, skypilot, swanlab, tensorboard, weights-and-biases |
| 13-multimodal | blip-2, clip, llava, segment-anything, stable-diffusion, whisper |
| 14-agents | autogpt, crewai, dspy, guidance, instructor, langchain, llamaindex, outlines |
| 15-evaluation | bigcode-evaluation-harness, lm-evaluation-harness, nemo-evaluator, phoenix, langsmith |
| 16-compute | lambda-labs, modal |
| 17-ml-paper-writing | academic-plotting, ml-paper-writing, presenting-conference-talks, systems-paper-writing |
| 18-research-ideation | brainstorming-research-ideas, creative-thinking-for-research |
| 19-robotics | cosmos-policy, openpi |
| 20-multimodal-generation | audiocraft, logo-design |
| 21-long-context | long-context |

## Pitfalls

- The upstream repo may update. If paths change, use the `terminal` tool with `ls` to discover the current structure.
- This gateway indexes skills by domain. For specific skill names, use `search_files` on the cloned repo.
- Always clone with `--depth 1` to minimize disk usage and transfer time.

## Verification

Run: `hermes skills inspect orchestra-research`
