---
name: psyche-monitor
description: Monitor and interact with the Psyche decentralized AI training network. Track training runs, mining pool status, model checkpoints on HuggingFace, and network health via Solana.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Psyche, Nous Research, Decentralized AI, Solana, Training, Mining Pool, HuggingFace]
    related_skills: []
---

# Psyche Network Monitor

This skill enables monitoring and interaction with the **Psyche decentralized AI training network**, built by Nous Research on the Solana blockchain.

## What is Psyche?

Psyche is a decentralized infrastructure that allows anyone to participate in training large language models. Instead of relying on centralized server farms, Psyche distributes training across independent computers worldwide, coordinated by Solana smart contracts.

**Key Technologies:**
- **DisTrO**: Reduces bandwidth requirements by 3x using DCT compression + 1-bit quantization
- **Overlapped Training**: Nodes train on the next step while sharing previous results
- **P2P via Iroh**: UDP hole-punching + QUIC protocol with ~90% direct connection rate

## Available Actions

### 1. List Training Runs
```
Use the psyche_monitor tool with action "list_runs"
```
Returns all known training runs with their status, plus latest models from the PsycheFoundation HuggingFace organization.

### 2. Run Details
```
Use the psyche_monitor tool with action "run_details" and run_id "consilience-40b-1"
```
Get detailed information about a specific training run including model architecture, dataset, and HuggingFace checkpoint data.

### 3. Model Checkpoints
```
Use the psyche_monitor tool with action "checkpoints"
```
Lists model checkpoint files from HuggingFace. Consilience 40B checkpoints are auto-uploaded every 500 training steps.

### 4. Mining Pool Status
```
Use the psyche_monitor tool with action "pool_status"
```
Shows how the mining pool works, smart contract features, and how to contribute funds to training runs.

### 5. Network Statistics
```
Use the psyche_monitor tool with action "network_stats"
```
Returns network overview including Solana health status, ecosystem links, funding info, and active models.

### 6. Contribution Guide
```
Use the psyche_monitor tool with action "contribute"
```
Comprehensive guide on how to contribute: compute power, mining pool funds, code contributions, Atropos environments, and community engagement.

## Key Links

| Resource | URL |
|----------|-----|
| Dashboard | https://psyche.network |
| Documentation | https://docs.psyche.network |
| GitHub | https://github.com/PsycheFoundation/psyche |
| Forum | https://forum.nousresearch.com |
| HuggingFace | https://huggingface.co/PsycheFoundation |

## Consilience 40B - Flagship Model

The first major Psyche training run:
- **40B parameters** with MLA (Multi-head Latent Attention) architecture
- **20 trillion tokens** from FineWeb + FineWeb-2 + The Stack V2
- Largest distributed pre-training run ever conducted over the internet
- Dense model (not MoE) based on DeepSeek V3 architecture
- Auto-updated checkpoints on HuggingFace every 500 steps

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Mining pool full | Check back frequently - pool capacity fluctuates |
| Wallet won't connect | Ensure you're using a Solana-compatible wallet (Phantom, Solflare) |
| No testnet access | Testnet is invite-only during early phase - watch Discord for openings |
| Checkpoint not loading | Model requires significant VRAM - minimum 3090 for inference |
