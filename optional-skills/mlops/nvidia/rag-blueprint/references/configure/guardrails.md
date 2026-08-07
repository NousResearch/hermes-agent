# NeMo Guardrails

## When to Use
- User wants content safety, topic control, or jailbreak prevention
- User asks to enable/disable guardrails

## Restrictions
- Not available on B200 GPUs
- Requires 2 extra GPUs with 48GB+ each (H100, A100 SXM 80GB, or RTX PRO 6000)
- Not supported in library mode or Helm deployments
- Jailbreak detection model not yet available out-of-the-box

## Process

1. Detect the deployment mode (guardrails are Docker-only — not supported on Helm or library mode). Edit the active env file for Docker
2. Read `docs/nemo-guardrails.md` for full setup and configuration
3. Choose deployment mode: self-hosted (local NIMs) or cloud-hosted (NVIDIA API)
4. For self-hosted: assign GPU IDs — read `docs/service-port-gpu-reference.md` for default GPU assignments and adjust for your system
5. Verify all three services healthy: `nemo-guardrails-microservice`, content-safety NIM, topic-control NIM
6. Enable in UI: Settings > Output Preferences > Guardrails toggle

## Agent-Specific Notes
- Cloud mode (`nemoguard_cloud` config) skips local NIM containers — only the microservice is needed
- Per-request toggle via `enable_guardrails` in `/generate` body requires server-level `ENABLE_GUARDRAILS=true` first
- Override guardrails URL with `NEMO_GUARDRAILS_URL` if running on a different host
- Content-safety and topic-control models are trained on single-turn data — multi-turn conversations may get inconsistent safety classifications
- Current guardrails only produce simple refusal responses ("I'm sorry. I can't respond to that.")

## Source Documentation
- `docs/nemo-guardrails.md` -- full setup, configuration, and customization of guardrail rules
