# Prompt Builder Integration

- Location: ~/hermes-workspace/Lex-Workspace/scripts/prompt-builder.py
- Injects identity, skills, global rules, and emits two headers parsed by Lex's dispatch wrapper:
  - HERMES-MODEL: {provider, model, base_url?} per agent from ~/.hermes/workspace/config/model-config.json
  - HERMES-EXPERIMENT: {recommendation_id, variant, run_id, dispatched_at} for SIE Routing Intel Phase 3 A/B dispatches
- Companion rule (AGENTS.md HR-2): delegate_task must consume the parsed model override instead of inheriting the parent model.
- ALE wiring: the experiment header’s run_id is threaded into ale-cli start-run to preserve the 1:1 join-key invariant across experiments.jsonl and ALE runs.
