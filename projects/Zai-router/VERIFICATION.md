# Verification

Run commands from the Hermes Agent repository root.

## Focused regression suite

```bash
.venv/Scripts/python.exe -m pytest \
  tests/plugins/model_providers/test_zai_indirect_profile.py \
  tests/agent/test_zai_indirect_anthropic.py \
  tests/agent/test_anthropic_adapter.py \
  tests/plugins/model_providers/test_zai_profile.py -q
```

## No-network transport check

```bash
.venv/Scripts/python.exe -c "from run_agent import AIAgent; a=AIAgent(provider='zai-indirect', model='glm-5.2', api_key='test-placeholder', quiet_mode=True, enabled_toolsets=[], skip_context_files=True, skip_memory=True); print(a.provider, a.api_mode, a.base_url, type(a._anthropic_client).__name__)"
```

Expected routing fields:

```text
zai-indirect anthropic_messages https://api.z.ai/api/anthropic Anthropic
```

## Picker-catalogue check

```bash
.venv/Scripts/python.exe -c "from hermes_cli.model_switch import list_authenticated_providers; rows=list_authenticated_providers(current_provider='zai-indirect', max_models=50, probe_custom_providers=False, for_picker=True); print([(r['slug'], r['name'], r['models']) for r in rows if r['slug']=='zai-indirect'])"
```

Expected result:

```text
[('zai-indirect', 'Z.ai Indirect', ['glm-5.2'])]
```

## Live smoke test

A live call requires a valid credential in the active profile:

```bash
.venv/Scripts/python.exe -m hermes_cli.main chat \
  --provider zai-indirect \
  --model glm-5.2 \
  --query "Reply with exactly: ZAI_ROUTER_OK" \
  --max-turns 1 --quiet --ignore-rules --source tool
```

The final model response must be exactly `ZAI_ROUTER_OK`.
