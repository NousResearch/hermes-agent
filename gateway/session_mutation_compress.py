"""Summary preparation has no canonical store; publication owns the only write."""
import asyncio
import json

from hermes_state_runtime import RuntimeStoreError


async def prepare_compress(authority, live, payload, prepared):
    from gateway.session_policy import restore_policy, policy_scope
    from agent.context_compressor import ContextCompressor
    from agent.agent_init import _parse_config_int
    from utils import is_truthy_value
    snapshot = prepared['snapshot']
    policy = restore_policy(snapshot['receipt']['policy'])
    config = policy.config(authority)
    options = config.get('compression') or {}
    if options.get('checkpoint_required'):
        raise RuntimeStoreError('compression_checkpoint_required')
    model, runtime = authority.runner._resolve_session_agent_runtime(source=live.source, session_key=live.route)
    if runtime.get('api_mode') == 'codex_app_server':
        raise RuntimeStoreError('runtime_coordination_required')
    history = authority.db.get_messages_as_conversation(snapshot['target'])
    messages = [m for m in history if m.get('role') in {'user', 'assistant', 'tool'}]
    def summarize():
        with policy_scope(policy, authority=authority):
            compressor = ContextCompressor(model, base_url=runtime.get('base_url') or '',
                api_key=runtime.get('api_key') or '', provider=runtime.get('provider') or '',
                api_mode=runtime.get('api_mode') or '', quiet_mode=True, abort_on_summary_failure=True,
                protect_first_n=options.get('protect_first_n', 3), protect_last_n=options.get('protect_last_n', 20),
                min_tail_user_messages=max(1, _parse_config_int(options.get('min_tail_user_messages', 1), 1)),
                custom_providers=config.get('custom_providers'))
            compressed = compressor.compress(json.loads(json.dumps(messages)), force=True,
                                               focus_topic=payload.get('focus'))
            if not any(m.get('_compressed_summary') for m in compressed):
                raise RuntimeStoreError('nothing_to_compress')
            return compressed
    compressed = await asyncio.to_thread(summarize)
    return dict(prepared, messages=compressed, in_place=is_truthy_value(options.get('in_place'), default=True))
