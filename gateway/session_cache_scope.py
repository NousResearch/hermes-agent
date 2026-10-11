"""Explicit single-query cache affinity, separate from every session/history identity."""
import hashlib
import json
from pathlib import Path
import re


def validate_cache_scope(params):
    """Clients send only a digest, never the user's label; ordinary launches send nothing."""
    if 'cache_scope' not in params:
        return
    value = params['cache_scope']
    if (params.get('source', 'cli') not in {'cli', 'tool', 'oneshot'}
            or not isinstance(value, str) or not re.fullmatch(r'cli_[0-9a-f]{32}', value)):
        from hermes_state_runtime import RuntimeStoreError
        raise RuntimeStoreError('invalid_params', field='cache_scope',
                                detail='requires a CLI-surface launch and a namespaced digest')


def bind_session_cache_scope(agent, policy):
    """Bind only the turn's parent, never all agents in the ambient policy context.

    Re-namespace at the owning gateway's profile/workspace boundary, including on a
    managed worker. Neither physical session IDs nor memory-provider routes change.
    """
    request = json.loads(policy.request_json) if policy else {}
    scope = request.get('cache_scope')
    if scope is not None:
        from hermes_constants import get_hermes_home
        identity = [str(get_hermes_home().resolve()), str(Path(policy.cwd).resolve()), scope]
        digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode('utf-8')).hexdigest()[:32]
        agent._cli_prompt_cache_scope = f'cli_{digest}'
    return agent
