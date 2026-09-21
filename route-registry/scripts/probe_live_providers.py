import json, os, sys
from pathlib import Path
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')
os.environ.setdefault('HERMES_HOME', '/home/kensei/.hermes')

from agent.auxiliary_client import resolve_provider_client  # noqa: E402

# model per provider: xkiro-free uses deepseek flash, xkiro-pro uses deepseek pro,
# bai uses glm-5.3-flash (all verified present in catalogues)
CASES = [
    ('custom:xkiro-free', 'deepseek/deepseek-v4-flash', 'https://api.xkiro.com/v1'),
    ('custom:xkiro-pro', 'deepseek/deepseek-v4-pro', 'https://api.xkiro.com/v1'),
    ('custom:bai', 'glm-5.3-flash', 'https://api.b.ai/v1'),
]
for provider, model, base_url in CASES:
    try:
        client, resolved_model = resolve_provider_client(
            provider=provider,
            model=model,
            explicit_base_url=base_url,
            explicit_api_key=None,
        )
        # prove the header reached the client
        hdrs = getattr(client, '_custom_headers', None) or getattr(client, 'default_headers', {})
        has_ua = any('User-Agent' in str(k) for k in hdrs) if isinstance(hdrs, dict) else False
        # fire a real tiny completion
        resp = client.chat.completions.create(
            model=resolved_model or model,
            messages=[{'role': 'user', 'content': 'Reply with the single word OK.'}],
            max_tokens=8,
            timeout=40,
        )
        text = (resp.choices[0].message.content or '')[:20]
        print(f'{provider}: OK model={resolved_model or model} ua_header={has_ua} reply={text!r}')
    except Exception as exc:
        print(f'{provider}: FAIL {type(exc).__name__}: {str(exc)[:160]}')
