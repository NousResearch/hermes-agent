"""Optional route presentation from OpenAI-compatible catalogue metadata.

No inference/lifecycle probes. Only explicit opt-in providers receive a bounded,
background GET /models. Route IDs and provider identities remain unchanged.
"""
from __future__ import annotations

import json
import math
import threading
import time
import unicodedata
import urllib.request

from hermes_cli.providers import custom_provider_aliases

_MAX_BYTES = 65536
_MAX_AGE = 15
_STATES = {'ready': '● Ready', 'loading': '◐ Loading', 'idle': '○ Idle / unloaded',
           'error': '! Error', 'unavailable': '! Unavailable', 'unknown': '? Unknown'}


def _text(value, default=''):
    if not isinstance(value, str):
        return default
    return ''.join(c for c in value if not unicodedata.category(c).startswith('C'))[:120].strip() or default


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch_metadata(entry):
    url = str(entry.get('base_url') or entry.get('url') or '').rstrip('/') + '/models'
    headers = dict(entry.get('extra_headers') or {})
    key = entry.get('api_key')
    if key:
        headers['Authorization'] = f'Bearer {key}'
    request = urllib.request.Request(url, headers=headers, method='GET')
    # Do not forward endpoint credentials through an untrusted redirect.
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(request, timeout=1.0) as response:
        raw = response.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get('data'), list):
        return {}
    return {m['id']: m['metadata'] for m in data['data'] if isinstance(m, dict)
            and isinstance(m.get('id'), str) and isinstance(m.get('metadata'), dict)}


def attach_picker_presentation(rows, user_providers=None, custom_providers=None, *, invalidate=None):
    """Attach static labels immediately; refresh optional metadata off the UI thread.

    Each open owns its rows and worker results. An older open cannot overwrite a
    new picker. Expiry invalidates residency, not the published backing identity.
    """
    entries = [dict(value, name=key) for key, value in (user_providers or {}).items() if isinstance(value, dict)]
    entries += [e for e in (custom_providers or []) if isinstance(e, dict)]
    workers = []
    for row in rows:
        for entry in entries:
            aliases = set(custom_provider_aliases(entry.get('name', ''), entry.get('provider_key', '')))
            aliases.add(str(entry.get('name', '')).lower())
            if str(row.get('slug', '')).lower() not in aliases:
                continue
            url = str(entry.get('base_url') or entry.get('url') or '').rstrip('/')
            if url != str(row.get('api_url') or row.get('base_url') or '').rstrip('/'):
                continue
            models = entry.get('models')
            labels = {key: _text(value.get('picker_label')) for key, value in models.items()
                      if isinstance(value, dict) and _text(value.get('picker_label'))} if isinstance(models, dict) else {}
            if labels or entry.get('picker_metadata') is True:
                row['picker_presentation'] = {'labels': labels, 'models': {}, 'observed_at': None}
            if entry.get('picker_metadata') is True:
                def refresh(target=row, config=dict(entry), static=labels):
                    try:
                        metadata = _fetch_metadata(config)
                    except Exception:
                        metadata = {}
                    target['picker_presentation'] = {'labels': static, 'models': metadata,
                                                     'observed_at': time.monotonic()}
                    if invalidate is not None:
                        invalidate()
                worker = threading.Thread(target=refresh, daemon=True, name='picker-metadata')
                worker.start()
                workers.append(worker)
            break
    return workers


def _nonnegative_number(value):
    # JSON integers can exceed float range; compare before math converts them.
    return type(value) in (int, float) and 0 <= value <= 1.7976931348623157e308 and math.isfinite(value)


def _fresh_residency(details, received_at):
    """A new GET cannot renew an old observation; no cross-host clock subtraction.

    Older metadata without an age contract retains its identity but cannot
    establish current residency. The UI also caps observation age at 15 seconds.
    """
    freshness = details.get('freshness')
    if not isinstance(freshness, dict) or freshness.get('stale') is not False:
        return False
    age, maximum = freshness.get('age_s'), freshness.get('max_age_s')
    if not all(_nonnegative_number(v) for v in
               (age, maximum, details.get('observed_at'), received_at)) or maximum <= 0:
        return False
    elapsed = time.monotonic() - received_at
    return 0 <= elapsed and age + elapsed <= min(maximum, _MAX_AGE)


def route_fields(row, model):
    presentation = row.get('picker_presentation') or {}
    details = presentation.get('models', {}).get(model) or {}
    if not isinstance(details, dict):
        details = {}
    role = details.get('role')
    label = presentation.get('labels', {}).get(model)
    if not label and role in ('auto', 'main', 'aux'):
        label = _text(row.get('name'), 'Provider') + ':' + role.title()
    if not label:
        return None
    backing = _text(details.get('backing_model'), 'Unknown model')
    residency = details.get('residency') if _fresh_residency(details, presentation.get('observed_at')) else 'unknown'
    state = _STATES.get(_text(residency), _STATES['unknown'])
    # Role is not residency: shared auxiliary metadata carries the main backing.
    if details.get('mode') == 'shared-main':
        state = 'Shared main · ' + state
    return label, backing, state


def _friendly_backing(model):
    """Configured friendly name for a stable route when no live metadata exists."""
    try:
        from cli import _reverse_alias_for_display
        friendly = _reverse_alias_for_display(model)
        return friendly if friendly and friendly != model else None
    except Exception:
        return None


def model_label(row, model):
    fields = route_fields(row, model)
    if not fields:
        return model
    label, backing, state = fields
    if backing == 'Unknown model':
        friendly = _friendly_backing(model)
        if friendly:
            return f"{label} · {friendly}" + (f" · {state}" if state != '? Unknown' else '')
        # Without live metadata the backing identity is unknowable; rendering the raw ID
        # adds nothing (the role label IS the route), so show label + state only.
        return f"{label} · {state}" if state != '? Unknown' else label
    return ' · '.join(fields)
