"""Presentation and durable handoff of an admitted webhook's actual final."""
import hashlib
import json
from pathlib import Path

from gateway.platforms.base import SendResult
from gateway.response_filters import is_autonomous_silence_response


def result_id(home, admission_id):
    # Keep the original native receipt domain across upgrades. The intent key is separate.
    identity = ['webhook-final-v1', str(Path(home).resolve()), admission_id]
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def wrapped_message(route, content):
    from agent.redact import redact_sensitive_text
    # A broken redactor refuses delivery; it must not fall back to raw text or a
    # placeholder that can be mistaken for successfully completed source work.
    route = redact_sensitive_text(route, force=True)
    content = redact_sensitive_text(content, force=True)
    if not isinstance(route, str) or not isinstance(content, str):
        raise ValueError('redaction unavailable')
    return (f'[Webhook automation "{route}" output: not the user. '
            f'Review it, act on anything that needs action, and summarize for the chat.]\n\n{content}')


def presentation(result, outcome, *, muted=False, category='result', response=None):
    """Freeze the source's presentation decision, separately from its actual task outcome."""
    from gateway.config import Platform
    from gateway.warning_notifications import warning_notifications_enabled
    text = result.get('final_response') if response is None else response
    category = 'diagnostic' if outcome in {'failed', 'interrupted', 'cancelled'} else category
    if outcome == 'unknown':
        return dict(disposition='ambiguous', category=category, text='', reason='source_unknown')
    if muted or (category == 'diagnostic' and not warning_notifications_enabled(Platform.WEBHOOK)):
        return dict(disposition='suppressed', category=category, text='', reason='source_policy')
    if not isinstance(text, str) or not text.strip() or is_autonomous_silence_response(text):
        return dict(disposition='suppressed', category=category, text='', reason='empty_or_silent')
    return dict(disposition='pending', category=category, text=text)


async def deliver_settled(authority, adapter, event, admission_id):
    """None for another destination; otherwise expose real intent/native state, not 202."""
    from gateway.webhook_final_intent import reconcile_final
    try:
        value = await reconcile_final(authority, adapter, admission_id)
        if value is None:
            return None
        state = value['state']
        return SendResult(success=state in {'queued', 'claimed', 'settled', 'suppressed'},
            message_id=value['delivery_id'], error=None if state in {'queued', 'claimed', 'settled', 'suppressed'}
            else 'Webhook Bot Chat ' + state,
            raw_response={'status': state, 'delivery_id': value['delivery_id'], 'origin_kind': value['origin_kind'],
                          'source_admission_id': admission_id, 'source_outcome': value['source_outcome'],
                          'native_status': value.get('native_status'), 'reason': value.get('reason')})
    except Exception:  # health: allow BLE001 -- source/transport failures may contain private text
        return SendResult(success=False, error='Webhook final unavailable or unauthorized')
