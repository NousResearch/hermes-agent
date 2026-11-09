"""Stable presentation identity for a durable Responses admission.

The model label, creation time and compact ID/index ordering survive response-body eviction.
Item IDs derive from response identity and occurrence; no additional transcript/output copy is retained.
"""
import hashlib
import json

_PREFIXES = {'function_call': 'fc', 'function_call_output': 'fco', 'reasoning': 'rs', 'message': 'msg'}


def durable_response_id(request_key):
    """The public response id of an Idempotency-Key request; the store derives it again for an
    identity whose replay record was never written."""
    return 'resp_' + hashlib.sha256(request_key.encode()).hexdigest()[:28]


def next_item_id(response_id, counters, kind, *, call_id=None, commentary=False):
    # The final message is distinct from any live commentary, which is not part of the answer.
    key = (kind, call_id, bool(commentary))
    occurrence = counters.get(key, 0)
    counters[key] = occurrence + 1
    digest = hashlib.sha256(json.dumps([response_id, *key, occurrence], separators=(',', ':')).encode()).hexdigest()
    return _PREFIXES[kind] + '_' + digest[:24]


def identify_output(items, response_id):
    counters = {}
    return [dict(item, id=next_item_id(response_id, counters, item['type'],
                                      call_id=item.get('call_id'), commentary=item.get('phase') == 'commentary'))
            for item in items]
