"""Opt-in adapter metadata for durable native events, never delegated authorization."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import logging
import math
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_BYTES = 16 * 1024
_MAX_DEPTH = 8
_MAX_NODES = 1024


def project_adapter_metadata(event):
    """Return an isolated, bounded JSON object, or None for a refused shape."""
    metadata = event.metadata
    if type(metadata) is not dict:
        return None
    prefix = event.source.platform.value + '_'
    if any(type(key) is not str or not key.startswith(prefix) or key == prefix for key in metadata):
        return None
    try:
        _check_json(metadata, 0, [0])
        encoded = json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode('utf-8')) > _MAX_BYTES:
            return None
        return json.loads(encoded)
    except (TypeError, ValueError, UnicodeError, RecursionError, OverflowError):
        return None


def _check_json(value, depth, count):
    count[0] += 1
    if count[0] > _MAX_NODES or depth > _MAX_DEPTH:
        raise ValueError('metadata budget')
    kind = type(value)
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError('metadata key')
            _check_json(key, depth + 1, count)
            _check_json(item, depth + 1, count)
    elif kind is list:
        for item in value:
            _check_json(item, depth + 1, count)
    elif kind is str:
        if len(value) > _MAX_BYTES:
            raise ValueError('metadata string')
    elif kind is float:
        if not math.isfinite(value):
            raise ValueError('metadata number')
    elif value is not None and kind not in (bool, int):
        raise ValueError('metadata value')


def validate_adapter_metadata(runner, event, provenance=None):
    """Run a pure validator on the currently bound adapter, with no raw replay data.

    The callback gets its own event copy: mutation cannot rewrite the admitted payload.
    No callback (or anything other than True) is a refusal. Exceptions fail closed.
    """
    projected = project_adapter_metadata(event)
    if projected is None or runner is None:
        return False
    adapter = runner._adapter_for_source(event.source)
    check = getattr(type(adapter), 'validate_native_metadata', None)
    if check is None:
        return False
    from gateway.run import _profile_runtime_scope
    try:
        # Pure synchronous checks reuse the owner's snapshot; external secret
        # providers hydrate at async lifecycle boundaries, never on this loop.
        scope = (_profile_runtime_scope(Path(provenance['transport_home']), hydrate_secrets=False)
                 if provenance is not None else nullcontext())
        # Dataclass replacement drops process-local source attributes (transport refs,
        # locks, callback contexts); only detached data reaches the pure validator.
        isolated = deepcopy(replace(event, source=replace(event.source),
                                    metadata=projected, raw_message=None))
        with scope:
            accepted = check(adapter, isolated)
        # A synchronous validator cannot borrow another connector's ownership.
        return accepted is True and runner._adapter_for_source(event.source) is adapter
    except Exception:
        logger.exception('Adapter native metadata validation failed')
        return False
