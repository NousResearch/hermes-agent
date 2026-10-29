"""Required, load-owned outbound policy seam. No domain identities or regexes.

Policies receive data only, never a send handle. Missing, disabled, broken,
slow, invalid or generation-changed required policies deny before transport.
Required declarations belong to host configuration, not plugin enablement.
"""
from dataclasses import dataclass, field
from types import MappingProxyType
from collections.abc import Mapping
from contextvars import ContextVar
from functools import wraps
import copy
import inspect
import threading
import queue
import logging

logger = logging.getLogger(__name__)
REGISTRY_WAIT_SECONDS = 5.0
POLICY_TIMEOUT_SECONDS = 0.25
_CAPACITY = threading.BoundedSemaphore(4)
_PIN = ContextVar('outbound_policy_pin', default=None)


class OutboundDenied(RuntimeError):
    pass


class OutboundHeld(OutboundDenied):
    def __init__(self, reference):
        super().__init__("outbound held locally, not sent")
        self.reference = reference


@dataclass(frozen=True)
class OutboundEnvelope:
    platform: str
    destination: str
    content: str
    reply_to: object = None
    metadata: Mapping = field(default_factory=lambda: MappingProxyType({}))
    kind: str = 'send'
    references: tuple = ()


@dataclass(frozen=True)
class OutboundDecision:
    action: str
    target_chat_id: str
    reply_to: object = None
    metadata: Mapping = field(default_factory=dict)
    route_class: str = 'policy'
    reason: str = 'allowed'
    original_chat_id: str = ''

    @property
    def rerouted(self):
        return self.original_chat_id != self.target_chat_id


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise OutboundDenied('unsupported envelope metadata')


def _snapshot(platform):
    from hermes_cli.config import load_config
    from hermes_cli.plugins import get_plugin_manager
    from hermes_cli.plugins_loader import in_plugin_load_worker
    name, stage = '<configuration>', 'configuration'
    try:
        cfg = load_config()
        names = cfg.get('outbound', {}).get('required_policies', {}).get(platform, [])
        if not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names) or len(names) != len(set(names)):
            raise ValueError()
        manager = get_plugin_manager()
        if not names:
            return manager, ()
        name, stage = ','.join(names), 'discovery'
        # register() runs on a deadline worker whose parent owns the discovery
        # lock. Never wait for our own parent, nor authorize a partial load.
        if in_plugin_load_worker():
            raise RuntimeError('outbound policy during plugin registration')
        # Forced discovery unloads all load-owned policies before re-registering.
        # Share its lock through initialization AND the entire snapshot read;
        # get_plugin_manager() alone neither loads nor waits for discovery.
        if not manager._discovery_lock.acquire(timeout=REGISTRY_WAIT_SECONDS):
            raise TimeoutError('plugin discovery busy')
        try:
            manager.discover_and_load()
            slots = []
            for name in names:
                stage = 'registration'
                owner, provider = manager._outbound_policies[name]
                info = manager._plugins[owner]
                stage = 'enablement'
                if not info.enabled or info.error or owner not in cfg.get('plugins', {}).get('enabled', []):
                    raise ValueError()
                slots.append((name, owner, provider))
            return manager, tuple(slots)
        finally:
            manager._discovery_lock.release()
    except Exception as exc:
        # No content, destination, command, credentials or arbitrary exception
        # text: this diagnostic also serves approval-card failures.
        logger.error('required outbound policy unavailable platform=%s policy=%s stage=%s cause=%s',
                     platform, name, stage, type(exc).__name__)
        raise OutboundDenied('required outbound policy unavailable') from exc


def _same(a, b):
    return a[0] is b[0] and len(a[1]) == len(b[1]) and all(
        x[:2] == y[:2] and x[2] is y[2] for x, y in zip(a[1], b[1]))


def bounded_call(callback, *args):
    # Bounded daemon workers: timeout never leaves a transport-capable callback,
    # blocks future sends on saturation and cannot hang interpreter shutdown.
    if not _CAPACITY.acquire(blocking=False):
        raise OutboundDenied('outbound policy capacity exhausted')
    result = queue.Queue(maxsize=1)
    def run():
        try:
            result.put((True, callback(*args)))
        except BaseException:
            result.put((False, None))
        finally:
            _CAPACITY.release()
    threading.Thread(target=run, daemon=True).start()
    try:
        okay, decision = result.get(timeout=POLICY_TIMEOUT_SECONDS)
    except queue.Empty:
        raise OutboundDenied('outbound policy timed out') from None
    if not okay:
        raise OutboundDenied('policy callback failed')
    return decision


def _evaluate(provider, envelope):
    decision = bounded_call(provider.evaluate, envelope)
    if type(decision) is not OutboundDecision or decision.action not in ('allow', 'hold') or not isinstance(decision.target_chat_id, str) or not decision.target_chat_id or not isinstance(decision.metadata, Mapping):
        raise OutboundDenied('outbound policy denied or invalid')
    if decision.action == 'hold':
        if not isinstance(decision.reason, str) or not decision.reason:
            raise OutboundDenied('invalid hold reference')
        raise OutboundHeld(decision.reason)
    return decision


def enforce_outbound(platform, chat_id, content, reply_to=None, metadata=None, *, kind='send', references=()):
    platform = str(getattr(platform, 'value', platform))
    snapshot = _snapshot(platform)
    pin = _PIN.get()
    if pin is not None and pin[0] == platform and not _same(pin[1], snapshot):
        raise OutboundDenied('outbound policy generation changed')
    original = str(chat_id)
    destination, reply, meta = original, reply_to, dict(metadata or {})
    last = OutboundDecision('allow', destination, reply, meta, original_chat_id=original)
    for _, _, provider in snapshot[1]:
        envelope = OutboundEnvelope(platform, destination, str(content or ''), reply,
                                  _freeze(meta), kind, tuple(references))
        last = _evaluate(provider, envelope)
        destination, reply, meta = last.target_chat_id, last.reply_to, dict(last.metadata)
        if not _same(snapshot, _snapshot(platform)):
            raise OutboundDenied('outbound policy generation changed')
    final = OutboundDecision('allow', destination, reply, meta, last.route_class, last.reason, original)
    return destination, reply, meta, final


def guard_adapter_methods(cls):
    """Host-owned common public send/edit boundary, including media overrides.

    A request-local pin spans every chunk/media await. No plugin monkeypatching.
    Existing transport-specific choke points may enforce again on exact text.
    """
    for name, method in list(cls.__dict__.items()):
        if not (name.startswith('send') or name.startswith('edit_')) or not inspect.iscoroutinefunction(method):
            continue
        signature = inspect.signature(method)
        if 'chat_id' not in signature.parameters:
            continue
        def wrap(fn, sig):
            @wraps(fn)
            async def guarded(self, *args, **kwargs):
                bound = sig.bind(self, *args, **kwargs)
                bound.apply_defaults()
                platform = str(getattr(self.platform, 'value', self.platform))
                token = None
                if _PIN.get() is None:
                    token = _PIN.set((platform, _snapshot(platform)))
                try:
                    content = '\n'.join(str(bound.arguments.get(k) or '') for k in ('content', 'caption', 'message', 'prompt', 'title'))
                    refs = tuple(str(bound.arguments[k]) for k in ('file_path', 'image_path', 'audio_path', 'video_path', 'image_url', 'animation_url', 'images') if bound.arguments.get(k))
                    # Reference text is supplied too: absent captions are not approval.
                    destination, reply, meta, _ = enforce_outbound(platform, bound.arguments['chat_id'], content,
                        bound.arguments.get('reply_to'), bound.arguments.get('metadata'), kind=fn.__name__, references=refs)
                    bound.arguments['chat_id'] = destination
                    if 'reply_to' in sig.parameters:
                        bound.arguments['reply_to'] = reply
                    if 'metadata' in sig.parameters:
                        bound.arguments['metadata'] = meta
                    return await fn(*bound.args, **bound.kwargs)
                finally:
                    if token is not None:
                        _PIN.reset(token)
            return guarded
        setattr(cls, name, wrap(method, signature))


def pin_egress(platform):
    def decorator(fn):
        @wraps(fn)
        async def pinned(*args, **kwargs):
            token = None
            if _PIN.get() is None:
                token = _PIN.set((platform, _snapshot(platform)))
            try:
                return await fn(*args, **kwargs)
            finally:
                if token is not None:
                    _PIN.reset(token)
        return pinned
    return decorator


async def guarded_transport_call(platform, method, *args, **kwargs):
    """Last pre-network generation check; transport-specific routing already ran."""
    snapshot = _snapshot(platform)
    pin = _PIN.get()
    if pin is not None and pin[0] == platform and not _same(pin[1], snapshot):
        raise OutboundDenied('outbound policy generation changed before transport')
    if pin is None and snapshot[1]:
        text = str(kwargs.get('text') or kwargs.get('caption') or '')
        target, _reply, _, decision = enforce_outbound(platform, kwargs.get('chat_id', ''), text, kwargs.get('reply_to_message_id'), kind=getattr(method, '__name__', 'transport'))
        if decision.rerouted:
            if getattr(method, '__name__', '').startswith('edit'):
                raise OutboundDenied('cannot retarget an existing message edit')
            kwargs['chat_id'] = int(target) if target.lstrip('-').isdigit() else target
            for key in ('message_thread_id','direct_messages_topic_id','reply_to_message_id'):
                kwargs.pop(key, None)
    result = await method(*args, **kwargs)
    return result


def policy_recipients(platform):
    """Required platform policies may publish a finite recipient allowlist.

    This is classification only, not a send grant. Missing/broken/changed policy
    gives the empty set; actual sends still evaluate the complete envelope.
    """
    try:
        snapshot = _snapshot(platform)
        recipients = None
        for _, _, provider in snapshot[1]:
            values = bounded_call(provider.recipients)
            if type(values) is not frozenset or any(type(x) is not str for x in values):
                return frozenset()
            recipients = values if recipients is None else recipients & values
        if not _same(snapshot, _snapshot(platform)):
            return frozenset()
        return recipients or frozenset()
    except Exception:
        return frozenset()
