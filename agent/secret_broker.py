"""In-memory credential broker — NVIDIA OpenShell's placeholder pattern,
applied at Hermes's LLM HTTP boundary.

When ``security.credential_broker.enabled`` is set, the real provider API key
is **not** placed on the constructed ``openai``-SDK client. The SDK receives an
opaque placeholder; a per-request ``httpx`` hook swaps the real key into the
outgoing ``Authorization`` / ``x-api-key`` header just before the request
leaves. The real key lives only in this module's in-memory table.

Honest scope — Hermes is a single process, so this is **defense-in-depth, not
a boundary**: it keeps the key off the (widely-referenced, loggable) SDK client
object, and a placeholder never escapes upstream. It does not hide the key from
in-process code, which ``SECURITY.md`` §2.5 already treats as trusted, and the
key still exists in ``os.environ`` / the credential pool. It covers the
OpenAI-compatible (``openai`` SDK) path, the AWS Bedrock chokepoint
(``bedrock_adapter._get_bedrock_runtime_client`` /
``_get_bedrock_control_client``), and the native Anthropic SDK chokepoint
(``anthropic_adapter.build_anthropic_client``). Anthropic-on-Bedrock via
``AnthropicBedrock`` is covered transitively when routed through
``bedrock_adapter``; when the Anthropic SDK is used directly, the SDK's
exposed knobs determine coverage. SigV4 mode keeps broker placeholders on the
cached botocore signer and resolves the real key pair only inside
``get_frozen_credentials()`` for the signing snapshot.

Every public setup function is best-effort and fail-open: a broker-install
failure must degrade to "use the real key directly", never break client
construction. SigV4 broker-table misses after successful wrapping are the
exception: they fail closed by signing with placeholders so AWS rejects the
request rather than reintroducing real credentials onto the cached signer.
"""

from __future__ import annotations

import re
import secrets
import threading
from collections import OrderedDict
from typing import Optional

_PLACEHOLDER_PREFIX = "hermes-broker-"
_PLACEHOLDER_RE = re.compile(rf"{_PLACEHOLDER_PREFIX}[0-9a-f]{{32}}")

# bounded broker table. A long-running gateway that
# rotates API keys (or a multi-tenant deployment that registers many
# distinct keys) would otherwise accumulate dead entries — each entry
# is small but the secret value itself sits in memory past the SDK's
# need. 1024 is generous for single-key and small-multi-key installs;
# operators with more tenants can tune via ``SecretBroker.max_entries``.
_DEFAULT_MAX_ENTRIES = 1024


class SecretBroker:
    """Thread-safe two-way map between real secrets and opaque placeholders.

    Bounded by ``max_entries`` (default ``1024``). When the cap is hit, the
    oldest registered entry is evicted FIFO-style — a long-lived gateway
    that rotates API keys can't grow its in-memory table without bound.
    Deduplicated registrations refresh recency, so a frequently-used key
    isn't aged out by a burst of one-shot registrations.
    """

    def __init__(self, *, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        # OrderedDict gives us O(1) move_to_end / popitem(last=False) for
        # FIFO eviction without an explicit linked list.
        self._by_secret: "OrderedDict[str, str]" = OrderedDict()
        self._by_placeholder: "OrderedDict[str, str]" = OrderedDict()
        self._max_entries = max(1, int(max_entries))

    def register(self, secret: str) -> str:
        """Return a stable placeholder for *secret* (deduplicated)."""
        with self._lock:
            existing = self._by_secret.get(secret)
            if existing is not None:
                # Refresh recency so an actively-used key isn't aged out
                # by a burst of one-shot registrations.
                self._by_secret.move_to_end(secret)
                self._by_placeholder.move_to_end(existing)
                return existing
            placeholder = _PLACEHOLDER_PREFIX + secrets.token_hex(16)
            self._by_secret[secret] = placeholder
            self._by_placeholder[placeholder] = secret
            while len(self._by_secret) > self._max_entries:
                evicted_secret, evicted_placeholder = self._by_secret.popitem(last=False)
                self._by_placeholder.pop(evicted_placeholder, None)
            return placeholder

    def resolve(self, placeholder: str) -> Optional[str]:
        """Return the real secret for *placeholder*, or None if unknown."""
        with self._lock:
            return self._by_placeholder.get(placeholder)

    def resolve_in(self, text: str) -> str:
        """Replace every known placeholder occurring inside *text* with its secret."""
        if _PLACEHOLDER_PREFIX not in text:
            return text
        with self._lock:
            table = dict(self._by_placeholder)

        def _sub(match: "re.Match[str]") -> str:
            return table.get(match.group(0), match.group(0))

        return _PLACEHOLDER_RE.sub(_sub, text)

    @staticmethod
    def is_placeholder(value: str) -> bool:
        return isinstance(value, str) and _PLACEHOLDER_RE.fullmatch(value) is not None


_broker: Optional[SecretBroker] = None
_broker_lock = threading.Lock()


def get_broker() -> SecretBroker:
    """Return the process-wide broker singleton."""
    global _broker
    if _broker is None:
        with _broker_lock:
            if _broker is None:
                _broker = SecretBroker()
    return _broker


# Cached for process lifetime — ``apply_to_client_kwargs`` runs on every LLM
# client build; re-parsing YAML each time is measurable at high request volume.
# Call ``install_broker_signal_handler()`` from gateway startup to install a
# SIGHUP handler that invalidates this cache automatically — that lets an
# operator flip ``security.credential_broker.enabled`` in ``config.yaml`` and
# pick it up without a process restart. On Windows (no SIGHUP), or if no
# signal handler is installed, operators must restart the gateway for a
# config change to take effect.
_broker_enabled_cache: Optional[bool] = None
_broker_config_cache_lock = threading.Lock()
_signal_handler_installed = False
_signal_handler_lock = threading.Lock()


def clear_broker_config_cache() -> None:
    """Drop cached broker config so the next check re-reads config.yaml.

    Deliberately lock-free: this runs inside the SIGHUP handler, which CPython
    executes on the main thread between bytecodes. If the main thread is
    inside ``broker_enabled()`` holding ``_broker_config_cache_lock`` (a
    non-reentrant lock) when the signal lands, acquiring the lock here would
    deadlock the process. A bare global assignment is atomic under the GIL;
    the worst-case race is a concurrent ``broker_enabled()`` overwriting the
    invalidation with a just-computed value, which an operator can resolve by
    sending SIGHUP again — strictly better than a deadlock.
    """
    global _broker_enabled_cache
    _broker_enabled_cache = None


def install_broker_signal_handler() -> bool:
    """Install a SIGHUP handler that invalidates the broker config cache.

    lets operators flip
    ``security.credential_broker.enabled`` in ``config.yaml`` and signal the
    gateway with ``kill -HUP <pid>`` to pick it up without a process restart.
    The handler also chains any previously-registered SIGHUP handler so we
    don't clobber operator-installed reload hooks.

    Returns:
        ``True`` if the handler is now installed (this call or a prior one);
        ``False`` on Windows (no ``signal.SIGHUP`` attribute) or if the
        install raised for any reason. Broker-config staleness is
        operationally surprising but a signal-install failure must not
        break gateway startup.

    Idempotent: subsequent calls are no-ops and return ``True`` immediately.

    Chosen over ``SIGUSR1`` because the upstream gateway already wires
    ``loop.add_signal_handler(signal.SIGUSR1, restart_signal_handler)`` for
    the in-process restart pipeline (``/restart``, ``/update``,
    ``hermes gateway stop --restart``); chaining through asyncio's signal
    bridge is brittle, and clobbering it would break those flows. SIGHUP
    is unused in the gateway runtime and is the canonical "reload config"
    signal for Unix daemons.

    Call from gateway startup AFTER ``setup_logging`` so any chained-handler
    diagnostics surface on the in-scope logger. See
    ``integration/UPSTREAM-BROKER-WIRING.md`` Edit 4 for the operator wiring.
    """
    global _signal_handler_installed
    import signal

    if not hasattr(signal, "SIGHUP"):
        return False
    with _signal_handler_lock:
        if _signal_handler_installed:
            return True
        try:
            prior = signal.getsignal(signal.SIGHUP)

            def _chained_sighup_handler(signum, frame):
                clear_broker_config_cache()
                if callable(prior) and prior not in (
                    signal.SIG_DFL,
                    signal.SIG_IGN,
                ):
                    try:
                        prior(signum, frame)
                    except Exception:
                        # Best-effort chaining — never break our own handler
                        # because someone else's raised.
                        pass

            signal.signal(signal.SIGHUP, _chained_sighup_handler)
            _signal_handler_installed = True
            return True
        except Exception:
            return False


def broker_enabled() -> bool:
    """Return ``security.credential_broker.enabled`` (default False, fail-safe)."""
    global _broker_enabled_cache
    with _broker_config_cache_lock:
        if _broker_enabled_cache is not None:
            return _broker_enabled_cache
        try:
            from hermes_cli.config import cfg_get, load_config
            from utils import is_truthy_value

            enabled = is_truthy_value(
                cfg_get(
                    load_config(),
                    "security",
                    "credential_broker",
                    "enabled",
                    default=False,
                )
            )
        except Exception:
            enabled = False
        _broker_enabled_cache = enabled
        return enabled


# ── httpx request hook ──────────────────────────────────────────────────────

_HEADERS_TO_RESOLVE = ("authorization", "x-api-key", "api-key")
_hook_install_lock = threading.Lock()


def _rewrite_request_headers(request, broker: SecretBroker) -> None:
    """Swap any placeholder in the auth headers of *request* for the real secret."""
    for header in _HEADERS_TO_RESOLVE:
        value = request.headers.get(header)
        if value and _PLACEHOLDER_PREFIX in value:
            request.headers[header] = broker.resolve_in(value)


def install_request_hook(http_client, broker: SecretBroker) -> None:
    """Attach (once) a request hook that resolves broker placeholders at egress.

    Works for both sync ``httpx.Client`` and async ``httpx.AsyncClient``.
    """
    try:
        import httpx
    except ImportError:
        return

    with _hook_install_lock:
        if getattr(http_client, "_hermes_broker_hooked", False):
            return

        hooks = getattr(http_client, "event_hooks", None)
        if not isinstance(hooks, dict):
            return

        if isinstance(http_client, httpx.AsyncClient):
            async def _hook(request):  # type: ignore[no-redef]
                _rewrite_request_headers(request, broker)
        else:
            def _hook(request):  # type: ignore[no-redef]
                _rewrite_request_headers(request, broker)

        request_hooks = list(hooks.get("request", []))
        request_hooks.append(_hook)
        hooks["request"] = request_hooks
        http_client.event_hooks = hooks
        try:
            http_client._hermes_broker_hooked = True
        except Exception:
            pass


def apply_to_client_kwargs(client_kwargs: dict) -> None:
    """Broker the ``api_key`` in *client_kwargs* in place, if broker mode is on.

    Replaces the real key with a placeholder and installs the resolving hook on
    the kwargs' ``http_client``. No-op (fail-open) when broker mode is off, when
    there is no Hermes-managed httpx client to carry the hook, or on any error.
    """
    try:
        if not broker_enabled():
            return
        http_client = client_kwargs.get("http_client")
        if http_client is None:
            return  # no Hermes httpx client to carry the egress hook
        api_key = client_kwargs.get("api_key")
        if not api_key or not isinstance(api_key, str):
            return
        broker = get_broker()
        if broker.is_placeholder(api_key):
            install_request_hook(http_client, broker)
            return  # already brokered
        client_kwargs["api_key"] = broker.register(api_key)
        install_request_hook(http_client, broker)
    except Exception:
        # Defense-in-depth only — never break client construction.
        pass


# ── AWS Bedrock (boto3) broker integration ──────────────────────────────────


def register_bearer_token(token: str) -> str:
    """Register *token* with the process broker and return its placeholder.

    Convenience wrapper over :meth:`SecretBroker.register` used by the Bedrock
    adapter when wiring the ``AWS_BEARER_TOKEN_BEDROCK`` path.
    """
    return get_broker().register(token)


def register_aws_credentials(access_key: str, secret_key: str) -> "tuple[str, str]":
    """Register a SigV4 credential pair and return ``(id_placeholder, secret_placeholder)``.

    Both halves are registered separately so the broker can resolve them
    independently inside the broker-backed credentials object.
    """
    placeholder_id, placeholder_secret, _placeholder_token = (
        register_aws_credential_triplet(access_key, secret_key, None)
    )
    return placeholder_id, placeholder_secret


def register_aws_credential_triplet(
    access_key: str,
    secret_key: str,
    token: Optional[str] = None,
) -> "tuple[str, str, Optional[str]]":
    """Register a SigV4 credential set and return placeholder values.

    Static AWS credentials may include a session token (for example
    ``AWS_SESSION_TOKEN``). Treat it like the key pair: keep only the
    placeholder on botocore's cached signer and resolve the real value in the
    frozen signing snapshot.
    """
    broker = get_broker()
    return (
        broker.register(access_key),
        broker.register(secret_key),
        broker.register(token) if token else None,
    )


def _brokered_sigv4_credentials(credentials, broker: SecretBroker):
    """Return a placeholder-at-rest SigV4 credentials object, or ``None``.

    The upstream Bedrock wiring must first replace static real credentials with
    broker placeholders. This guard deliberately rejects refreshable/non-static
    credential sources so broker mode does not break botocore's refresh
    contract for assume-role, IMDS, SSO, web identity, or similar providers.
    """
    access_placeholder = getattr(credentials, "access_key", None)
    secret_placeholder = getattr(credentials, "secret_key", None)
    token_placeholder = getattr(credentials, "token", None)
    if not (
        SecretBroker.is_placeholder(access_placeholder)
        and SecretBroker.is_placeholder(secret_placeholder)
    ):
        return None
    if token_placeholder and not SecretBroker.is_placeholder(token_placeholder):
        return None

    import botocore.credentials as _bc

    method = getattr(credentials, "method", None)
    account_id = getattr(credentials, "account_id", None)

    class _BrokeredSigV4Credentials(_bc.Credentials):
        def __init__(self) -> None:
            kwargs = {
                "access_key": access_placeholder,
                "secret_key": secret_placeholder,
                "token": token_placeholder,
                "method": method,
            }
            if account_id is not None:
                kwargs["account_id"] = account_id
            try:
                super().__init__(**kwargs)
            except TypeError:
                kwargs.pop("account_id", None)
                super().__init__(**kwargs)
                if account_id is not None:
                    try:
                        self.account_id = account_id
                    except Exception:
                        pass
            self._hermes_broker_sigv4 = True

        def get_frozen_credentials(self):
            real_access_key = broker.resolve(self.access_key)
            real_secret_key = broker.resolve(self.secret_key)
            real_token = broker.resolve(self.token) if self.token else None
            if (
                not real_access_key
                or not real_secret_key
                or (self.token and real_token is None)
            ):
                return super().get_frozen_credentials()
            try:
                return _bc.ReadOnlyCredentials(
                    real_access_key,
                    real_secret_key,
                    real_token,
                    getattr(self, "account_id", None),
                )
            except TypeError:
                return _bc.ReadOnlyCredentials(
                    real_access_key,
                    real_secret_key,
                    real_token,
                )

    return _BrokeredSigV4Credentials()


def install_bedrock_event_hook(client, *, mode: str) -> None:
    """Attach (once) Bedrock broker handling for *client*.

    Bearer-token mode installs a botocore event hook. SigV4 mode keeps the
    historical function name for call-site stability but wraps credentials
    instead of registering an event handler.

    ``mode="bearer"``
        Registers a ``before-send.<service>`` handler that rewrites the
        ``Authorization: Bearer <placeholder>`` header (emitted by botocore's
        :class:`BearerAuth` from the client's ``_auth_token``) into the real
        bearer token at egress time. Mirrors the OpenAI broker shape: the real
        token never sits on ``client._request_signer._auth_token`` between
        calls — only the placeholder does.

    ``mode="sigv4"``
        Wraps the signer's placeholder credentials with a broker-backed
        :class:`botocore.credentials.Credentials` object. The cached signer
        continues to expose placeholders at rest; botocore's signing path calls
        ``get_frozen_credentials()``, which resolves the real key pair (and
        optional session token) only for the immutable signing snapshot. The
        wrapper is installed only when the signer already carries broker
        placeholders for a static key pair; refreshable credential providers
        are left untouched. If a broker-table miss occurs after wrapping, the
        signing snapshot falls back to placeholders and AWS rejects the
        request; SigV4 intentionally fails closed rather than putting real
        credentials back on the cached signer.

    Idempotent: a second call on the same client is a no-op (mirrors the
    ``_hermes_broker_hooked`` attribute pattern from
    :func:`install_request_hook`).

    Installer fail-open: any unexpected setup error is swallowed — Bedrock
    requests still go out using the cached client's existing credentials.
    """
    try:
        events = getattr(getattr(client, "meta", None), "events", None)
        if events is None:
            return
        if getattr(events, "_hermes_broker_hooked", False):
            return

        service_id = getattr(client.meta, "service_model", None)
        if service_id is None:
            return
        service_name = service_id.service_id.hyphenize()

        broker = get_broker()

        if mode == "bearer":
            def _bearer_hook(request, **_kw):
                try:
                    auth = request.headers.get("Authorization")
                    if auth and _PLACEHOLDER_PREFIX in auth:
                        request.headers["Authorization"] = broker.resolve_in(auth)
                except Exception:
                    # Best-effort: never break Bedrock requests.
                    pass

            events.register(f"before-send.{service_name}", _bearer_hook)

        elif mode == "sigv4":
            signer = getattr(client, "_request_signer", None)
            if signer is None:
                return
            creds = getattr(signer, "_credentials", None)
            if getattr(creds, "_hermes_broker_sigv4", False):
                pass
            else:
                wrapped = _brokered_sigv4_credentials(creds, broker)
                if wrapped is None:
                    return
                signer._credentials = wrapped

        else:
            return  # unknown mode — silent no-op

        try:
            events._hermes_broker_hooked = True
        except Exception:
            pass
    except Exception:
        # Defense-in-depth only — never break client construction.
        pass
