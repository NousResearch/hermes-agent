"""Provider profile base class.

A ProviderProfile declares everything about an inference provider in one place:
auth, endpoints, client quirks, request-time quirks. The transport reads this
instead of receiving 20+ boolean flags.

Provider profiles are DECLARATIVE — they describe the provider's behavior.
They do NOT own client construction, credential rotation, or streaming.
Those stay on AIAgent.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agent.account_usage import AccountUsageSnapshot

logger = logging.getLogger(__name__)

# Sentinel for "omit temperature entirely" (Kimi: server manages it)
OMIT_TEMPERATURE = object()


def _profile_user_agent() -> str:
    """Return a ``hermes-cli/<version>`` UA string, with a stable fallback.

    Used by ``ProviderProfile.fetch_models`` so the catalog probe is not
    served the default ``Python-urllib/<ver>`` UA — some providers
    (OpenCode Zen, etc.) sit behind a WAF that returns 403 for that.
    """
    try:
        from hermes_cli.version_info import get_version_info  # lazy: avoid layer cycle at import time
        return f"hermes-cli/{get_version_info().base_version}"
    except Exception:
        return "hermes-cli"


_RESOLVE_LOCK = threading.RLock()  # re-entrant: a callback may read another LazyHeaders


class LazyHeaders(dict[str, str]):
    """``default_headers`` whose callable values (a User-Agent carrying the release) run on first read.

    A source checkout resolves the release from git, and every CLI start imports all provider plugins
    (``hermes_cli.config`` lists their env vars), so headers built at import made every command pay git
    subprocesses for a request it may never send. Keys are stored up front, so ``in``/``len`` stay lazy,
    and the class stays a real ``dict``: plugins copy, merge, mutate and serialize ``default_headers``.

    Every method that reads or writes a value resolves first; ``__iter__`` is overridden so CPython's
    ``dict(x)``/``{**x}``/``f(**x)`` merge leaves its raw-storage fast path for ``keys()`` +
    ``__getitem__``, and ``json.dumps`` takes ``items()`` because the stored keys make the dict
    non-empty. Copies (``.copy()``, ``|``, ``copy``/``pickle``) are plain resolved dicts.
    """

    __slots__ = ("_pending",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pending = any(callable(value) for value in dict.values(self))

    def _resolved(self) -> LazyHeaders:
        if self._pending:
            # One resolver: concurrent first reads (parallel aux clients) would each walk git.
            with _RESOLVE_LOCK:
                if self._pending:
                    for key, value in list(dict.items(self)):
                        if callable(value):
                            dict.__setitem__(self, key, value())
                    self._pending = False
        return self

    def __getitem__(self, key: str) -> str:
        return dict.__getitem__(self._resolved(), key)

    def __iter__(self):
        return dict.__iter__(self._resolved())

    def __reversed__(self):
        return dict.__reversed__(self._resolved())

    def __repr__(self) -> str:
        return dict.__repr__(self._resolved())

    def __eq__(self, other: object) -> bool:
        return dict.__eq__(self._resolved(), other._resolved() if isinstance(other, LazyHeaders) else other)

    def __ne__(self, other: object) -> bool:
        equal = self.__eq__(other)
        return equal if equal is NotImplemented else not equal

    def __or__(self, other: Any) -> Any:
        return dict.__or__(self._resolved(), other)

    def __ror__(self, other: Any) -> Any:
        return dict.__or__(other, self._resolved()) if isinstance(other, dict) else NotImplemented

    def __ior__(self, other: Any) -> LazyHeaders:
        return dict.__ior__(self._resolved(), other)

    def __reduce__(self) -> tuple:
        return (dict, (dict.copy(self._resolved()),))

    def get(self, key: str, default: Any = None) -> Any:
        return dict.get(self._resolved(), key, default)

    def keys(self):
        return dict.keys(self._resolved())

    def values(self):
        return dict.values(self._resolved())

    def items(self):
        return dict.items(self._resolved())

    def copy(self) -> dict[str, str]:
        return dict.copy(self._resolved())

    def __setitem__(self, key: str, value: str) -> None:
        dict.__setitem__(self._resolved(), key, value)

    def __delitem__(self, key: str) -> None:
        dict.__delitem__(self._resolved(), key)

    def pop(self, *args: Any) -> Any:
        return dict.pop(self._resolved(), *args)

    def popitem(self) -> tuple[str, str]:
        return dict.popitem(self._resolved())

    def setdefault(self, key: str, default: Any = None) -> Any:
        return dict.setdefault(self._resolved(), key, default)

    def update(self, *args: Any, **kwargs: Any) -> None:
        dict.update(self._resolved(), *args, **kwargs)


@dataclass
class ProviderProfile:
    """Base provider profile — subclass or instantiate with overrides."""

    # ── Identity ─────────────────────────────────────────────
    name: str
    api_mode: str = "chat_completions"
    aliases: tuple = ()

    # ── Human-readable metadata ───────────────────────────────
    display_name: str = ""       # e.g. "GMI Cloud" — shown in picker/labels
    description: str = ""        # e.g. "GMI Cloud (multi-model direct API)" — picker subtitle
    signup_url: str = ""         # e.g. "https://www.gmicloud.ai/" — shown during setup

    # True when the provider is pre-release / not for public listing. Discovery surfaces (provider
    # pickers, the setup provider list, the dashboard accounts tab) skip it unless ``listed()``
    # says otherwise; resolution by name is untouched, so an explicit ``model.provider`` /
    # ``hermes auth add <name>`` keeps working.
    hidden: bool = False

    # ── Auth & endpoints ─────────────────────────────────────
    env_vars: tuple = ()
    base_url: str = ""
    models_url: str = ""  # explicit models endpoint; falls back to {base_url}/models
    auth_type: str = "api_key"   # api_key|oauth_device_code|oauth_external|external_process|aws_sdk|plugin-defined
    # None preserves transport defaults (external-process clients default to non-streaming).
    # True opts a streaming-capable process client in; False disables streaming for this provider.
    supports_streaming: bool | None = field(default=None, kw_only=True)
    supports_health_check: bool = True  # False → doctor skips /models probe for this provider
    # False → fetch_models returns None without a network call (catalog comes from an SDK/subprocess).
    supports_model_listing: bool = True

    # ── Provider-owned auth (optional; non-api-key plugins) ──────────
    # ``auth_handler(action, args) -> bool``: ``hermes auth add|status|logout|refresh <name>`` calls it
    # FIRST with the parsed CLI namespace; truthy = the plugin owned the action, falsy = built-in path.
    # ``refresh_credential(entry) -> Mapping | None``: the credential pool's refresh of a pooled OAuth
    # row — return the rotated fields (``access_token``, ``refresh_token``, ``expires_at_ms`` …) or raise.
    # Both own their own token endpoints; Hermes passes no secrets beyond the pooled row itself.
    # ``classify_api_error(error, *, status_code, error_code, message, body, model) -> Mapping | None``:
    # consulted by ``agent.error_classifier.classify_api_error`` for THIS provider's failures only, after
    # the generic ``transform_api_error_classification`` plugin hooks and before the built-in pipeline.
    # Return ``{"reason": <FailoverReason name>, ...hint flags}`` to override, ``None`` to decline.
    auth_handler: Callable[[str, Any], Any] | None = None
    refresh_credential: Callable[[Any], Any] | None = None
    classify_api_error: Callable[..., Any] | None = None

    # ── Vision support ────────────────────────────────────────
    # True when the provider's API accepts image content inside
    # tool-result messages natively.  Set on providers that expose
    # multimodal models via tool results (Anthropic Messages API,
    # OpenAI Chat Completions, Gemini, MiniMax, etc.).
    # Falls back to model-catalog lookup when False and the provider
    # has no registered profile.
    supports_vision: bool = False

    # True when the provider's API accepts list-type tool message
    # content (multipart with image_url parts).  Defaults to True for
    # backward compatibility.  Set to False for providers that accept
    # multimodal user messages but reject list-type tool content
    # (e.g. Xiaomi MiMo, which returns 400 "text is not set").
    supports_vision_tool_messages: bool = True

    # True only when this provider's Chat Completions endpoint explicitly
    # documents ``prompt_cache_key`` as an accepted request body field.  This
    # is deliberately opt-in: many OpenAI-compatible endpoints reject unknown
    # top-level fields rather than ignoring them.
    supports_prompt_cache_key: bool = False

    # Private replay carriers use a namespaced '<provider>.native_assistant' type
    # in reasoning_details. Only this profile may receive its matching carrier;
    # other providers (including an unregistered fallback) get ordinary details only.
    native_reasoning_details_type: str | None = None

    # ── External-process providers (auth_type="external_process") ──
    # An agent CLI driven over stdio (ACP) rather than an HTTP endpoint. These
    # describe how to launch it; hermes_cli/auth.py's
    # resolve_external_process_provider_credentials() reads them instead of
    # hardcoding one vendor's binary. Env vars are checked in order and win
    # over the static defaults, so an operator can point at a custom build.
    process_command: str = ""            # default binary, e.g. "kimchi"
    process_args: tuple = ()             # default argv tail, e.g. ("--mode", "acp")
    process_command_env_vars: tuple = ()  # env overrides for the binary, in priority order
    process_args_env_var: str = ""       # env override for argv (shlex-split)

    # ── Model catalog ─────────────────────────────────────────
    # fallback_models: curated list shown in /model picker when live fetch fails.
    # Only agentic models that support tool calling should appear here.
    fallback_models: tuple = ()
    # model_aliases: short name -> id in fallback_models, for providers whose catalog is not
    # in models.dev (external processes); `/model <alias>` resolves here before core guessing.
    model_aliases: dict = field(default_factory=dict)

    # hostname: base hostname for URL→provider reverse-mapping in model_metadata.py
    # e.g. "api.gmi-serving.com". Derived from base_url when empty.
    hostname: str = ""

    # ── Client-level quirks (set once at client construction) ─
    default_headers: dict[str, str] = field(default_factory=dict)

    # ── Request-level quirks ─────────────────────────────────
    # Temperature: None = use caller's default, OMIT_TEMPERATURE = don't send
    fixed_temperature: Any = None
    default_max_tokens: int | None = None
    # ``response_format`` types the API rejects outright (e.g. ("json_schema",)); aux requests omit them up front.
    unsupported_response_formats: tuple = ()
    default_aux_model: str = (
        ""  # cheap model for auxiliary tasks (compression, vision, etc.)
    )
    # empty = use main model

    # Per-model metadata in the canonical model_overrides schema. Partial entries
    # patch catalog metadata; explicit user overrides still win. Exact model IDs.
    model_capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Every field below was added after the released positional layout above: keyword-only, so a
    # plugin subclass's own positional options keep their constructor slots.
    _: KW_ONLY

    # ── L6: provider-owned setup + sign-in surfaces (all optional; None/empty = core's generic path) ──
    # ``account_card``: Accounts-tab card overrides ``{name, flow, cli_command, docs_url}`` for the
    # Desktop/dashboard (``/api/providers/oauth``); non-empty also lists a Keys-tab profile there.
    account_card: dict[str, str] = field(default_factory=dict)
    # ``account_status() -> Mapping``: that card's live status (``logged_in``, ``source``, ``source_label``…);
    # subprocess- and network-free, it runs on every Accounts list. None = ``auth.get_auth_status(name)``.
    account_status: Callable[[], Any] | None = None
    # In-app device-code sign-in (Desktop "Login" button; the card's flow becomes ``device_code``):
    # ``start_device_login() -> {user_code, verification_url, expires_in, interval, state}`` asks the
    # provider for a code; ``poll_device_login(state) -> {status: pending|approved|denied|expired,
    # interval?, credential?, message?}`` makes ONE poll; ``save_device_login(credential)`` persists it.
    # Core owns the wait loop and cancellation and calls save inside the session's profile scope.
    start_device_login: Callable[[], Any] | None = None
    poll_device_login: Callable[[Any], Any] | None = None
    save_device_login: Callable[[Any], None] | None = None
    # ``model_setup_flow(config, current_model, args)``: this provider's own ``hermes model`` step
    # (credential prompt, catalog, persist); None = the generic flow for its auth_type.
    model_setup_flow: Callable[[Any, str, Any], Any] | None = None
    # ``setup_reasoning_efforts(model) -> tuple | None``: levels the setup effort picker offers for
    # *model*; ``()`` = no reasoning control, None = the generic ladder gated by models.dev.
    setup_reasoning_efforts: Callable[[str], Any] | None = None

    # ── Hooks (override in subclass for complex providers) ───

    def fetch_account_usage(
        self, *, base_url: str | None = None, api_key: str | None = None
    ) -> AccountUsageSnapshot | None:
        """Return an account-usage snapshot for this provider, if available.

        The ``/usage`` command invokes this only when no built-in account
        usage fetcher owns the provider. Implementations may make their
        provider-specific request and must return an ``AccountUsageSnapshot``
        or ``None``; exceptions fail open at the dispatch boundary.
        """
        return None

    def resolve_aux_model(self, *, vision: bool = False) -> str:
        """Return a LIVE cheap-model id for auxiliary tasks, or "".

        ``default_aux_model`` is a hardcoded id in source, so it rots: when the
        provider retires that model every auxiliary call spends a round-trip
        404ing before the retry net catches it. Providers that publish a
        machine-readable recommendation should override this and query it, so
        the cheap tier tracks the upstream catalog instead of a constant a human
        has to remember to bump.

        Contract: cheap to call (implementations must cache — this runs on
        client-resolution paths), never raises, and returns "" when it has no
        answer so the caller falls through to ``default_aux_model``.
        """
        return ""

    def get_hostname(self) -> str:
        """Return the provider's base hostname for URL-based detection.

        Uses self.hostname if set explicitly, otherwise derives it from base_url.
        e.g. 'https://api.gmi-serving.com/v1' → 'api.gmi-serving.com'
        """
        if self.hostname:
            return self.hostname
        if self.base_url:
            from urllib.parse import urlparse
            return urlparse(self.base_url).hostname or ""
        return ""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Provider-specific message preprocessing.

        Called AFTER codex field sanitization, BEFORE developer role swap.
        Default: pass-through.
        """
        return messages

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        """Provider-specific extra_body fields.

        Merged into the API kwargs extra_body. Default: empty dict.
        """
        return {}

    def default_reasoning_config(self, model: str | None = None) -> dict | None:
        """Reasoning config the main loop sends when ``agent.reasoning_effort`` is unset.

        None (default) hands the unset state to ``build_api_kwargs_extras`` as ``reasoning_config=None``,
        where each profile already decides (Nous/OpenRouter fill medium; Anthropic omits). A profile
        that would otherwise leave the route's own default in charge returns the config here so the
        agent records it as what went on the wire (the reasoning-rejection ladder reads that).
        """
        return None

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Provider-specific kwargs split between extra_body and top-level api_kwargs.

        Returns (extra_body_additions, top_level_kwargs).
        The transport merges extra_body_additions into extra_body, and
        top_level_kwargs directly into api_kwargs.

        This split exists because some providers put reasoning config in
        extra_body (OpenRouter: extra_body.reasoning) while others put it
        as top-level api_kwargs (Kimi: api_kwargs.reasoning_effort).

        Default: ({}, {}).
        """
        return {}, {}

    def build_client_kwargs_extras(self, **context: Any) -> dict[str, Any]:
        """Provider-specific OpenAI client keyword arguments.

        Values are defaults: explicit runtime/custom-provider settings win.
        The returned mapping must be cheap to build and must not perform I/O.
        """
        return {}

    def default_vision_model(self) -> str | None:
        """Return a default vision model id for this provider, or None.

        Overrideable hook for providers that discover their vision default at
        runtime (e.g. from a live catalog) rather than pinning one in code.
        Keeps provider-specific vision discovery inside the provider's plugin
        instead of a name-check branch in shared vision resolution.

        Default: None (no provider-specific vision model — the caller falls
        back to the user's chat model or the aggregator chain).
        """
        return None

    def get_model_context_length(self, model: str) -> int | None:
        """Provider-qualified context bound; explicit user overrides take precedence."""
        return None

    def get_usage_cost(self, model: str, usage: Any) -> Any | None:
        """Optional CostResult from canonical usage; distinguish estimates from invoices."""
        return None

    def get_max_tokens(self, model: str | None) -> int | None:
        """Return the default max_tokens cap for *model*.

        Overrideable hook for providers that need per-model output caps —
        e.g. a relay that fronts several upstream backends, each with a
        different completion-token limit. The transport calls this when
        the user hasn't set an explicit max_tokens.

        Default: return self.default_max_tokens (the static profile field),
        ignoring the model name. Override in a subclass to vary the cap
        per-model.
        """
        return self.default_max_tokens

    def supported_reasoning_efforts(
        self, model: str | None
    ) -> tuple[str, ...] | None:
        """Declared reasoning-effort vocabulary for *model* on this provider.

        Overrideable hook for providers whose gateway validates
        ``reasoning.effort`` per model instead of ignoring or clamping
        unknown levels server-side (Ramp Router derives this from its live
        ``/v1/models`` catalog). The Responses transport consults it before
        falling back to its built-in per-backend vocabularies; it is the
        profile-declared analog of the OpenRouter catalog clamp on the
        chat-completions path (``openrouter_model_reasoning_capabilities``).

        Tri-state contract:
          - ``None`` — unknown/undeclared: the transport keeps its default
            vocabulary for the wire (this base implementation).
          - ``()`` — the model accepts NO reasoning parameters at all; the
            transport must omit reasoning fields entirely (some gateways
            return HTTP 400 rather than ignoring them).
          - non-empty tuple — clamp the requested effort onto these levels
            (``agent.reasoning_effort.clamp_effort`` semantics: nearest
            weaker supported level, never escalate).

        Implementations are called on the per-request hot path and must not
        block on network I/O — answer from a cache and return None while
        cold.
        """
        return None

    # ── L1 wire seams (per-model wire, endpoint-keyed headers, max-tokens param) ──
    # Endpoint-keyed hooks take the request's ``base_url``: core asks the named provider's profile first,
    # then any registered profile overriding the hook (``agent.provider_route_hooks``), so a ``custom``
    # route pointed at this provider's host is served too. Return None / ``{}`` for an endpoint that is
    # not yours. All four are pure: no I/O unless the docstring says so.

    def api_mode_for_model(self, model: str, *, api_key: str | None = None) -> str | None:
        """Wire (``api_mode``) this provider serves *model* on, or None to keep the static ``api_mode``.

        ``api_key=None`` is the pure hot path (agent init, fallback, aux client builds): answer from the
        id alone. A string ``api_key`` comes from resolution-time callers (``/model``, runtime resolve),
        which may consult a cached live catalog with that credential to normalize the id first.
        """
        return None

    def client_headers(
        self, *, base_url: str, api_key: str = "", has_images: bool = False,
    ) -> dict[str, str] | None:
        """Default headers for an OpenAI client built for *base_url*, or None when this profile has no
        endpoint-specific headers there (core then uses the static ``default_headers``). ``has_images``
        asks for the headers of a request-scoped client carrying native image parts. Core merges user
        ``model.default_headers`` over the result and re-adds missing keys on snapshot rebuilds."""
        return None

    def request_headers(self, *, base_url: str, model: str = "", user_initiated: bool = False) -> dict[str, str]:
        """Per-request headers (``extra_headers``) for *base_url*; ``user_initiated`` is True only on the
        first attempt of a user turn's first API call (attribution). Pure: core memoizes the answer per argument set."""
        return {}

    def max_tokens_param(self, *, base_url: str, model: str = "") -> str | None:
        """``"max_completion_tokens"`` / ``"max_tokens"`` for requests to *base_url*, or None for the
        core heuristic (direct OpenAI / Azure host or newer OpenAI model families). Pure: memoized per arguments."""
        return None

    def listed(self) -> bool:
        """Whether discovery surfaces may offer this provider.

        Every non-``hidden`` profile is listed. A ``hidden`` (pre-release) one surfaces once the user
        opted in by name (``hermes auth add <name>`` put a row in its credential pool), so the
        pickers can then offer it. Reads auth.json only: it runs on every picker build.
        """
        if not self.hidden:
            return True
        from hermes_cli.auth import read_credential_pool  # lazy: providers/ must not import hermes_cli at load
        return bool(read_credential_pool(self.name))

    def create_client(self, **client_kwargs: Any) -> Any | None:
        """Return a provider-specific client, or ``None`` for the standard one.

        Most providers speak OpenAI-compatible HTTP and want the shared
        ``openai.OpenAI`` client the core builds — they inherit this and return
        ``None``. A provider whose wire protocol is not HTTP at all (the ACP
        subprocess shims) or which needs a native SDK overrides this and
        returns its own client object.

        ``client_kwargs`` is the same mapping the core would have passed to
        ``openai.OpenAI`` (``api_key``, ``base_url``, ``command``, ``args``,
        timeouts, headers…). Unknown keys must be tolerated: the core adds to
        this mapping over time, so an override should accept ``**kwargs`` and
        pick what it needs rather than enumerate.

        Returning ``None`` (the default) is always safe — the caller falls
        through to its existing construction path.

        This is the hook that lets a provider ship *outside* this tree: with it,
        a profile registered from ``~/.hermes/plugins/model-providers/`` or a
        pip entry point can supply its own transport without any core edit. See
        ``website/docs/developer-guide/model-provider-plugin.md``.
        """
        return None

    def setup_status(self, **kwargs: Any) -> dict[str, Any] | None:
        """External-process providers: ``{available, logged_in, plan, detail, login_command}`` from the
        CLI itself so setup can gate on login. ``None`` = nothing to report beyond executable presence."""
        return None

    def discover_models(self, **kwargs: Any) -> list[dict[str, Any]] | None:
        """External-process providers: the account's live picker as ``[{id, label, note}]`` without
        any inference request; ``None`` falls back to ``fallback_models``."""
        return None

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the live model list from the provider's models endpoint.

        Returns a list of model ID strings, or None if the fetch failed or
        the provider does not support live model listing.

        Resolution order for the endpoint URL:
          1. base_url + "/models", but ONLY when the caller passed a base_url
             that differs from this profile's default (a user-configured
             model.base_url pointing at a proxy/custom endpoint). Callers
             pass base_url unconditionally — falling back to the profile
             default when the user configured nothing — so equality with
             self.base_url means "not customised" and must not shadow
             models_url.
          2. self.models_url  (explicit override — use when the models
             endpoint differs from the inference base URL, e.g. OpenRouter
             exposes a public catalog at /api/v1/models while inference is
             at /api/v1)
          3. self.base_url + "/models"  (standard OpenAI-compat fallback)

        The default implementation sends Bearer auth when api_key is given
        and forwards self.default_headers. Override to customise auth, path,
        response shape, or to return None for providers with no REST catalog.

        Callers must always fall back to the static _PROVIDER_MODELS list
        when this returns None.
        """
        if not self.supports_model_listing:
            return None
        caller_base = (base_url or "").strip()
        effective_base = caller_base or self.base_url
        custom_base = bool(caller_base) and (
            caller_base.rstrip("/") != (self.base_url or "").rstrip("/")
        )
        if custom_base:
            url = caller_base.rstrip("/") + "/models"
        else:
            url = (self.models_url or "").strip()
            if not url:
                if not effective_base:
                    return None
                url = effective_base.rstrip("/") + "/models"

        import json
        import urllib.request

        from hermes_cli.urllib_security import open_credentialed_url

        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        # Some providers (e.g. OpenCode Zen) sit behind a WAF that blocks
        # the default ``Python-urllib/<ver>`` User-Agent.  Set a generic
        # hermes-cli UA so the catalog endpoint is reachable.
        req.add_header("User-Agent", _profile_user_agent())
        for k, v in self.default_headers.items():
            req.add_header(k, v)

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            from hermes_cli.chat_catalog import chat_catalog_ids

            return chat_catalog_ids(items)
        except Exception as exc:
            logger.debug("fetch_models(%s): %s", self.name, exc)
            return None

    # ── Lane L4: catalog / metadata seams ─────────────────────
    # Hosts (exact or any subdomain) that serve THIS provider's API, so a ``custom`` endpoint or an
    # auto-routed aux client at one of them is recognised as this provider (``providers.profile_for_host``).
    # It also makes ``reasoning_payload`` the reasoning gate there, except on hosts core gates itself
    # (OpenRouter, Ollama Cloud, Nous Portal, Vercel AI Gateway, the ``lmstudio`` provider).
    hostnames: tuple = ()
    # Hosts that identify this provider's catalog for model metadata (context length, models.dev) but
    # take other credentials; matched only by ``profile_for_host(url, metadata=True)``.
    metadata_hostnames: tuple = ()
    # models.dev provider id when it differs from ``name`` (e.g. ``fireworks`` -> ``fireworks-ai``).
    models_dev_id: str = ""

    def live_context_length(self, model: str, *, api_key: str = "", base_url: str = "") -> int | None:
        """Context window from the provider's live, credentialed catalog; consulted after the persistent cache."""
        return None

    def normalize_model_id(self, model: str, *, api_key: str = "") -> str | None:
        """Provider-native id for *model* (alias table, catalog repair); None = core's generic normalization."""
        return None

    # ── Lane L2: provider-owned credential seams ──────────────────────────────
    # Credentials are commonly ambient (another tool's login, a token exported for other purposes):
    # never auto-detected as the inference provider, and the auxiliary fallback chain skips the
    # provider unless the user explicitly configured it.
    ambient_credentials: bool = field(default=False, kw_only=True)

    def resolve_api_key_secret(self) -> tuple[str, str] | None:
        """Provider-owned ``(runtime_api_key, source)`` resolution; None = generic env/pool ladder."""
        return None

    def runtime_base_url(self, *, api_key: str, default: str, env_url: str) -> str | None:
        """Runtime base URL derived from the credential (e.g. an account endpoint); None = env or default."""
        return None

    def prepare_pool_entry(self, entry: Any, api_key: str) -> str:
        """Turn the selected pool row's secret into the runtime key (may update ``entry``); default as-is."""
        return api_key

    def seed_pool_credentials(self, *, suppressed_sources: frozenset[str],
                              explicitly_configured: bool) -> list[dict] | None:
        """Pool rows ``{source, access_token, base_url, label}`` this provider seeds itself; skip any source in
        ``suppressed_sources`` (``hermes auth remove``). Plain data, so it crosses ``plugins.isolation: host``.
        Overriding it replaces generic env seeding (that loop would overwrite exchanged tokens with the raw env
        value), so an override returns ``[]`` for none; only the base implementation keeps generic env seeding."""
        return None

    def remove_credential_source(self, source: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
        """``hermes auth remove`` of *source*: ``(sources_to_suppress, hints)``; None = generic removal."""
        return None

    def refresh_runtime_credentials(self, *, reason: str) -> tuple[str, str | None] | None:
        """Re-mint the live client's ``(api_key, base_url or None)`` on ``reason`` (``unauthorized`` |
        ``stale`` | ``auxiliary_unauthorized``); None = nothing fresher. Error path only, I/O allowed."""
        return None

    def is_stale_credential_error(self, *, status_code: int | None, message: str) -> bool:
        """True when a non-auth error really means the credential went stale (pure, no I/O)."""
        return False

    def credential_refresh_notice(self, *, reason: str) -> str:
        """Status line after ``refresh_runtime_credentials(reason=...)`` succeeded; "" = generic wording."""
        return ""

    def request_rejection_hint(self, *, status_code: int | None, base_url: str) -> tuple[str, ...]:
        """User-facing lines explaining an endpoint's hard rejection of *any* route (pure); default none."""
        return ()

    def external_auth_evidence(self) -> tuple[bool, str | None] | None:
        """Subprocess-free ``(verified, source)`` login evidence for an external CLI; None = binary resolves."""
        return None

    def missing_credentials_hint(self) -> str | None:
        """One sentence appended to every "No usable credentials found" surface (runtime resolution, CLI and
        Desktop readiness), e.g. the provider's own sign-in command; None = the generic message alone."""
        return None

    # ── Lane L3: Responses-route quirks and message-level reasoning carriers ───
    # Every field and hook below applies only on hosts the profile declares in ``hostnames``;
    # the defaults are the generic wire. Hot path: answers must be pure and cheap (no I/O). A hook that
    # raises is logged once and answers the default for the rest of the session (provider_route_hooks).
    responses_replay_item_ids: bool = True  # replayed Responses ``message`` items keep their ``id``
    responses_prompt_cache_key: bool = True  # the Responses body carries ``prompt_cache_key``
    responses_issuer_kind: str = ""  # persisted stamp for encrypted reasoning minted here ("" = URL-derived)
    responses_trusts_completed_status: bool = True  # False: a reasoning-only ``completed`` response continues
    responses_native_compaction: bool = True  # False: never send ``context_management``
    responses_profile_reasoning: bool = False  # True: ``reasoning`` = ``reasoning_payload`` + summary, no ``include``
    message_carrier_opaque: str = ""  # top-level assistant key of an opaque reasoning blob (latest streamed wins)
    message_carrier_text: str = ""  # top-level assistant key of readable reasoning text (streamed parts concatenate)
    replays_tool_call_signatures: bool = False  # tool calls keep Gemini ``extra_content`` signatures (by provider name)

    def reasoning_payload(self, model: str | None, reasoning_config: dict | None, *, base_url: str) -> dict | None:
        """Verbatim ``reasoning`` object the endpoint takes for *model* (catalog-clamped), or None for none.
        Pure: memoized per arguments."""
        return None

    def replay_message_carriers(self, values: dict[str, Any], *, model: str | None, base_url: str) -> dict[str, Any]:
        """Captured message carriers (``message_carrier_*`` keys) to put back on a replayed assistant message.
        Decide from which keys are present, not their content: core asks once per (model, base_url, key set)
        and replays the same keys for every message with that key set (an answer whose values are not
        copies of input values is asked per message)."""
        return {}

    def replays_function_thought_signature(self, *, model: str | None, base_url: str) -> bool:
        """True when replayed tool calls keep ``function.thought_signature`` (stripped everywhere else).
        Pure: memoized per arguments."""
        return False

    # --- I2-cat: provider-owned catalog sources ---
    # ``picker_catalog(force_refresh) -> list | None``: the provider's own model-picker catalog (own
    # credential lookup, live fetch, curated fallback); set = a live source, tried before the generic
    # profile fetch like a built-in per-provider fetcher. None return falls through to that path.
    picker_catalog: Callable[[bool], Any] | None = field(default=None, kw_only=True)
    # Curated ids registered as this provider's row of core's static catalog (model detection,
    # defaults, validation) the way a built-in provider's curated list is; empty = no row.
    static_models: tuple = field(default=(), kw_only=True)
    # Credential files (``~`` allowed) whose mtime is folded into every cached catalog's fingerprint.
    catalog_credential_files: tuple = field(default=(), kw_only=True)

    def probe_models(
        self, *, base_url: str, api_key: str | None = None, timeout: float = 5.0,
    ) -> tuple[list[str] | None, str, str] | None:
        """``(model ids or None, probed_url, resolved_base_url)`` for a ``/models`` probe of *base_url*
        this provider answers from its own catalog; None = the generic HTTP probe."""
        return None

    def invalidate_catalog_cache(self) -> None:
        """Drop in-process catalog memos (``/model --refresh``); default none."""

    # --- I2-static: provider-table placement ---
    picker_after: str = ""  # provider slug this row follows in the provider list ("" = appended)
    picker_group: tuple = ()  # (group_id, label, description): fold with same-group rows in interactive pickers
    catalog_aliases: bool = False  # ``aliases`` also normalize in the model catalog (``/model``, pickers)
    aggregator: bool = False  # catalog re-exposes many vendors' models (vendor/family ids, alias resolution)
    # HermesOverlay fields (transport, auth_type, extra_env_vars, base_url_override, base_url_env_var) for
    # the provider's models.dev-keyed definition, keyed by ``models_dev_id`` or ``name``; {} = none.
    provider_overlay: dict = field(default_factory=dict)
    # Extra names that normalize to the ``provider_overlay`` definition (its id, else ``name``); () = none.
    overlay_aliases: tuple = ()

    # --- I2-auth: hermes-auth registry row mirrored from this profile (hermes_cli/auth_plugin_providers.py) ---
    # auth_type of the registry row when credential resolution differs from the setup flow ``auth_type`` names
    # (an env-token provider with its own login); "" = ``auth_type``.
    registry_auth_type: str = ""
    # Env var overriding ``base_url`` on the registry row when it is not listed in ``env_vars``; "" = none.
    base_url_env_var: str = ""
    # Registry id this provider's row sits right after, keeping its place in priority-ordered scans
    # (aux api-key chain, env auto-detect); "" or an unknown id = appended.
    registry_after: str = ""
