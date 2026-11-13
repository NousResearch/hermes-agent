"""``hermes fallback check`` — readiness of every fallback-chain entry without an agent session (#63852).

Static mode resolves each entry with the switch's own kwargs; ``--live`` sends one request on the
client the swap builds. The live request never goes through ``call_llm``: its recovery ladder would
retry or answer from another provider and hide the dead entry. Exit status is 1 when any entry is
not ready, so cron and CI can gate on it. User-facing contract: docs ``fallback-providers.md``.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import re
import time
from datetime import UTC, datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Room for reasoning models, which spend their first tokens thinking (same rationale as the local
# runtime's touch generation); providers that need no cap never receive it (``_forwards_max_tokens``).
_LIVE_MAX_TOKENS = 512
_LIVE_TIMEOUT_S = 30.0
_LIVE_PROMPT = [{"role": "user", "content": "Reply with exactly: OK"}]
_NOT_PROBED = frozenset({"auto", "moa"})
# Router meta-models (``openrouter/auto``, ``[<model>]@preset/<slug>``) answer under the model they
# picked, so the reported id proves nothing.
_ROUTER_AUTO_MODEL = "openrouter/auto"
_ROUTER_PRESET_MARKER = "@preset/"
# Vertex's OpenAI-compatible surface may name a model by its (project-scoped) publisher path.
_VERTEX_PUBLISHER_PATH = re.compile(r"^(?:projects/[^/]+/locations/[^/]+/)?publishers/([^/]+)/models/")
# Responses-style terminal states that mean no answer, even when partial text came back.
_FAILED_STATUSES = frozenset({"failed", "cancelled"})  # not "incomplete": a token cap still answered
# Azure reports the base model, not the deployment name; hosts padded like
# ``anthropic_endpoints._is_azure_anthropic_endpoint`` so sovereign clouds match too.
_AZURE_HOST_LABELS = (".openai.azure.", ".services.ai.azure.", ".cognitiveservices.azure.")
# A provider may answer with a dated snapshot of the requested alias (gpt-4o → gpt-4o-2024-08-06,
# claude-sonnet-4 → claude-sonnet-4-20250514) or a ``-latest`` pointer; anything else is another model.
_SNAPSHOT_SUFFIX = r"(?:-\d{4}-?\d{2}-?\d{2}|-\d{4}|-latest)?"


@dataclasses.dataclass
class EntryCheck:
    """One fallback entry's verdict; ``inference_ready`` stays None until ``--live`` probes it."""

    provider: str
    model: str
    base_url: str = ""
    api_mode: str = ""
    auth_source: str = ""
    configured: bool = False
    auth_ready: bool = False
    inference_ready: Optional[bool] = None
    latency_ms: Optional[int] = None
    model_returned: str = ""
    error_class: str = ""
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.configured and self.auth_ready and self.inference_ready is not False

    @property
    def status(self) -> str:
        """``ready`` / ``ready (static)``, else the ``error_class`` every not-ready path sets."""
        if self.ready:
            return "ready" if self.inference_ready else "ready (static)"
        return self.error_class


def _redacted(text: Any, secrets: tuple[str, ...] = (), max_chars: Optional[int] = None) -> str:
    """Known credential values removed first (pattern redaction misses keys without a known
    prefix), then the support redactor."""
    from hermes_cli.debug_redaction import redact_debug_support_text
    text = str(text or "")
    for secret in secrets:
        text = text.replace(secret, "***")
    return redact_debug_support_text(text, max_chars=max_chars)


def _secret_values(*candidates: Any) -> tuple[str, ...]:
    return tuple(c for c in candidates if isinstance(c, str) and len(c) >= 8)


def _reported_model_is_evidence(provider: str, model: str, client: Any) -> bool:
    """Whether a response's ``model`` field can prove which model answered.

    Only a real OpenAI SDK client hands back the provider's own field: the Anthropic Messages and
    Codex Responses adapters rebuild the response with the REQUESTED model. Local or private-network
    servers (llama.cpp, LM Studio, Ollama, LAN boxes) often report a placeholder or a weights path,
    and a Bedrock ARN is not what the endpoint reports back.
    """
    from openai import OpenAI
    from hermes_agent.agent.model_metadata import is_local_endpoint
    from providers import get_provider_profile
    from utils import base_url_hostname
    base_url = str(getattr(client, "base_url", "") or "")
    host = f".{base_url_hostname(base_url)}."
    lowered = model.strip().lower()
    is_azure = (getattr(get_provider_profile(provider), "name", provider) == "azure-foundry"  # and its aliases
                or any(label in host for label in _AZURE_HOST_LABELS))
    return (isinstance(client, OpenAI) and not is_local_endpoint(base_url) and not is_azure
            and not lowered.startswith(("arn:", _ROUTER_AUTO_MODEL)) and _ROUTER_PRESET_MARKER not in lowered)


def _canonical_model_id(model_id: str) -> str:
    return _VERTEX_PUBLISHER_PATH.sub(r"\1/", model_id.strip().lower())


def answered_by_requested_model(requested: str, returned: str) -> bool:
    """True when ``returned`` (the provider-reported model) is ``requested`` or a dated snapshot of it.

    A namespace is compared when both ids carry one (``org-a/m`` is not ``org-b/m``); a router may
    drop the requested one (``anthropic/claude-x`` answered as ``claude-x``). A ``:tag`` the provider
    reports must equal the requested tag (``qwen3:8b`` is not ``qwen3:32b``); a router may drop it
    (``x/llama-3:free`` answered as ``llama-3``). An empty ``returned`` counts as a match: some
    providers omit the field and adapter wires hide it (:func:`_reported_model_is_evidence`).
    """
    if not returned:
        return True
    want, _, want_tag = _canonical_model_id(requested).partition(":")
    got, _, got_tag = _canonical_model_id(returned).partition(":")
    if got_tag and got_tag != want_tag:
        return False
    if "/" not in want or "/" not in got:
        want, got = want.rsplit("/", 1)[-1], got.rsplit("/", 1)[-1]
    return bool(re.fullmatch(re.escape(want) + _SNAPSHOT_SUFFIX, got))


def _without_sdk_retries(client: Any) -> Any:
    """``client`` with OpenAI SDK retries off, so one probe is one request. Some builders keep the
    SDK default (Vertex, Bedrock Mantle), including inside the Codex Responses wrapper."""
    from openai import OpenAI
    from hermes_agent.agent.auxiliary_client import CodexAuxiliaryClient
    if isinstance(client, OpenAI):
        return client.with_options(max_retries=0)
    if isinstance(client, CodexAuxiliaryClient) and isinstance(client._real_client, OpenAI):
        return CodexAuxiliaryClient(client._real_client.with_options(max_retries=0), client.chat.completions._model)
    return client


def check_entry_static(entry: dict[str, Any]) -> EntryCheck:
    """Resolve ``entry`` the way the fallback switch does; never sends an inference request."""
    from hermes_cli.auth import AuthError
    from hermes_cli.fallback_config import fallback_entry_runtime_kwargs
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.runtime_readiness import missing_credentials_suffix, runtime_has_credentials

    # Lowercased as the swap's callers pass it, so ``provider: MoA`` is ``moa`` everywhere below.
    result = EntryCheck(provider=str(entry.get("provider") or "").strip().lower(), model=str(entry.get("model") or ""))
    try:
        runtime = resolve_runtime_provider(**fallback_entry_runtime_kwargs(entry))
    except AuthError as exc:
        # An unknown provider is a config error; every other AuthError is a credential problem,
        # which is also what makes the real resolution-time walker skip the entry.
        result.configured = exc.code != "invalid_provider"
        result.error_class, result.error = ("auth" if result.configured else "config"), _redacted(exc)
        return result
    except Exception as exc:  # health: allow BLE001 -- per-entry diagnostic boundary; logged with traceback
        logger.debug("fallback check: %s/%s did not resolve", result.provider, result.model, exc_info=True)
        result.error_class, result.error = "config", _redacted(exc) or type(exc).__name__
        return result
    result.configured = True
    from hermes_agent.agent.chat_completion_helpers import _fallback_api_mode_hint
    result.base_url = _redacted(runtime.get("base_url"))
    explicit, hinted = _fallback_api_mode_hint(entry, result.provider, str(entry.get("base_url") or "").strip())
    # bind_route_entry keeps an explicit or non-chat hint and re-detects only a default chat one.
    result.api_mode = hinted if explicit or hinted != "chat_completions" else str(runtime.get("api_mode") or hinted)
    result.auth_source = str(runtime.get("source") or "")
    result.auth_ready = runtime_has_credentials(runtime)
    if not result.auth_ready:
        result.error_class = "auth"
        result.error = f"No usable credentials found for {result.provider}.{missing_credentials_suffix(result.provider)}"
    return result


def check_entry_live(entry: dict[str, Any], result: EntryCheck) -> EntryCheck:
    """Send one minimal completion for ``entry`` (already statically ready) and record the verdict."""
    from hermes_agent.agent.auxiliary_client import _build_call_kwargs, _create_with_progress_once, _elapsed_ms, _provider_requires_stream
    from hermes_cli.fallback_config import resolve_entry_api_key
    from hermes_cli.timeouts import get_provider_request_timeout
    from hermes_agent.agent.error_classifier import classify_api_error
    from hermes_agent.agent.route_binding import resolve_route_entry_client

    provider, secrets, started = result.provider, _secret_values(resolve_entry_api_key(entry)), None
    try:
        # raw_codex=False: the chat-completions adapter, so one create() call fits every wire.
        route = resolve_route_entry_client(entry, provider, result.model, raw_codex=False)
        client, model = route.client, route.model
        if client is None:
            result.inference_ready, result.error_class = False, "auth"
            result.error = "the provider router built no client for this entry (no usable credential)"
            return result
        client = _without_sdk_retries(client)
        secrets += _secret_values(getattr(client, "api_key", None))
        base_url = str(getattr(client, "base_url", "") or "")
        timeout = get_provider_request_timeout(provider, model) or _LIVE_TIMEOUT_S
        kwargs = _build_call_kwargs(provider, model, list(_LIVE_PROMPT),
                                    max_tokens=_LIVE_MAX_TOKENS, timeout=timeout, base_url=base_url)
        started = time.monotonic()  # provider latency, not SDK import / client construction
        raw = _create_with_progress_once(client, kwargs, None, provider=provider,
                                         force_stream=_provider_requires_stream(provider, base_url))
        result.latency_ms = _elapsed_ms(started)
        # Inside the boundary: an error envelope without an error object raises and is classified
        # here, and nothing in an odd body can fail the whole command.
        _judge_answer(result, raw, model, client, secrets)
    except Exception as exc:  # health: allow BLE001 -- the probe reports every provider failure as its verdict
        if started is not None and result.latency_ms is None:  # none for an entry that sent no request
            result.latency_ms = _elapsed_ms(started)
        verdict = classify_api_error(exc, provider=provider, model=result.model)
        result.inference_ready, result.error_class = False, f"live_{verdict.reason.value}"
        result.error = _redacted(exc, secrets, max_chars=300) or type(exc).__name__
        # No traceback: exception text can carry the key, and log redaction is pattern-based too.
        logger.debug("fallback check: live probe of %s/%s failed: %s", result.provider, result.model, result.error)
    return result


def _body_failure(raw: Any) -> tuple[Optional[dict[str, Any]], bool]:
    """``(provider error, failed status)`` from the body AND a gateway ``{"data": ...}`` envelope,
    read before any normalization: unwrapping rebuilds the answer without ``status`` / ``error``."""
    from hermes_agent.agent import auxiliary_call_outcome as outcome
    from hermes_agent.agent.auxiliary_client import _field
    data = _field(raw, "data")
    layers = [raw, data] if isinstance(data, dict) else [raw]
    choices = _field(raw, "choices")
    if isinstance(choices, list) and choices:  # OpenRouter puts an upstream failure on the choice
        layers.append(choices[0])
    # Some providers always send an ``error`` object; one whose fields are all empty is not a failure.
    embedded = next((e for e in map(outcome.embedded_provider_error, layers) if e and any(e.values())), None)
    failed = any(str(_field(layer, "status") or "").lower() in _FAILED_STATUSES for layer in layers)
    return embedded, failed


def _judge_answer(result: EntryCheck, raw: Any, model: str, client: Any, secrets: tuple[str, ...]) -> None:
    """Verdict for an answered request: a provider error in the body, no usable completion, another
    model, or ready."""
    from hermes_agent.agent import auxiliary_call_outcome as outcome
    from hermes_agent.agent.auxiliary_client import _recover_aux_response_message, _unwrap_data_envelope, extract_content_or_reasoning
    from hermes_agent.agent.errors import InvalidProviderResponseError
    from hermes_agent.agent.transports.chat_completions import is_router_timeout_shim
    provider = result.provider  # already lowercased by check_entry_static
    embedded, failed = _body_failure(raw)
    response = None
    if embedded is None and not failed:
        response = _unwrap_data_envelope(raw)
        # Responses-style text (``output_text`` / ``output[]``) outside ``choices``, as the aux path accepts.
        if not getattr(response, "choices", None):
            try:
                response = _recover_aux_response_message(response) or response
            except (TypeError, ValueError, AttributeError):  # a malformed output[] is no answer
                response = None
    returned = str(getattr(response, "model", "") or "") if _reported_model_is_evidence(provider, model, client) else ""
    result.model_returned = _redacted(returned, secrets)
    choices = getattr(response, "choices", None)
    if embedded is not None:  # an HTTP-200 relay of an upstream failure: classify it like the aux path
        error = InvalidProviderResponseError(str(embedded.get("message") or embedded))
        setattr(error, outcome.PROVIDER_ERROR_ATTR, embedded)
        result.inference_ready = False
        result.error_class = f"live_{outcome.error_class(error, provider=provider, model=result.model)}"
        result.error = _redacted(error, secrets, max_chars=300)
    # Partial text under a failed status, or beside a choice-level error (OpenRouter's
    # ``finish_reason: "error"``), is not an answer. An empty stream is re-aggregated into a choice
    # with an empty message: content (or reasoning) is the evidence a model ran, not the shape.
    elif (failed or not choices or getattr(choices[0], "message", None) is None
            or str(getattr(choices[0], "finish_reason", "") or "").lower() == "error"
            or is_router_timeout_shim(response) or not extract_content_or_reasoning(response)):
        result.inference_ready, result.error_class = False, "live_response"
        result.error = "the provider answered without a usable completion"
    elif not answered_by_requested_model(model, returned):
        result.inference_ready, result.error_class = False, "model_mismatch"
        result.error = _redacted(f"requested {model!r} but {returned!r} answered", secrets)
    else:
        result.inference_ready = True


def run_checks(chain: list[dict[str, Any]], *, live: bool) -> list[EntryCheck]:
    """Static checks in chain order; with ``live``, probe the statically ready entries in parallel."""
    from hermes_cli.left_core_migration import installs_held
    with installs_held():  # a diagnostic: resolving a provider whose plugin left core must not install it
        results = [check_entry_static(entry) for entry in chain]
    if live:
        from hermes_cli.doctor_connectivity import run_probes
        # Not one fixed route, so one request proves nothing: ``auto`` picks its backend per call (and can
        # walk the chain); ``moa`` is a fan-out preset whose facade only an agent binds. Both stay static.
        # IMDS stays on: an EC2 instance role is a real Bedrock credential for an inference request.
        run_probes([(r.provider, functools.partial(check_entry_live, entry, r))
                    for entry, r in zip(chain, results) if r.ready and r.provider not in _NOT_PROBED],
                   aws_imds=True)
    return results


def _print_table(results: list[EntryCheck], *, live: bool) -> None:
    def mark(value: Optional[bool]) -> str:
        return "—" if value is None else "ok" if value else "FAIL"

    print()
    print(f"  {'#':<3}{'Provider / model':<44}{'Config':<8}{'Auth':<8}{'Live':<8}{'Latency':<10}Status")
    for index, r in enumerate(results, 1):
        latency = f"{r.latency_ms} ms" if r.latency_ms is not None else "—"
        label = f"{r.provider} / {r.model}"
        print(f"  {index:<3}{label[:43]:<44}{mark(r.configured):<8}{mark(r.auth_ready if r.configured else None):<8}"
              f"{mark(r.inference_ready):<8}{latency:<10}{r.status}")
        if r.error:
            print(f"     {r.error}")
    print()
    if not live:
        print("  Static check only (no inference request sent). Use --live to send one minimal request per entry.")
        print()


def cmd_fallback_check(args) -> None:
    """``hermes fallback check [--live] [--json]``; exits 1 when any entry is not ready."""
    from hermes_cli.config import load_config_readonly
    from hermes_cli.fallback_config import get_fallback_chain
    from hermes_cli.profiles import get_active_profile_name

    chain = get_fallback_chain(load_config_readonly())
    results = run_checks(chain, live=args.live)
    ready = all(r.ready for r in results)
    if args.json:
        print(json.dumps({
            "profile": get_active_profile_name(), "live": bool(args.live), "ready": ready,
            "checked_at": datetime.now(UTC).isoformat(),
            "results": [{**dataclasses.asdict(r), "ready": r.ready, "status": r.status} for r in results],
        }, indent=2))
    elif not results:
        print("\n  No fallback providers configured. Add one with: hermes fallback add\n")
    else:
        _print_table(results, live=args.live)
    if not ready:
        raise SystemExit(1)
