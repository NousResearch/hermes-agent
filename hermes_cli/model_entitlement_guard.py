"""Entitlement gate for a model pick that is about to become the profile default.

WHY THIS EXISTS (kanban t_27cf7a7b, 2026-09-28)
-----------------------------------------------
On 2026-09-28 10:24 a model switch persisted ``model.default = glm-5.3-flashx``
(provider ``custom:zai``, base_url the Z.AI coding endpoint) into
``profiles/hephaestus/config.yaml``. The slug is not in the account's plan, so
every session afterwards paid

    429 {"error":{"code":"1311","message":"Your current subscription plan does not
         yet include access to GLM-5.3-FlashX"}}

and then a fallback hop before it could work. Nothing on the persist path probed
the pick: the endpoint's ``/models`` catalog had already been verified (the model
is listed, it is simply not entitled) and the health plane read 429 as a
self-healing rate-limit window.

WHAT THIS DOES
--------------
``ensure_pick_entitled`` live-probes the model ON THE ENDPOINT THE PICK WOULD RUN
ON — one tiny completion, sent on THE SAME WIRE the session will use
(``/chat/completions``, ``/responses`` or ``/v1/messages``, per
``hermes_cli.providers.determine_api_mode``) — and, when the provider's own error
body proves the model unusable for this account, refuses to let the pick become
the default:

  * the config write never happens, so the profile is left on its previous
    default and nothing has to be reverted;
  * the session may still run the pick — ``hermes -m <model>`` or a mid-session
    ``/model`` — which is how a temporary override was always expressed;
  * ``HERMES_ALLOW_UNENTITLED_PICK=1`` turns the refusal into a warning for
    anyone who wants the old warn-and-persist behaviour:
    ``HERMES_ALLOW_UNENTITLED_PICK=1 hermes model``.

WHERE IT IS WIRED (the persist surfaces, by construction)
--------------------------------------------------------
  * ``model_setup_flows_common._persist_model`` — the standard setup step, and
    through it every ``_finish_model`` caller (openrouter, ai-gateway, copilot,
    zai/GLM, deepseek, …);
  * ``model_setup_flows_common._activate_provider_model`` — the OAuth flows
    (openai-codex, xai-oauth, qwen-oauth, minimax-oauth);
  * ``model_setup_flows._nous_persist_selection`` and ``auth_nous._login_nous``
    — the Nous sign-in path, which persists its own selection;
  * ``model_setup_flows_custom`` — both custom-endpoint flows;
  * ``model_switch.persist_model_selection`` — ``/model --global`` and the
    dashboard's main-slot pick (``web_server_config`` routes through it).

EXEMPT BY DESIGN: ``moa`` (``is_exempt_provider``) — its ``model.default`` is a
preset NAME resolved against the user's own configured providers, so there is no
endpoint to probe. The exemption is explicit, not an omission.

FAIL-OPEN BY DESIGN. The probe is an extra HTTP call to a third party, so every
way it can be inconclusive — no endpoint or credential to test, a local endpoint,
a transport error, a 5xx, a 429 that is a real throttle — allows the pick and
says nothing. Only a *proven* refusal (an entitlement/billing verdict from
``agent.error_classifier``, or an unambiguous 402/410) blocks. A bare 404 does NOT
block either: a wrong-wire 404 must never be mistaken for an entitlement wall,
and a real 404 wall (a retired free slug) blocks through its error body. A guard
that blocked on a flaky network or on a protocol mismatch would be worse than the
bug it fixes.

The classification is the runtime's own: ``agent.error_classifier`` already owns
``FailoverReason.model_entitlement`` (the zai ``1311`` code was added to it with
this card), so the gate and the fallback path agree on what "not entitled" means.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Seconds allowed for the entitlement probe. Short: it runs inside an
#: interactive pick, and an inconclusive probe is a pass, not a block.
PROBE_TIMEOUT = 8.0

#: Set to any value to restore warn-and-persist behaviour for a pick the probe
#: proves unusable (the pre-2026-09-28 behaviour, kept as an escape hatch).
POLICY_ENV = "HERMES_ALLOW_UNENTITLED_PICK"

#: Verdicts from ``agent.error_classifier`` that mean "this account cannot use
#: this model here" rather than "try again later".
REFUSAL_REASONS = frozenset({"model_entitlement", "billing"})

#: Statuses decisive on their own. ``402`` (payment required) and ``410`` (gone)
#: are unambiguous whatever the body says. ``404`` is deliberately NOT here: it is
#: also what a WRONG WIRE returns (the pick would run on ``/responses`` or
#: ``/v1/messages`` but a probe sent to ``/chat/completions``), so a bare 404 must
#: stay inconclusive. A 404 that really is an entitlement wall carries it in the
#: body ("no longer free", "requires available credits") and blocks through
#: ``REFUSAL_REASONS`` instead.
REFUSAL_STATUSES = frozenset({402, 410})

#: Providers whose default is a VIRTUAL preset name, not a wire route: there is
#: no endpoint to probe, so the gate cannot judge them and must not pretend to.
EXEMPT_PROVIDERS = frozenset({"moa"})

#: A local endpoint is the user's own server: no plan to be entitled by.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"})

_USER_AGENT = "hermes-cli/entitlement-guard"


class _ProbeError(Exception):
    """Minimal exception carrying the status + body ``classify_api_error`` reads."""

    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(json.dumps(body)[:200] if body else str(status_code))


def policy() -> str:
    """``"refuse"`` (default) or ``"warn"`` (``HERMES_ALLOW_UNENTITLED_PICK`` set)."""
    return "warn" if os.environ.get(POLICY_ENV, "").strip() else "refuse"


def is_exempt_provider(provider: str) -> bool:
    """True for a provider the gate cannot judge (a virtual preset, not a route).

    Only ``moa`` today: its ``model.default`` is a preset NAME resolved against
    the user's own configured providers — there is no endpoint the pick would run
    on, so a probe would test nothing. The exemption is stated here, in the
    module docstring and in ``tests/hermes_cli/test_model_entitlement_guard.py``
    so it reads as a decision rather than an oversight.
    """
    slug = (provider or "").strip().lower()
    if ":" in slug:
        slug = slug.split(":", 1)[1]
    return slug in EXEMPT_PROVIDERS


def block_message(model: str, *, provider: str = "", rationale: str = "") -> str:
    """The refusal copy. Names the session-override path and the escape hatch."""
    where = f"{provider}/" if provider else ""
    lines = [
        f"  ✗ {where}{model} was NOT saved as the default — this account is not entitled to it.",
    ]
    if rationale:
        lines.append(f"    {rationale}")
    lines += [
        "    Your previous default is unchanged. To use it for this session only:",
        f"      hermes -m {model}",
        f"    To save it anyway:  {POLICY_ENV}=1 hermes model",
    ]
    return "\n".join(lines)


def _is_local_endpoint(base_url: str) -> bool:
    try:
        host = urllib.parse.urlparse(base_url).hostname or ""
    except Exception:
        return True
    return host.lower() in _LOCAL_HOSTS


def _wire_for(provider: str, base_url: str, model: str) -> str:
    """The wire protocol this pick would actually run on.

    A probe must speak the SAME protocol the session will: an OAuth route that
    requires ``/responses`` or ``/v1/messages`` 404s a ``/chat/completions``
    request, and that 404 is about the probe, not about entitlement. Falls back
    to ``chat_completions`` — the transport that answers everywhere — when the
    provider table is unavailable.
    """
    try:
        from hermes_cli.providers import determine_api_mode

        return determine_api_mode(provider, base_url, model) or "chat_completions"
    except Exception:
        return "chat_completions"


def _wire_request(
    base_url: str, api_key: str, model: str, wire: str,
) -> tuple[str, dict, str | None]:
    """``(url, body, auth_header_name)`` for *wire*.

    ``auth_header_name`` is ``None`` for Bearer. Anthropic-Messages on a
    third-party endpoint (MiniMax) authenticates with Bearer, native Anthropic
    with ``x-api-key``; the adapter owns that choice, so mirror its rule rather
    than guessing on a host we do not recognise.
    """
    root = base_url.rstrip("/")
    ping = "ping"
    if wire == "codex_responses":
        return (root + "/responses",
                {"model": model, "input": [{"role": "user", "content": ping}],
                 "max_output_tokens": 16, "stream": False},
                None)
    if wire == "anthropic_messages":
        auth = "x-api-key"
        try:
            from agent.anthropic_endpoints import _is_minimax_anthropic_endpoint

            if _is_minimax_anthropic_endpoint(base_url):
                auth = None  # MiniMax uses Bearer even on its Messages route
        except Exception:
            auth = None  # unknown third-party Messages host: Bearer is the safer guess
        return (root + "/v1/messages",
                {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": ping}]},
                auth)
    return (root + "/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": ping}],
             "max_tokens": 1, "stream": False},
            None)


def _post_chat_completion(base_url: str, api_key: str, model: str, timeout: float,
                          *, provider: str = "") -> tuple[int, dict]:
    """One tiny completion on the wire *this pick* would run on.

    Returns ``(status_code, parsed_body)``. Raises on transport failure (the
    caller treats that as inconclusive). The credential is never logged or
    echoed.
    """
    wire = _wire_for(provider, base_url, model)
    url, payload, auth_header = _wire_request(base_url, api_key, model, wire)
    headers = {
        "Content-Type": "application/json",
        # python-urllib's default UA is WAF-blocked by some gateways (nous 1010).
        "User-Agent": _USER_AGENT,
    }
    if auth_header:
        headers[auth_header] = api_key
        if auth_header == "x-api-key":
            headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode(errors="replace")
            return resp.status, _parse_body(raw)
    except urllib.error.HTTPError as exc:
        raw = ""
        try:
            raw = exc.read().decode(errors="replace")
        except Exception:
            pass
        return exc.code, _parse_body(raw)


def _parse_body(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _classify(status: int, body: dict, *, model: str, provider: str, base_url: str) -> str:
    """Reason string for a failed probe, "" when the failure is not a refusal."""
    from agent.error_classifier import classify_api_error

    body = body or {}
    verdict = classify_api_error(
        _ProbeError(status, body), provider=provider, model=model, base_url=base_url)
    if verdict.reason.value in REFUSAL_REASONS:
        return str(verdict.message or "").strip()
    if status in REFUSAL_STATUSES:
        return str(verdict.message or "").strip() or f"HTTP {status}"
    if status == 429:
        # A 429 that is NOT an entitlement/billing verdict is a throttle window:
        # the model is usable, just busy. Never block on it.
        return ""
    return ""


def refusal_for_pick(
    model: str, *, provider: str = "", base_url: str = "", api_key: str = "",
    timeout: float = PROBE_TIMEOUT,
) -> Optional[str]:
    """Why *model* must not become the default on this endpoint, or ``None``.

    ``None`` covers both "entitled" and "could not prove otherwise".
    """
    if not model or not base_url or not api_key:
        return None
    if _is_local_endpoint(base_url) or is_exempt_provider(provider):
        return None
    try:
        status, body = _post_chat_completion(base_url, api_key, model, timeout,
                                             provider=provider)
    except Exception as exc:
        logger.debug("entitlement probe inconclusive for %s/%s: %s", provider, model, exc)
        return None
    if 200 <= status < 300:
        return None
    if status >= 500:
        # A server fault says nothing about entitlement.
        return None
    try:
        return _classify(status, body, model=model, provider=provider, base_url=base_url) or None
    except Exception as exc:  # classifier unavailable -> never block a pick
        logger.debug("entitlement classification failed for %s: %s", model, exc)
        return None


def ensure_pick_entitled(
    model: str, *, provider: str = "", base_url: str = "", api_key: str = "",
    timeout: float = PROBE_TIMEOUT,
) -> bool:
    """True when the pick may be persisted; False when the gate refused it.

    Prints the refusal copy itself (the flows only need to stop writing).
    """
    try:
        reason = refusal_for_pick(
            model, provider=provider, base_url=base_url, api_key=api_key, timeout=timeout)
    except Exception as exc:  # never break model selection
        logger.debug("entitlement gate skipped for %s: %s", model, exc)
        return True
    if not reason:
        return True
    rationale = " ".join(str(reason).split())[:200]
    if policy() == "warn":
        print(f"  ⚠ {provider + '/' if provider else ''}{model} may not be usable on this account:")
        print(f"    {rationale}")
        print(f"    Saving it anyway ({POLICY_ENV} is set).")
        return True
    print(block_message(model, provider=provider, rationale=rationale))
    return False


def resolve_endpoint(provider: str, *, model_cfg: Optional[dict] = None) -> tuple[str, str]:
    """``(base_url, api_key)`` this home would use for *provider*, or ``("", "")``.

    Reads the home's own config (``providers.<slug>`` first, then the active
    ``model`` section when it names the same provider). Returns empty strings when
    either half is unknown — the caller then skips the probe rather than testing
    the wrong endpoint. Credentials are resolved through the profile secret scope
    (never another profile's process env).
    """
    slug = (provider or "").strip()
    if ":" in slug:
        slug = slug.split(":", 1)[1]
    if not slug:
        return "", ""
    try:
        from hermes_cli.config import get_env_value, load_config

        cfg = load_config() or {}
    except Exception:
        return "", ""
    section = model_cfg if isinstance(model_cfg, dict) else (cfg.get("model") or {})
    if not isinstance(section, dict):
        section = {}

    entry: dict = {}
    for table in ("providers", "custom_providers"):
        table_cfg = cfg.get(table)
        if isinstance(table_cfg, dict) and isinstance(table_cfg.get(slug), dict):
            entry = dict(table_cfg[slug])
            break

    base_url = str(entry.get("base_url") or "")
    key_env = str(entry.get("key_env") or entry.get("api_key_env") or "")
    if str(section.get("provider") or "").strip() in (slug, f"custom:{slug}"):
        base_url = str(section.get("base_url") or base_url)
        key_env = key_env or str(section.get("key_env") or section.get("api_key_env") or "")

    api_key = str(entry.get("api_key") or "")
    if not api_key and str(section.get("provider") or "").strip() in (slug, f"custom:{slug}"):
        api_key = str(section.get("api_key") or "")
    if api_key.startswith("${") and api_key.endswith("}"):
        key_env = key_env or api_key[2:-1]
        api_key = ""

    if not api_key and key_env:
        api_key = _read_secret(key_env, get_env_value)
    return base_url, api_key


def _read_secret(name: str, fallback_reader: Any) -> str:
    """Secret for *name* from the profile scope, else the home's ``.env``."""
    try:
        from agent.secret_scope import get_secret_str

        value = get_secret_str(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    try:
        return str(fallback_reader(name) or "")
    except Exception:
        return ""
