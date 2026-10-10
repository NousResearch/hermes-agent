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
ON (one tiny chat completion) and, when the provider's own error body proves the
model unusable for this account, refuses to let the pick become the default:

  * the config write never happens, so the profile is left on its previous
    default and nothing has to be reverted;
  * the session may still run the pick — ``hermes -m <model>`` or a mid-session
    ``/model`` — which is how a temporary override was always expressed;
  * ``HERMES_ALLOW_UNENTITLED_PICK=1`` turns the refusal into a warning for
    anyone who wants the old warn-and-persist behaviour:
    ``HERMES_ALLOW_UNENTITLED_PICK=1 hermes model``.

FAIL-OPEN BY DESIGN. The probe is an extra HTTP call to a third party, so every
way it can be inconclusive — no endpoint or credential to test, a local endpoint,
a transport error, a 5xx, a 429 that is a real throttle — allows the pick and
says nothing. Only a *proven* refusal (an entitlement/billing verdict from
``agent.error_classifier``, or an unambiguous 402/404/410) blocks. A guard that
blocked on a flaky network would be worse than the bug it fixes.

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

#: Statuses that are decisive on their own (no body pattern needed).
REFUSAL_STATUSES = frozenset({402, 404, 410})

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


def _post_chat_completion(base_url: str, api_key: str, model: str, timeout: float) -> tuple[int, dict]:
    """One tiny chat completion; returns ``(status_code, parsed_body)``.

    Raises on transport failure (the caller treats that as inconclusive). The
    credential is never logged or echoed.
    """
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "stream": False,
    }).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # python-urllib's default UA is WAF-blocked by some gateways (nous 1010).
        "User-Agent": _USER_AGENT,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=payload, headers=headers)
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
    if _is_local_endpoint(base_url):
        return None
    try:
        status, body = _post_chat_completion(base_url, api_key, model, timeout)
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
