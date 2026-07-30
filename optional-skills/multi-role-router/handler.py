"""
multi-role-router — reference hook for message:pre_route auto-routing.

Install: copy this directory to ~/.hermes/hooks/multi-role-router/
Configure roles in ~/.hermes/config.yaml under `roles:` (see README).

The classifier uses the auxiliary LLM slot configured in config.yaml
(auxiliary.triage_specifier or falls back to the compression model).
It passes the last N exchanges as context so continuations ("ok thanks",
"and what about X?") stay in the current session rather than switching.

Event: message:pre_route
Return: {"decision": "switch_session", "session_id": "<id>"} or None/{}
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOOK_DIR = Path(__file__).parent
META_FILE = HOOK_DIR / "meta.yaml"
_META_LOCK = threading.Lock()

# How many recent exchanges to pass as continuation context
HISTORY_WINDOW = 3

# Signals that a message is almost certainly a continuation of the current
# session rather than a topic switch.  Skip the classifier when we see these.
CONTINUATION_PATTERNS: List[str] = [
    r"^(ok|okay|thanks|thank you|got it|great|sure|sounds good|perfect|cool|nice|alright|yep|yup|yeah|nope|no|yes)\W*$",
    r"^(and|also|but|so|then|what about|how about)\b",
    r"^(can you|could you|please|can we)\b.{0,40}(also|too|as well)\b",
]
CONTINUATION_RE = re.compile(
    r"^(?:"
    r"ok(?:\s+(?:thanks?|got\s+it|sounds?\s+good|cool|fine|great|perfect|sure|works?))?"
    r"|thanks?\s*(?:you)?"
    r"|got\s+it"
    r"|sounds?\s+good"
    r"|makes?\s+sense"
    # Only conjunction-style openers ("and also", "but what about") are safe
    # continuations; "what/how/why/when/where/which" at the start of a sentence
    # are almost always new-topic questions and must reach the classifier.
    r"|(?:and|also|but|so|then)\b.{0,80}"
    r"|(?:what|how|why)\s+about\b.{0,60}"
    r")\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_CONTINUATION_RE = CONTINUATION_RE

# Default role definitions — override in config.yaml under `roles:`
DEFAULT_ROLES: Dict[str, Dict[str, Any]] = {
    "code-worker": {
        "description": (
            "Software development, debugging, code review, writing or modifying "
            "source files, build systems, tests, git operations, package management."
        ),
        "keywords": ["code", "debug", "function", "class", "bug", "test", "git", "build"],
    },
    "knowledge-worker": {
        "description": (
            "Research, summarization, document writing, Q&A, information retrieval, "
            "web search, analysis of text/data, drafting prose or reports."
        ),
        "keywords": ["research", "summarize", "explain", "write", "document", "find"],
    },
    "ml-worker": {
        "description": (
            "Machine learning experiments, model training, dataset preparation, "
            "fine-tuning, evaluation metrics, GPU/TPU job management, MLflow/W&B."
        ),
        "keywords": ["train", "model", "dataset", "epoch", "loss", "accuracy", "fine-tune"],
    },
    "ops-worker": {
        "description": (
            "DevOps, infrastructure, containers, CI/CD, shell scripting, server "
            "management, cloud deployments, monitoring, on-call operations."
        ),
        "keywords": ["deploy", "docker", "kubernetes", "server", "pipeline", "infra", "shell"],
    },
    "default": {
        "description": (
            "General-purpose tasks that don't clearly fit another role, or when "
            "uncertain which role applies."
        ),
        "keywords": [],
    },
}


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _load_meta() -> Dict[str, Any]:
    """Load persistent hook state from meta.yaml (role→session_id map + history).

    Discards files that are partially written (i.e. unparseable) and returns {}.
    """
    if not META_FILE.exists():
        return {}
    try:
        text = META_FILE.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            # Treat empty or non-dict files (e.g. truncated writes) as clean slate
            logger.warning("[multi-role-router] meta.yaml contained non-dict content; resetting.")
            return {}
        return data
    except Exception as exc:
        logger.warning("[multi-role-router] Could not read meta.yaml (discarding): %s", exc)
        return {}


def _save_meta(data: Dict[str, Any]) -> None:
    """Persist hook state to meta.yaml using an atomic write (temp + os.replace).

    The temp file is created in META_FILE.parent (not HOOK_DIR) so that
    os.replace is guaranteed to be atomic — both paths must be on the same
    filesystem for a cross-file atomic rename.
    """
    try:
        content = yaml.safe_dump(data, default_flow_style=False, allow_unicode=True)
        target = Path(META_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(target.parent), prefix=".meta_tmp_", suffix=".yaml"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp_path, str(META_FILE))
        except Exception:
            # Clean up temp file if the replace failed
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception as exc:
        logger.warning("[multi-role-router] Could not write meta.yaml: %s", exc)


def _update_meta_session(role: str, session_id: str, current_role: str, message: str, response: str) -> None:
    """Update meta.yaml with the new current session and append to history.

    Uses a threading lock + atomic write so concurrent hook invocations cannot
    interleave their load/mutate/save cycles or leave a partial file on disk.
    """
    with _META_LOCK:
        meta = _load_meta()
        meta.setdefault("sessions", {})[role] = session_id
        meta["current_role"] = role
        # Rolling history — list of {role, user, assistant} dicts
        history: List[Dict[str, str]] = meta.get("history", [])
        history.append({"role": current_role, "user": message[:300], "assistant": response[:300]})
        # Trim to window
        meta["history"] = history[-(HISTORY_WINDOW * 2):]
        _save_meta(meta)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_hermes_config() -> Dict[str, Any]:
    """Load ~/.hermes/config.yaml, returning {} on any error."""
    try:
        # Use hermes_cli if available (running inside the gateway process)
        from hermes_cli.config import get_hermes_home
        config_path = get_hermes_home() / "config.yaml"
    except ImportError:
        config_path = Path.home() / ".hermes" / "config.yaml"

    if not config_path.exists():
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.warning("[multi-role-router] Could not read config.yaml: %s", exc)
        return {}


def _get_roles(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return role definitions from config.

    Non-empty user-defined roles REPLACE the defaults entirely so users can
    define a clean, custom role set without inheriting built-in roles they
    don't want.  Individual role entries still fall back to matching default
    fields so partial definitions work as expected.  When the config is
    missing or invalid, DEFAULT_ROLES is returned unchanged.
    """
    user_roles = config.get("roles", {})
    if not isinstance(user_roles, dict) or not user_roles:
        return DEFAULT_ROLES
    merged = {}
    for name, defn in user_roles.items():
        if isinstance(defn, dict):
            merged[name] = {**DEFAULT_ROLES.get(name, {}), **defn}
    return merged or DEFAULT_ROLES


def _get_auxiliary_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract auxiliary.triage_specifier or fall back to auxiliary.compression.

    Guards against null/non-dict values at every level so a malformed
    config.yaml cannot cause an AttributeError.
    """
    aux = config.get("auxiliary", {})
    if not isinstance(aux, dict):
        aux = {}
    triage = aux.get("triage_specifier", {})
    if isinstance(triage, dict) and triage:
        return triage
    # Fall back to compression slot (cheapest text task)
    comp = aux.get("compression", {})
    return comp if isinstance(comp, dict) else {}


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _build_classifier_prompt(
    roles: Dict[str, Dict[str, Any]],
    current_role: str,
    history: List[Dict[str, str]],
    message: str,
) -> str:
    """Build the compact single-turn prompt sent to the triage LLM."""
    role_list = []
    for name, defn in roles.items():
        role_list.append(f"- {name}: {defn.get('description', '')}")
    roles_text = "\n".join(role_list)

    history_text = ""
    if history:
        lines = []
        for entry in history[-HISTORY_WINDOW:]:
            r = entry.get("role", "?")
            u = entry.get("user", "")
            a = entry.get("assistant", "")
            lines.append(f"[{r}] user: {u}\n[{r}] assistant: {a}")
        history_text = "\n\n".join(lines)

    return f"""You are a message router. Classify the new message into exactly one role.

ROLES:
{roles_text}

CURRENT ROLE: {current_role}

{"RECENT EXCHANGES:" + chr(10) + history_text if history_text else "(no prior context)"}

NEW MESSAGE: {message}

Respond with ONLY the role name, no punctuation, no explanation.
If the message is a continuation of the current topic, respond with the current role name.
Valid role names: {", ".join(roles.keys())}"""


def _call_auxiliary_llm(prompt: str, aux_cfg: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
    """Call the auxiliary LLM and return the raw text response, or None on failure."""
    # Prefer the gateway-internal auxiliary_client (fast, handles all providers)
    try:
        from agent.auxiliary_client import call_llm
        resp = call_llm(
            task="compression",  # use compression slot — cheap text, no vision
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32,
            timeout=aux_cfg.get("timeout", 15),
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except ImportError:
        pass  # not running inside gateway — fall back to direct HTTP
    except Exception as exc:
        logger.warning("[multi-role-router] auxiliary_client call failed: %s", exc)
        return None

    # Direct HTTP fallback (used when the hook runs outside the gateway process)
    model_section = config.get("model", {})
    if not isinstance(model_section, dict):
        model_section = {}
    base_url = (
        aux_cfg.get("base_url", "")
        or model_section.get("base_url", "")
    ).rstrip("/")
    api_key = (
        aux_cfg.get("api_key", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
    )
    model = aux_cfg.get("model", "") or "google/gemini-flash-1.5-8b"

    if not base_url:
        logger.warning("[multi-role-router] No base_url configured for auxiliary LLM; cannot classify.")
        return None

    try:
        import httpx

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 32,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with httpx.Client(timeout=aux_cfg.get("timeout", 15)) as client:
            r = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip() if content else None
    except Exception as exc:
        logger.warning("[multi-role-router] Direct HTTP auxiliary call failed: %s", exc)
        return None


def _classify_message(
    message: str,
    current_role: str,
    history: List[Dict[str, str]],
    roles: Dict[str, Dict[str, Any]],
    aux_cfg: Dict[str, Any],
    config: Dict[str, Any],
) -> str:
    """Return the best role name for *message*, defaulting to current_role on any failure."""
    prompt = _build_classifier_prompt(roles, current_role, history, message)
    try:
        raw = _call_auxiliary_llm(prompt, aux_cfg, config)
    except Exception:
        logger.warning("[multi-role-router] LLM call failed, keeping current role", exc_info=True)
        return current_role
    if not raw:
        return current_role

    # Clean up and validate the response
    candidate = raw.lower().strip().strip(".,;:\"'")
    if candidate in roles:
        return candidate

    # Fuzzy: match role names with word boundaries, longest first to avoid
    # shorter names matching as substrings of longer ones (e.g. "ml" inside
    # "ml-worker").
    lowered = raw.lower()
    for role_name in sorted(roles, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(role_name.lower())}(?!\w)", lowered):
            return role_name

    logger.debug("[multi-role-router] Unrecognised role '%s', keeping current '%s'", raw, current_role)
    return current_role


# ---------------------------------------------------------------------------
# Hook entry point
# ---------------------------------------------------------------------------

async def handle(event_type: str, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    message:pre_route handler.

    Context keys (see gateway/hooks.py):
      platform, user_id, chat_id, thread_id, chat_type,
      session_id, session_key, message

    Returns:
      {"decision": "switch_session", "session_id": "<id>"}  -- redirect
      None or {}                                             -- pass-through
    """
    if event_type != "message:pre_route":
        return None

    message: str = context.get("message", "") or ""
    current_session_id: str = context.get("session_id", "") or ""

    if not message.strip():
        return None

    # ------------------------------------------------------------------
    # Fast path: continuation detection (skip classifier entirely)
    # ------------------------------------------------------------------
    if _CONTINUATION_RE.match(message.strip()):
        logger.debug("[multi-role-router] Continuation detected, staying in current session.")
        return None

    # ------------------------------------------------------------------
    # Load config + persistent state
    # ------------------------------------------------------------------
    config = _load_hermes_config()

    # Respect the /role auto off flag
    router_config = config.get("multi_role_router")
    if not isinstance(router_config, dict):
        router_config = {}
    if not router_config.get("auto", True):
        return None

    roles = _get_roles(config)
    aux_cfg = _get_auxiliary_config(config)
    meta = _load_meta()

    current_role: str = meta.get("current_role", "default")
    if current_role not in roles:
        current_role = "default"

    history: List[Dict[str, str]] = meta.get("history", [])

    # ------------------------------------------------------------------
    # Classify
    # ------------------------------------------------------------------
    try:
        target_role = _classify_message(
            message=message,
            current_role=current_role,
            history=history,
            roles=roles,
            aux_cfg=aux_cfg,
            config=config,
        )
    except Exception:
        logger.warning("[multi-role-router] Classification failed; staying put.", exc_info=True)
        return None

    logger.debug(
        "[multi-role-router] message=%r current_role=%s target_role=%s",
        message[:80],
        current_role,
        target_role,
    )

    # ------------------------------------------------------------------
    # Decide — mutate + save under lock to prevent concurrent interleaving.
    # _META_LOCK serialises threads within this process; if hook instances
    # run in separate processes, a filesystem lock would be needed instead.
    # ------------------------------------------------------------------
    with _META_LOCK:
        # Reload meta inside the lock so we act on the freshest state
        # (another thread may have saved between our initial load and here).
        meta = _load_meta()
        sessions: Dict[str, str] = meta.setdefault("sessions", {})
        # Record the current session under the current_role so future
        # switches can find it again.  Done before any potential redirect.
        if current_session_id:
            sessions[current_role] = current_session_id

        if target_role == current_role:
            # No switch needed — save current session mapping and pass through
            _save_meta(meta)
            return None

        target_session_id = sessions.get(target_role, "")

        if not target_session_id:
            # No existing session for this role — let the gateway create a new
            # one by updating our bookkeeping to reflect the impending role
            # change without redirecting (the gateway's normal session-creation
            # path runs).  The NEXT message will find this role in meta.yaml
            # and, once a session_id is established, route correctly.
            meta["current_role"] = target_role
            _save_meta(meta)
            logger.info(
                "[multi-role-router] New role '%s' — no prior session, gateway will create one.",
                target_role,
            )
            return None

        if target_session_id == current_session_id:
            _save_meta(meta)
            return None

        # Switch!
        meta["current_role"] = target_role
        _save_meta(meta)

    logger.info(
        "[multi-role-router] Switching %s → %s (session %s → %s)",
        current_role,
        target_role,
        current_session_id,
        target_session_id,
    )
    return {"decision": "switch_session", "session_id": target_session_id}
