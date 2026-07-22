"""
Router plugin — pre-turn classifier that routes complex tasks to the orchestrator.

Architecture:
  User message
      │
      ▼
  pre_agent_dispatch hook fires
      │
      ▼
  Classifier (cheap aux model via auxiliary_client)
      │
      ├── simple → allow normal agent dispatch (Flash handles it)
      │
      └── complex → call orchestrator subprocess
                     → return routed result (bypass agent)

The classifier uses the configured triage_specifier aux model (default:
gemini-flash) — cheap and fast. The orchestrator subprocess runs
``hermes -p <profile> chat -q "<prompt>"`` with full tool access.

Configuration (in config.yaml):
  router:
    enabled: false              # master on/off (opt-in, default: false)
    classifier:
      task: "triage_specifier"  # aux task for classification
      model: ""                 # override aux model
    orchestrator:
      profile: "orchestrator"   # profile name for heavy tasks
      timeout: 600              # subprocess timeout (seconds)
      pass_history: true        # pass session history to orchestrator
    rules:
      always_simple: []         # regex patterns → always stay on Flash
      always_complex: []        # regex patterns → always go to Pro
      threshold: 0.5            # confidence threshold for complex routing
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────
_DEFAULT_CLASSIFIER_TASK = "triage_specifier"
_DEFAULT_ORCHESTRATOR_PROFILE = "orchestrator"
_DEFAULT_ORCHESTRATOR_TIMEOUT = 600

# ── Classification prompt ─────────────────────────────────────────────
_CLASSIFIER_SYSTEM_PROMPT = """You are a task complexity classifier for an AI agent routing system.
Analyze the user's message and classify it as either "simple" or "complex".

SIMPLE tasks (route to Flash — cheap/fast model):
- Greetings, casual chat, small talk
- Simple factual questions ("what is X", "when did Y happen")
- Status checks, yes/no questions
- Short translations, word definitions
- Simple file operations (list files, read a small file)
- Known-answer questions that don't need deep analysis
- Formatting/conversion requests that are mechanical

COMPLEX tasks (route to Pro — powerful model):
- Code implementation, debugging, architecture design
- System administration, configuration, deployment
- Data analysis, SQL queries, pandas operations
- Multi-step reasoning, planning, strategy
- Research, investigation requiring web search
- File editing, patching, refactoring
- Infrastructure changes, Proxmox operations
- Any task requiring tool use beyond basic read/search
- Mathematical proofs, algorithm design
- Contract/document generation or editing
- Financial analysis, tax calculations
- Real estate valuation or comparison

Respond with ONLY a JSON object:
{"classification": "simple"|"complex", "confidence": 0.0-1.0, "reason": "one-line reason"}"""


# ── Helpers ───────────────────────────────────────────────────────────

def _load_router_config() -> dict:
    """Load router configuration from config.yaml."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return cfg.get("router", {})
    except Exception:
        return {}


def _match_patterns(message: str, patterns: list[str]) -> bool:
    """Check if the message matches any of the given regex patterns."""
    if not patterns:
        return False
    for pattern in patterns:
        try:
            if re.search(pattern, message, re.IGNORECASE):
                return True
        except re.error:
            logger.warning("Invalid router pattern: %s", pattern)
    return False


def _classify_with_llm(message: str, task: str, model_override: str) -> dict:
    """Classify a message using the configured aux LLM."""
    try:
        from agent.auxiliary_client import call_llm
    except ImportError:
        logger.warning("auxiliary_client not available, defaulting to simple")
        return {"classification": "simple", "confidence": 0.0,
                "reason": "classifier unavailable"}

    messages = [
        {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": message[:4000]},
    ]

    try:
        call_kwargs: dict = {
            "task": task,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.0,
            "timeout": 30,
        }
        if model_override:
            call_kwargs["model"] = model_override
        response = call_llm(**call_kwargs)

        content = ""
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content or ""
        elif isinstance(response, str):
            content = response

        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

        result = json.loads(content)
        classification = str(result.get("classification", "simple")).lower()
        if classification not in ("simple", "complex"):
            classification = "simple"

        return {
            "classification": classification,
            "confidence": float(result.get("confidence", 0.5)),
            "reason": str(result.get("reason", "")),
        }
    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("Classifier failed: %s, defaulting to simple", exc)
        return {"classification": "simple", "confidence": 0.0,
                "reason": f"classifier error: {exc}"}


def classify(message: str, router_cfg: dict | None = None) -> dict:
    """Classify a user message as simple or complex.

    Returns:
        dict with keys: classification ("simple"|"complex"),
        confidence (0-1), reason (str)
    """
    if router_cfg is None:
        router_cfg = _load_router_config()

    if not router_cfg.get("enabled", False):
        return {"classification": "simple", "confidence": 0.0,
                "reason": "router disabled"}

    rules_cfg = router_cfg.get("rules", {})
    threshold = float(rules_cfg.get("threshold", 0.5))

    # Check always-simple patterns first
    always_simple = list(rules_cfg.get("always_simple", []))
    if _match_patterns(message, always_simple):
        return {"classification": "simple", "confidence": 1.0,
                "reason": "always_simple pattern match"}

    # Check always-complex patterns
    always_complex = list(rules_cfg.get("always_complex", []))
    if _match_patterns(message, always_complex):
        return {"classification": "complex", "confidence": 1.0,
                "reason": "always_complex pattern match"}

    # Run LLM classifier
    classifier_cfg = router_cfg.get("classifier", {})
    task = classifier_cfg.get("task", _DEFAULT_CLASSIFIER_TASK)
    model_override = classifier_cfg.get("model", "")
    result = _classify_with_llm(message, task, model_override)

    # Apply threshold
    if result["classification"] == "complex" and result["confidence"] < threshold:
        result["classification"] = "simple"
        result["reason"] = f"below threshold ({result['confidence']:.2f} < {threshold})"

    return result


def _build_orchestrator_prompt(
    user_message: str,
    history: list[dict] | None = None,
    session_context: dict | None = None,
) -> str:
    """Build the prompt for the orchestrator subprocess."""
    parts = []

    if history:
        recent = history[-6:]  # last 3 exchanges
        if recent:
            parts.append("=== Previous conversation ===")
            for msg in recent:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                content = str(content)[:2000]
                parts.append(f"[{role}]: {content}")
            parts.append("")

    parts.append("=== Current user request ===")
    parts.append(user_message)

    if session_context:
        cwd = session_context.get("cwd", "")
        if cwd:
            parts.insert(0, f"Working directory: {cwd}")

    return "\n".join(parts)


def _resolve_hermes_bin() -> str:
    """Resolve the hermes binary path."""
    hermes_bin = os.environ.get("HERMES_BIN", "")
    if hermes_bin:
        return hermes_bin
    for candidate in [
        "/usr/local/bin/hermes",
        os.path.expanduser("~/.local/bin/hermes"),
        "/usr/bin/hermes",
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return "hermes"


_NOISE_LINE_RE = re.compile(
    r"^(Tokens|Cost|Model|Session|⏺|●|○|────|╭─|╰─|├─|│)"
)


def _is_noise_line(cleaned_line: str) -> bool:
    """Return True if *cleaned_line* is a noise line from hermes output."""
    stripped = cleaned_line.strip()
    if not stripped:
        return True
    return bool(_NOISE_LINE_RE.match(stripped))


def _extract_hermes_response(output: str) -> str:
    """Extract the assistant's response from hermes chat output."""
    lines = output.split("\n")
    cleaned = []
    for line in lines:
        if _is_noise_line(line):
            continue
        cleaned.append(line)
    result = "\n".join(cleaned).strip()
    return result or output


def route_to_orchestrator(
    user_message: str,
    history: list[dict] | None = None,
    session_context: dict | None = None,
    router_cfg: dict | None = None,
    stream_callback: Callable[[str], Any] | None = None,
) -> str:
    """Route a complex task to the orchestrator profile.

    Spawns ``hermes -p <profile> chat -q "<prompt>"`` and returns the full
    response synchronously.

    When ``stream_callback`` is provided, the orchestrator's stdout is read
    line-by-line and each line (after filtering noise) is passed to the
    callback so the user sees progressive output instead of waiting silently
    for the entire subprocess to finish.  The callback receives one cleaned
    line at a time.  Errors and noise lines are skipped.
    """
    if router_cfg is None:
        router_cfg = _load_router_config()

    orch_cfg = router_cfg.get("orchestrator", {})
    profile = orch_cfg.get("profile", _DEFAULT_ORCHESTRATOR_PROFILE)
    timeout_val = int(orch_cfg.get("timeout", _DEFAULT_ORCHESTRATOR_TIMEOUT))
    pass_history = bool(orch_cfg.get("pass_history", True))

    if pass_history:
        prompt = _build_orchestrator_prompt(user_message, history, session_context)
    else:
        prompt = user_message

    hermes_bin = _resolve_hermes_bin()

    try:
        logger.info(
            "Router plugin: routing to orchestrator (profile=%s, timeout=%ds, prompt_len=%d)%s",
            profile, timeout_val, len(prompt),
            " [streaming]" if stream_callback else "",
        )

        cmd = [hermes_bin, "-p", profile, "chat", "-q", prompt, "-Q"]

        env = os.environ.copy()
        env.pop("HERMES_PROFILE", None)

        if stream_callback:
            # ── Streaming mode: Popen + line-by-line iteration ─────────
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            all_lines: list[str] = []
            try:
                assert proc.stdout is not None
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    all_lines.append(line)
                    cleaned = line.rstrip("\n")
                    if _is_noise_line(cleaned):
                        continue
                    try:
                        stream_callback(cleaned + "\n")
                    except Exception:
                        pass  # stream callback failures are non-fatal
            finally:
                proc.wait(timeout=timeout_val)

            if proc.returncode != 0:
                stderr_text = proc.stderr.read()[:500] if proc.stderr else ""
                error_msg = (
                    f"Orchestrator exited with code {proc.returncode}\n"
                    f"stderr: {stderr_text}"
                )
                logger.error("Router plugin: orchestrator failed: %s", error_msg)
                return f"[Orchestrator error] {error_msg}"

            raw_output = "".join(all_lines).strip()
            response = _extract_hermes_response(raw_output)
            return response or raw_output

        else:
            # ── Batch mode (original behaviour) ────────────────────────
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_val,
                env=env,
            )

            if result.returncode != 0:
                error_msg = (
                    f"Orchestrator exited with code {result.returncode}\n"
                    f"stderr: {result.stderr[:500]}"
                )
                logger.error("Router plugin: orchestrator failed: %s", error_msg)
                return f"[Orchestrator error] {error_msg}"

            raw_output = result.stdout.strip()
            response = _extract_hermes_response(raw_output)
            return response or raw_output

    except FileNotFoundError:
        logger.error("Router plugin: hermes binary not found at %s", hermes_bin)
        return "[Orchestrator error] Hermes binary not found"
    except subprocess.TimeoutExpired:
        logger.error("Router plugin: orchestrator timed out after %ds", timeout_val)
        return f"[Orchestrator error] Timed out after {timeout_val}s"
    except Exception as exc:
        logger.exception("Router plugin: orchestrator routing failed")
        return f"[Orchestrator error] {exc}"


# ── Session-level classification cache ────────────────────────────────
_classification_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 300  # seconds


def _get_cached_classification(session_key: str) -> dict | None:
    """Get a cached classification for a session, if still valid."""
    if session_key in _classification_cache:
        timestamp, result = _classification_cache[session_key]
        if time.time() - timestamp < _CACHE_TTL:
            return result
        del _classification_cache[session_key]
    return None


def _cache_classification(session_key: str, result: dict) -> None:
    """Cache a classification result for a session."""
    _classification_cache[session_key] = (time.time(), result)


# ── Hook callback ─────────────────────────────────────────────────────

def _on_pre_agent_dispatch(**kwargs: Any) -> dict | None:
    """pre_agent_dispatch hook: classify and optionally route to orchestrator.

    Called before the agent processes a user message. If the message is
    classified as complex and the router is enabled, this function calls
    the orchestrator subprocess and returns a ``route`` action with the
    result, bypassing the normal agent dispatch.
    """
    message = kwargs.get("message", "")
    if not message or not isinstance(message, str):
        return None

    session_key = kwargs.get("session_key", "") or ""
    history = kwargs.get("history") or None
    stream_callback = kwargs.get("stream_callback") or None

    # Check config
    router_cfg = _load_router_config()
    if not router_cfg.get("enabled", False):
        return None

    # Check cache
    if session_key:
        cached = _get_cached_classification(session_key)
        if cached is not None:
            if cached.get("classification") == "complex":
                logger.info(
                    "Router plugin: cached complex → routing to orchestrator"
                )
            else:
                return None
        else:
            # Classify
            result = classify(message, router_cfg)
            if session_key:
                _cache_classification(session_key, result)

            if result.get("classification") != "complex":
                return None

            logger.info(
                "Router plugin: complex task (confidence=%.2f, reason=%s) → orchestrator",
                result.get("confidence", 0),
                result.get("reason", ""),
            )

    # Route to orchestrator
    source = kwargs.get("source") or None
    session_context: dict | None = None
    if source is not None:
        cwd = getattr(source, "cwd", None) or ""
        if not cwd:
            # Try getting session cwd from gateway
            gateway = kwargs.get("gateway")
            if gateway is not None and session_key:
                session = getattr(gateway, "session_store", None)
                if session is not None:
                    sess_data = getattr(session, "get", lambda _: {})(session_key)
                    cwd = sess_data.get("cwd", "") if isinstance(sess_data, dict) else ""
        if cwd:
            session_context = {"cwd": cwd}

    orch_response = route_to_orchestrator(
        user_message=message,
        history=history,
        session_context=session_context,
        router_cfg=router_cfg,
        stream_callback=stream_callback,
    )

    return {"action": "route", "result": orch_response,
            "streamed": bool(stream_callback)}


# ── Plugin entry point ────────────────────────────────────────────────

def register(ctx: Any) -> None:
    """Register the router plugin hooks."""
    ctx.register_hook("pre_agent_dispatch", _on_pre_agent_dispatch)
    logger.info("Router plugin registered: pre_agent_dispatch hook")
