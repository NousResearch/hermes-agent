"""Local-server probe helpers for ``agent.model_metadata`` (endpoint URL shaping and server-type verdicts)."""
import re
from typing import Any


def _localhost_to_ipv4(url: str) -> str:
    """``localhost`` HOST -> ``127.0.0.1`` (Windows dual-stack resolves ::1 first and pays a ~2s IPv6
    connect timeout on IPv4-only servers). Scheme-anchored so ``?upstream=http://localhost`` is untouched."""
    if not url or not isinstance(url, str):
        return url  # non-string values (test doubles, lazy config) pass through
    return re.sub(r"^(https?://)localhost(?=[:/]|$)", r"\g<1>127.0.0.1", url, count=1)


def tags_server_verdict(response: Any) -> "str | bool":
    """Verdict for the Ollama-compat ``/api/tags`` route, which OpenAI-compatible servers also serve.

    Ollama owns ``/api/tags``, so a payload carrying ``models`` is Ollama — unless the server
    self-labels otherwise. SGLang answers this route for Ollama client compatibility and tags every
    entry with ``details.format == "sglang"`` (digest ``sha256:sglang…``). Returning the owner from the
    check rather than adding another leg on the same path keeps the probe to one request per endpoint.
    """
    payload = response.json()
    if "models" not in payload:
        return False  # unchanged from the original ollama-leg predicate
    models = payload.get("models") if isinstance(payload, dict) else None
    for entry in models if isinstance(models, list) else []:
        if not isinstance(entry, dict):
            continue
        details = entry.get("details")
        fmt = details.get("format") if isinstance(details, dict) else None
        if isinstance(fmt, str) and fmt.strip().lower() == "sglang":
            return "sglang"
        if str(entry.get("digest") or "").strip().lower().startswith("sha256:sglang"):
            return "sglang"
    return "ollama"
