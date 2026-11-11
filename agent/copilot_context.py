"""Copilot context-tier metadata, independent of a model's total context window."""
from typing import Any


def context_cache_url(agent) -> str:
    """Cache key for learned context limits, scoped per channel tier: the long tier's
    learned cap must not leak onto the default tier (and vice versa)."""
    url = getattr(agent, "base_url", "") or ""
    extra = (getattr(agent, "request_overrides", {}) or {}).get("extra_body") or {}
    if getattr(agent, "provider", "") == "copilot-acp" and extra.get("copilot_context_tier") == "long_context":
        return url.rstrip("/") + "#copilot-context=long_context"
    return url


def context_tiers(model: dict[str, Any]) -> dict[str, int]:
    """Only advertise tier input budgets explicitly published by the runtime."""
    prices = (model.get("billing") or {}).get("tokenPrices") or {}
    limits = (model.get("capabilities") or {}).get("limits") or {}
    long_prices = prices.get("longContext") or {}
    candidates = {
        "default": prices.get("maxPromptTokens", prices.get("contextMax")),
        # capabilities.limits.max_prompt_tokens carries the LONG-tier input budget in the
        # live catalog (936k/922k/872k observed vs billing's 200k-272k default), so it only
        # backfills long_context — never the default tier (an oversized default would first-
        # hit the budget refusal and only self-correct after learning).
        "long_context": long_prices.get("maxPromptTokens",
            long_prices.get("contextMax", limits.get("max_prompt_tokens"))),
    }
    return {tier: budget for tier, budget in candidates.items() if type(budget) is int and budget > 0}


# --- Session-tier request overrides (turn assembly) ---

# Process cache: (command, model) -> tiers dict. A successful catalog read lives 1h; a failed
# one 5min (negative cache) so a down CLI cannot add a spawn attempt to every turn.
_TIER_BUDGET_CACHE: dict[tuple[str, str], tuple[float, dict[str, int]]] = {}
_TIER_TTL = 3600.0
_TIER_NEG_TTL = 300.0


def _copilot_creds() -> dict:
    """Existing copilot-acp credentials from the shared resolver; {} when unavailable."""
    try:
        from hermes_cli.auth import resolve_external_process_provider_credentials

        return resolve_external_process_provider_credentials("copilot-acp") or {}
    except Exception:  # health: allow BLE001 -- an unreadable credential store names no channel; a tier-only request still rides the launch args
        return {}


def _cached_context_tiers(model: str, *, creds: dict | None = None, timeout: float = 8.0) -> dict[str, int]:
    import time as _time

    creds = _copilot_creds() if creds is None else creds
    command, token = str(creds.get("command") or ""), str(creds.get("api_key") or "")
    if not command or not token:
        return {}
    key = (command, str(model))
    now = _time.monotonic()
    hit = _TIER_BUDGET_CACHE.get(key)
    if hit is not None and now - hit[0] < (_TIER_TTL if hit[1] else _TIER_NEG_TTL):
        return hit[1]
    try:
        tiers = next((context_tiers(row) for row in fetch_context_catalog(
            command=command, token=token, timeout=timeout)
            if str(row.get("id") or "") == str(model)), {})
    except Exception:  # health: allow BLE001 -- a failed catalog read caches as empty (negative TTL); the tier itself still rides the launch args
        tiers = {}
    _TIER_BUDGET_CACHE[key] = (now, tiers)
    return tiers


def tier_extra_body(tier: str | None, model: str, *, creds: dict | None = None, timeout: float = 8.0) -> dict:
    """``extra_body`` fragment for the copilot-acp channel: the tier always, the advertised
    input budget when the cached catalog has one. Tier-only is still correct — the tier rides
    the subprocess launch args, and without a budget the compressor falls back to the default
    window plus limit learning."""
    if tier not in ("default", "long_context"):
        return {}
    extra: dict[str, Any] = {"copilot_context_tier": tier}
    budget = _cached_context_tiers(model, creds=creds, timeout=timeout).get(tier)
    if type(budget) is int and budget > 0:
        extra["copilot_context_budget"] = budget
    return extra


def _read_sdk_frames(stream, inbox) -> None:
    import json
    try:
        while header := stream.readline():
            headers = {}
            while header.strip():
                key, value = header.decode("ascii").split(":", 1)
                headers[key.lower()] = value.strip()
                header = stream.readline()
            size = int(headers["content-length"])
            if not 0 < size <= 8_000_000:
                raise ValueError("Invalid metadata frame size")
            inbox.put(json.loads(stream.read(size)))
    except Exception as exc:  # health: allow BLE001 -- the reader's whole job is to deliver ANY death to the main loop via the queue
        inbox.put(exc)
    finally:
        inbox.put(EOFError("Copilot metadata process ended"))


def fetch_context_catalog(*, command: str, token: str, prefix_args=(), timeout: float = 15) -> list[dict]:
    """Read models.list using the installed runtime's SDK protocol; never create a session.

    ACP exposes model names but not tier prices/budgets. The headless metadata door uses
    Content-Length frames (not ACP newline frames); no Node SDK installation is required.
    """
    import json
    import queue
    import subprocess
    import tempfile
    import threading
    import time
    from pathlib import Path
    from agent.acp_client import _build_subprocess_env
    from agent.memory_provider import spawn_context_thread
    from hermes_cli._subprocess_compat import windows_hide_flags

    if not token:
        raise ValueError("Copilot metadata requires an existing credential")
    inbox = queue.Queue()
    deadline = time.monotonic() + timeout
    with tempfile.TemporaryDirectory(prefix="hermes-copilot-metadata-") as home:
        Path(home, "settings.json").write_text(json.dumps({"disableAllHooks": True, "memory": False}), encoding="utf-8")
        env = _build_subprocess_env()
        env.update(COPILOT_HOME=home, COPILOT_SDK_AUTH_TOKEN=token)
        argv = [command, *prefix_args, "--headless", "--stdio", "--no-auto-login",
                "--auth-token-env", "COPILOT_SDK_AUTH_TOKEN"]
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, cwd=home, env=env, creationflags=windows_hide_flags())
        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("Copilot metadata process did not expose stdin/stdout pipes.")
        reader = spawn_context_thread(_read_sdk_frames, args=(proc.stdout, inbox), name="copilot-metadata-reader")
        reader.start()
        try:
            result = {}
            requests = (("connect", {"supportedTaskKinds": ["agent", "client", "shell"]}), ("models.list", {}))
            for rid, (method, params) in enumerate(requests, 1):
                body = json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}).encode()
                proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
                proc.stdin.flush()
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Copilot model metadata timed out")
                    msg = inbox.get(timeout=remaining)
                    if isinstance(msg, Exception):
                        raise RuntimeError("Copilot model metadata stream ended") from msg
                    if msg.get("id") != rid:
                        continue
                    if "error" in msg:
                        raise RuntimeError("Copilot model metadata request failed; check existing CLI authentication")
                    result = msg.get("result") or {}
                    break
            return [row for row in result.get("models", []) if isinstance(row, dict)]
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5)
            reader.join(timeout=1)
            proc.stdout.close()
