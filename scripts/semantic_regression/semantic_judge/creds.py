"""Shared helpers: resolve working ollama-cloud credentials and call the judge."""

import json
import os

from .judge import LLMJudge

PROFILE_AUTH = os.path.expanduser(
    "~/.hermes/profiles/quan-code/auth.json"
)


def _load_creds(path=None):
    path = path or PROFILE_AUTH
    with open(path) as f:
        data = json.load(f)
    pool = data.get("credential_pool", {}) or {}
    return pool


def resolve_ollama_creds(path=None):
    """Return (base_url, api_key) for a usable ollama-cloud credential, or None.

    Resolution order (so the release gate works both on the VPS and in CI):
      1. Explicit env vars OLLAMA_BASE_URL / OLLAMA_API_KEY (CI injects these
         via GitHub Actions secrets).
      2. A healthy (last_status == "ok") ollama-cloud credential from the
         profile auth.json pool.
      3. Any ollama-cloud credential in the pool.
    """
    env_base = os.environ.get("OLLAMA_BASE_URL")
    env_key = os.environ.get("OLLAMA_API_KEY")
    if env_key:
        return env_base or "https://ollama.com/v1", env_key
    if path is None and not os.path.exists(PROFILE_AUTH):
        return None
    try:
        pool = _load_creds(path)
    except (OSError, json.JSONDecodeError):
        return None
    for cred in pool.get("ollama-cloud", []):
        if cred.get("last_status") == "ok" and cred.get("access_token"):
            base = cred.get("base_url") or "https://ollama.com/v1"
            return base, cred["access_token"]
    for cred in pool.get("ollama-cloud", []):
        if cred.get("access_token"):
            return cred.get("base_url") or "https://ollama.com/v1", cred["access_token"]
    return None


def default_judge(model="deepseek-v4-flash"):
    """Build an LLMJudge from env vars or the profile's ollama-cloud credential."""
    creds = resolve_ollama_creds()
    if not creds:
        raise RuntimeError(
            "No ollama-cloud credentials available (set OLLAMA_API_KEY or "
            "populate auth.json)"
        )
    base, key = creds
    return LLMJudge(base_url=base, api_key=key, model=model)

