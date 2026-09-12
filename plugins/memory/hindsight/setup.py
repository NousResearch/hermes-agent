"""`hermes memory setup` wizard for the Hindsight provider (``post_setup``)."""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

from agent.secret_scope import get_secret
from hermes_cli.secret_prompt import masked_secret_prompt

from . import templates as _hs_templates
from .embedded import _embedded_profile_env_path, _load_simple_env, _materialize_embedded_profile_env
from .settings import (
    _DEFAULT_API_URL, _DEFAULT_IDLE_TIMEOUT, _DEFAULT_LOCAL_URL, _DEFAULT_TIMEOUT, _MIN_CLIENT_VERSION,
    _PROVIDER_DEFAULT_MODELS,
)

_MODE_VALUES = ["cloud", "local_embedded", "local_external"]
_MODE_ITEMS = [
    ("Cloud", "Hindsight Cloud API (lightweight, just needs an API key)"),
    ("Local Embedded", "Run Hindsight locally (downloads ~200MB, needs LLM key)"),
    ("Local External", "Connect to an existing Hindsight instance"),
]


def _secret_prompt(label: str) -> str:
    """Masked prompt on a TTY; plain readline when stdin is piped."""
    sys.stdout.write(label)
    sys.stdout.flush()
    return masked_secret_prompt("") if sys.stdin.isatty() else sys.stdin.readline().strip()


def _select(title: str, items: list, values: list, current) -> str | None:
    """Curses pick from *values*, defaulting to *current*; None when cancelled."""
    from hermes_cli.memory_setup import _CANCELLED, _curses_select, _print_cancelled_setup

    default = values.index(current) if current in values else 0
    idx = _curses_select(title, items, default=default, cancel_returns=_CANCELLED)
    if idx == _CANCELLED:
        _print_cancelled_setup()
        return None
    return values[idx]


def _write_env(env_path: Path, env_writes: dict) -> None:
    """Update keys in place (BOM-tolerant read: a Notepad BOM would glue U+FEFF onto
    the first key and duplicate the line), append the rest."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    existing = env_path.read_text(encoding="utf-8-sig").splitlines() if env_path.exists() else []
    updated = set()
    new_lines = []
    for line in existing:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
        new_lines.append(f"{key}={env_writes[key]}" if key in env_writes else line)
        updated.add(key)
    new_lines.extend(f"{k}={v}" for k, v in env_writes.items() if k not in updated)
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _prompt_embedded_llm(llm_provider: str, provider_config: dict, env_writes: dict, hermes_env: Path) -> None:
    """local_embedded wizard step: endpoint (openai_compatible only), model, LLM key."""
    if llm_provider == "openai_compatible":
        existing_base_url = provider_config.get("llm_base_url", "")
        prompt = "  LLM endpoint URL (e.g. http://192.168.1.10:8080/v1)" + (f" [{existing_base_url}]" if existing_base_url else "")
        if val := input(prompt + ": ").strip():
            provider_config["llm_base_url"] = val
    elif llm_provider == "openrouter":
        provider_config["llm_base_url"] = "https://openrouter.ai/api/v1"
    current_model = provider_config.get("llm_model") or _PROVIDER_DEFAULT_MODELS.get(llm_provider, "gpt-4o-mini")
    val = input(f"  LLM model [{current_model}]: ").strip()
    provider_config["llm_model"] = val or current_model
    llm_key = _secret_prompt("  LLM API key: ")
    env_writes["HINDSIGHT_LLM_API_KEY"] = llm_key or _load_simple_env(hermes_env).get("HINDSIGHT_LLM_API_KEY", "")


def run_setup(provider, hermes_home: str, config: dict) -> None:
    """Interactive wizard — installs only the deps the selected mode needs."""
    from hermes_cli.config import save_config

    from . import _load_config

    print("\n  Configuring Hindsight memory:\n")

    existing_config = provider._config if isinstance(provider._config, dict) else _load_config()
    if not isinstance(existing_config, dict):
        existing_config = {}

    mode = _select("  Select mode", _MODE_ITEMS, _MODE_VALUES, existing_config.get("mode"))
    if mode is None:
        return
    provider_config: dict = dict(existing_config, mode=mode)
    env_writes: dict = {}
    hermes_env = Path(hermes_home) / ".env"

    llm_provider = ""
    if mode == "local_embedded":
        llm_items = [(p, f"default model: {m}") for p, m in _PROVIDER_DEFAULT_MODELS.items()]
        llm_provider = _select("  Select LLM provider", llm_items, list(_PROVIDER_DEFAULT_MODELS),
                               provider_config.get("llm_provider"))
        if llm_provider is None:
            return
        provider_config["llm_provider"] = llm_provider

    print("\n  Checking dependencies...")
    # Environment-aware install: sealed hosted venvs redirect to the durable data volume.
    from tools.lazy_deps import install_specs

    deps = ["hindsight-all"] if mode == "local_embedded" else [f"hindsight-client>={_MIN_CLIENT_VERSION}"]
    outcome = install_specs(deps, timeout=120)
    if outcome.ok:
        print("  ✓ Dependencies up to date")
    elif outcome.blocked:
        print(f"  ⚠ Cannot install dependencies: {outcome.reason}")
    else:
        print(f"  ⚠ Install failed:\n{(outcome.stderr or '').strip()}")
        print(f"  Run manually: uv pip install --python {sys.executable} {' '.join(deps)}")

    if mode == "cloud":
        print("\n  Get your API key at https://ui.hindsight.vectorize.io\n")
        existing_key = get_secret("HINDSIGHT_API_KEY", "") or ""
        masked = f"...{existing_key[-4:]}" if len(existing_key) > 4 else "set"
        api_key = _secret_prompt(f"  API key (current: {masked}, blank to keep): " if existing_key else "  API key: ")
        if api_key:
            env_writes["HINDSIGHT_API_KEY"] = api_key
        if val := input(f"  API URL [{_DEFAULT_API_URL}]: ").strip():
            provider_config["api_url"] = val
    elif mode == "local_external":
        val = input(f"  Hindsight API URL [{_DEFAULT_LOCAL_URL}]: ").strip()
        provider_config["api_url"] = val or _DEFAULT_LOCAL_URL
        if api_key := _secret_prompt("  API key (optional, blank to skip): "):
            env_writes["HINDSIGHT_API_KEY"] = api_key
    else:
        _prompt_embedded_llm(llm_provider, provider_config, env_writes, hermes_env)

    provider_config.setdefault("bank_id", "hermes")
    provider_config.setdefault("recall_budget", "mid")
    # Preserve explicit 0 timeouts instead of treating them as blank.
    timeouts = [("timeout", "HINDSIGHT_TIMEOUT", _DEFAULT_TIMEOUT)]
    if mode == "local_embedded":
        timeouts.append(("idle_timeout", "HINDSIGHT_IDLE_TIMEOUT", _DEFAULT_IDLE_TIMEOUT))
    for key, env_key, default in timeouts:
        value = provider_config.get(key)
        provider_config[key] = value = default if value is None else value
        env_writes[env_key] = str(value)
    config["memory"]["provider"] = "hindsight"
    save_config(config)
    provider.save_config(provider_config, hermes_home)
    if env_writes:
        _write_env(hermes_env, env_writes)

    # Starter template (best-effort) only where the API is reachable now
    # (local_embedded's daemon isn't up).
    if _hs_templates.supported_for_mode(mode):
        from hermes_cli.memory_setup import _CANCELLED, _curses_select

        default_url = _DEFAULT_LOCAL_URL if mode == "local_external" else _DEFAULT_API_URL
        _hs_templates.run_template_step(
            api_url=provider_config.get("api_url") or default_url,
            bank_id=provider_config.get("bank_id", "hermes"),
            api_key=env_writes.get("HINDSIGHT_API_KEY") or os.environ.get("HINDSIGHT_API_KEY", "") or None,
            select=_curses_select, cancelled=_CANCELLED,
        )

    if mode == "local_embedded":
        materialized_config = dict(provider_config)
        with contextlib.suppress(Exception):
            materialized_config = json.loads(
                (Path(hermes_home) / "hindsight" / "config.json").read_text(encoding="utf-8")
            )
        llm_api_key = (
            env_writes.get("HINDSIGHT_LLM_API_KEY", "")
            or _load_simple_env(hermes_env).get("HINDSIGHT_LLM_API_KEY", "")
            or _load_simple_env(_embedded_profile_env_path(materialized_config)).get("HINDSIGHT_API_LLM_API_KEY", "")
        )
        _materialize_embedded_profile_env(materialized_config, llm_api_key=llm_api_key or None)

    print(f"\n  ✓ Hindsight memory configured ({mode} mode)")
    if env_writes:
        print("  API keys saved to .env")
    print("\n  Start a new session to activate.\n")
