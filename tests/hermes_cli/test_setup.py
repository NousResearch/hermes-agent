"""Tests for setup.py configuration flows."""
import sys
import os
import json
import types


from hermes_cli.config import load_config, save_config
from hermes_cli import setup as setup_mod
from hermes_cli.setup import setup_model_provider


def _maybe_keep_current_tts(question, choices):
    if question != "Select TTS provider:":
        return None
    assert choices[-1].startswith("Keep current (")
    return len(choices) - 1


def _clear_provider_env(monkeypatch):
    for key in (
        "NOUS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)


def _clear_vercel_env(monkeypatch):
    for key in (
        "TERMINAL_VERCEL_RUNTIME",
        "VERCEL_OIDC_TOKEN",
        "VERCEL_TOKEN",
        "VERCEL_PROJECT_ID",
        "VERCEL_TEAM_ID",
    ):
        monkeypatch.delenv(key, raising=False)


def _stub_tts(monkeypatch):
    """Stub out TTS prompts so setup_model_provider doesn't block."""
    monkeypatch.setattr("hermes_cli.setup.prompt_choice", lambda q, c, d=0: (
        _maybe_keep_current_tts(q, c) if _maybe_keep_current_tts(q, c) is not None
        else d
    ))
    monkeypatch.setattr("hermes_cli.setup.prompt_yes_no", lambda *a, **kw: False)


def _write_model_config(tmp_path, provider, base_url="", model_name="test-model"):
    """Simulate what a _model_flow_* function writes to disk."""
    cfg = load_config()
    m = cfg.get("model")
    if not isinstance(m, dict):
        m = {"default": m} if m else {}
        cfg["model"] = m
    m["provider"] = provider
    if base_url:
        m["base_url"] = base_url
    if model_name:
        m["default"] = model_name
    save_config(cfg)


def test_setup_delegates_to_select_provider_and_model(tmp_path, monkeypatch):
    """setup_model_provider calls select_provider_and_model and syncs config."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)
    _stub_tts(monkeypatch)

    config = load_config()

    def fake_select():
        _write_model_config(tmp_path, "custom", "http://localhost:11434/v1", "qwen3.5:32b")

    monkeypatch.setattr("hermes_cli.main.select_provider_and_model", fake_select)

    setup_model_provider(config)
    save_config(config)

    reloaded = load_config()
    assert isinstance(reloaded["model"], dict)
    assert reloaded["model"]["provider"] == "custom"
    assert reloaded["model"]["base_url"] == "http://localhost:11434/v1"
    assert reloaded["model"]["default"] == "qwen3.5:32b"






def test_select_provider_and_model_warns_if_named_custom_provider_disappears(
    tmp_path, monkeypatch, capsys
):
    """If a saved custom provider is deleted mid-selection, show a warning instead of silently doing nothing."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_provider_env(monkeypatch)

    cfg = load_config()
    cfg["custom_providers"] = [{"name": "Local", "base_url": "http://localhost:8080/v1"}]
    save_config(cfg)

    def fake_prompt_provider_choice(choices, default=0):
        current = load_config()
        current["custom_providers"] = []
        save_config(current)
        return next(i for i, label in enumerate(choices) if label.startswith("Local (localhost:8080/v1)"))

    monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda provider: None)
    monkeypatch.setattr("hermes_cli.main._prompt_provider_choice", fake_prompt_provider_choice)
    monkeypatch.setattr(
        "hermes_cli.main._model_flow_named_custom",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("named custom flow should not run")),
    )

    from hermes_cli.main import select_provider_and_model

    select_provider_and_model()

    out = capsys.readouterr().out
    assert "selected saved custom provider is no longer available" in out








def test_modal_setup_persists_direct_mode_when_user_chooses_their_own_account(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_cli.setup.managed_nous_tools_enabled", lambda: True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select terminal backend:":
            return 2
        if question == "Select how Modal execution should be billed:":
            return 1
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    prompt_values = iter(["token-id", "token-secret", ""])

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: next(prompt_values))
    monkeypatch.setattr("hermes_cli.setup._prompt_container_resources", lambda config: None)
    monkeypatch.setattr(
        "hermes_cli.setup.get_nous_subscription_features",
        lambda config: type("Features", (), {"nous_auth_present": True})(),
    )
    monkeypatch.setitem(
        sys.modules,
        "tools.managed_tool_gateway",
        types.SimpleNamespace(
            is_managed_tool_gateway_ready=lambda vendor: vendor == "modal",
            resolve_managed_tool_gateway=lambda vendor: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "swe_rex", object())

    from hermes_cli.setup import setup_terminal_backend

    setup_terminal_backend(config)

    assert config["terminal"]["backend"] == "modal"
    assert config["terminal"]["modal_mode"] == "direct"


# test_setup_slack_* moved to tests/gateway/test_slack_plugin_setup.py — the
# _setup_slack wizard migrated to the slack plugin's interactive_setup (#41112).


def test_vercel_setup_configures_access_token_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_vercel_env(monkeypatch)
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "old-oidc")
    monkeypatch.setitem(sys.modules, "vercel", types.ModuleType("vercel"))
    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select terminal backend:":
            return 5
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    prompt_values = iter(["python3.13", "yes", "2", "4096", "token", "project", "team"])

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *args, **kwargs: next(prompt_values))

    from hermes_cli.setup import setup_terminal_backend

    setup_terminal_backend(config)

    assert config["terminal"]["backend"] == "vercel_sandbox"
    assert config["terminal"]["vercel_runtime"] == "python3.13"
    assert config["terminal"]["container_disk"] == 51200
    assert os.environ["TERMINAL_VERCEL_RUNTIME"] == "python3.13"
    assert "VERCEL_OIDC_TOKEN" not in os.environ
    assert os.environ["VERCEL_TOKEN"] == "token"
    assert os.environ["VERCEL_PROJECT_ID"] == "project"
    assert os.environ["VERCEL_TEAM_ID"] == "team"


def test_vercel_setup_prefills_project_and_team_from_link_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _clear_vercel_env(monkeypatch)
    project_root = tmp_path / "project"
    nested = project_root / "app" / "src"
    nested.mkdir(parents=True)
    vercel_dir = project_root / ".vercel"
    vercel_dir.mkdir()
    (vercel_dir / "project.json").write_text(
        json.dumps({"projectId": "linked-project", "orgId": "linked-team"}),
        encoding="utf-8",
    )
    monkeypatch.chdir(nested)
    monkeypatch.setitem(sys.modules, "vercel", types.ModuleType("vercel"))
    config = load_config()
    config["terminal"]["container_disk"] = 999

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select terminal backend:":
            return 5
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    prompt_values = iter(["node24", "no", "1", "5120", "token", "", ""])
    defaults = {}

    def fake_prompt(message, default="", **kwargs):
        defaults[message] = default
        value = next(prompt_values)
        return value or default

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)

    from hermes_cli.setup import setup_terminal_backend

    setup_terminal_backend(config)

    assert config["terminal"]["backend"] == "vercel_sandbox"
    assert config["terminal"]["container_persistent"] is False
    assert config["terminal"]["container_disk"] == 51200
    assert "VERCEL_OIDC_TOKEN" not in os.environ
    assert os.environ["VERCEL_TOKEN"] == "token"
    assert os.environ["VERCEL_PROJECT_ID"] == "linked-project"
    assert os.environ["VERCEL_TEAM_ID"] == "linked-team"
    assert defaults["    Vercel project ID"] == "linked-project"
    assert defaults["    Vercel team ID"] == "linked-team"


def test_blaxel_setup_never_prompts_for_ignored_cpu_or_container_disk(tmp_path, monkeypatch):
    """Blaxel must not reuse the shared container-resource prompt.

    The review on #20809 flagged that routing Blaxel through
    ``_prompt_container_resources`` asks for CPU and container disk, which the
    backend ignores. This pins the Blaxel-specific prompt: the questions asked
    are image, memory, persistence, volume size, and TTL — never CPU.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in ("BL_WORKSPACE", "BL_API_KEY", "TERMINAL_BLAXEL_IMAGE", "TERMINAL_BLAXEL_TTL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(sys.modules, "blaxel", types.ModuleType("blaxel"))
    config = load_config()

    def fake_prompt_choice(question, choices, default=0):
        if question == "Select terminal backend:":
            return 6
        raise AssertionError(f"Unexpected prompt_choice call: {question}")

    asked: list[str] = []
    answers = {
        "Sandbox image": "blaxel/py-app:latest",
        "Memory (MB)": "8192",
        "Keep a persistent volume per task? (yes/no)": "yes",
        "Persistent volume size (MB)": "20480",
        "Sandbox TTL (e.g. 24h)": "12h",
    }

    def fake_prompt(question, default=None, *args, **kwargs):
        label = question.strip()
        asked.append(label)
        return answers.get(label, default if default is not None else "")

    monkeypatch.setattr("hermes_cli.setup.prompt_choice", fake_prompt_choice)
    monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)

    from hermes_cli.setup import setup_terminal_backend

    setup_terminal_backend(config)

    assert config["terminal"]["backend"] == "blaxel"
    assert config["terminal"]["blaxel_image"] == "blaxel/py-app:latest"
    assert config["terminal"]["container_memory"] == 8192
    assert config["terminal"]["container_persistent"] is True
    assert config["terminal"]["container_disk"] == 20480
    assert config["terminal"]["blaxel_ttl"] == "12h"

    joined = " | ".join(asked).lower()
    assert "cpu" not in joined, f"Blaxel setup must not ask for CPU; asked: {asked}"
