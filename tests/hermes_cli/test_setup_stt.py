"""Tests for the speech-to-text section of the setup wizard."""
import os

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated ~/.hermes for setup writes."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return home


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in (
        "ELEVENLABS_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY",
        "XAI_API_KEY", "VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    for n in range(2, 11):
        monkeypatch.delenv(f"ELEVENLABS_API_KEY_{n}", raising=False)


def _scripted_choice(label_to_index):
    """Return a fake prompt_choice that picks the index by question label."""
    def _impl(question, choices, default=0, description=None):
        for label, idx in label_to_index.items():
            if label in question:
                return idx
        return default
    return _impl


def _scripted_inputs(answers):
    """Return a fake prompt() that pops from a list of answers in order."""
    queue = list(answers)

    def _impl(question, default=None, password=False):
        assert queue, f"unexpected prompt: {question}"
        return queue.pop(0)
    return _impl


def _scripted_yes_no(answers):
    queue = list(answers)

    def _impl(question, default=True):
        assert queue, f"unexpected yes/no prompt: {question}"
        return queue.pop(0)
    return _impl


def test_select_local_does_not_prompt_for_keys(hermes_home, monkeypatch):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.setup import setup_stt

    cfg = load_config()
    cfg.setdefault("stt", {})["provider"] = "openai"
    save_config(cfg)

    # Index 0 is the local provider.
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice", _scripted_choice({"STT provider": 0}),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *a, **kw: pytest.fail("local provider must not prompt for a key"),
    )

    cfg = load_config()
    setup_stt(cfg)

    assert load_config()["stt"]["provider"] == "local"


def test_keep_current_does_nothing(hermes_home, monkeypatch):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.setup import setup_stt

    cfg = load_config()
    cfg.setdefault("stt", {})["provider"] = "mistral"
    save_config(cfg)

    # Pick the last option (Keep current).
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        lambda q, c, d=0, description=None: len(c) - 1,
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *a, **kw: pytest.fail("keep-current must not prompt"),
    )

    setup_stt(cfg)
    assert load_config()["stt"]["provider"] == "mistral"


def test_elevenlabs_first_time_saves_primary_key(hermes_home, monkeypatch):
    from hermes_cli.config import load_config
    from hermes_cli.setup import setup_stt

    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        _scripted_choice({"STT provider": 5}),  # 5 = elevenlabs
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt", _scripted_inputs(["sk-primary"]),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_yes_no", _scripted_yes_no([False]),
    )

    cfg = load_config()
    setup_stt(cfg)

    assert os.environ.get("ELEVENLABS_API_KEY") == "sk-primary"
    assert load_config()["stt"]["provider"] == "elevenlabs"


def test_elevenlabs_reuses_existing_key_from_tts(hermes_home, monkeypatch):
    """Don't ask the user again if ELEVENLABS_API_KEY is already set."""
    from hermes_cli.config import load_config
    from hermes_cli.setup import setup_stt

    # User already configured ElevenLabs for TTS.
    monkeypatch.setenv("ELEVENLABS_API_KEY", "from-tts")

    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        _scripted_choice({"STT provider": 5}),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *a, **kw: pytest.fail("must not re-prompt for primary key"),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_yes_no", _scripted_yes_no([False]),
    )

    setup_stt(load_config())
    assert os.environ.get("ELEVENLABS_API_KEY") == "from-tts"
    assert load_config()["stt"]["provider"] == "elevenlabs"


def test_elevenlabs_add_another_key_loop(hermes_home, monkeypatch):
    """User can append fallback keys via the 'add another?' loop."""
    from hermes_cli.config import load_config
    from hermes_cli.setup import setup_stt

    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        _scripted_choice({"STT provider": 5}),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        _scripted_inputs(["sk-1", "sk-2", "sk-3"]),
    )
    # yes, yes, no — ends the loop after two fallbacks.
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_yes_no", _scripted_yes_no([True, True, False]),
    )

    setup_stt(load_config())

    assert os.environ.get("ELEVENLABS_API_KEY") == "sk-1"
    assert os.environ.get("ELEVENLABS_API_KEY_2") == "sk-2"
    assert os.environ.get("ELEVENLABS_API_KEY_3") == "sk-3"
    assert os.environ.get("ELEVENLABS_API_KEY_4") is None


def test_elevenlabs_resumes_after_existing_fallbacks(hermes_home, monkeypatch):
    """If _2 already exists, the next prompt offers _3, not _2 again."""
    from hermes_cli.config import load_config
    from hermes_cli.setup import setup_stt

    monkeypatch.setenv("ELEVENLABS_API_KEY", "primary")
    monkeypatch.setenv("ELEVENLABS_API_KEY_2", "second")

    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        _scripted_choice({"STT provider": 5}),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt", _scripted_inputs(["third"]),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt_yes_no", _scripted_yes_no([True, False]),
    )

    setup_stt(load_config())

    # _2 stays where it was, new key lands at _3.
    assert os.environ.get("ELEVENLABS_API_KEY_2") == "second"
    assert os.environ.get("ELEVENLABS_API_KEY_3") == "third"


def test_elevenlabs_empty_primary_aborts(hermes_home, monkeypatch):
    """If the user gives an empty key, leave the provider unchanged."""
    from hermes_cli.config import load_config, save_config
    from hermes_cli.setup import setup_stt

    cfg = load_config()
    cfg.setdefault("stt", {})["provider"] = "local"
    save_config(cfg)

    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        _scripted_choice({"STT provider": 5}),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt", _scripted_inputs([""]),
    )

    setup_stt(load_config())

    assert load_config()["stt"]["provider"] == "local"
    assert os.environ.get("ELEVENLABS_API_KEY") in (None, "")


def test_setup_sections_includes_stt():
    from hermes_cli.setup import SETUP_SECTIONS, setup_stt

    keys = [k for k, _, _ in SETUP_SECTIONS]
    assert "stt" in keys
    handler = dict([(k, h) for k, _, h in SETUP_SECTIONS])["stt"]
    assert handler is setup_stt


# ----------------------------------------------------------------------------
# Cloud-provider key-prompt branches.  Symmetric to the elevenlabs path but
# without the rotation loop — just confirms each provider writes the right
# env var when the user supplies a key, and aborts cleanly when they don't.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_idx,provider_key,env_var,prompted_label",
    [
        (1, "groq", "GROQ_API_KEY", "Groq API key"),
        (2, "openai", "VOICE_TOOLS_OPENAI_KEY", "OpenAI API key for STT"),
        (3, "mistral", "MISTRAL_API_KEY", "Mistral API key"),
        (4, "xai", "XAI_API_KEY", "xAI API key"),
    ],
)
def test_cloud_provider_first_time_saves_key(
    hermes_home, monkeypatch, provider_idx, provider_key, env_var, prompted_label,
):
    from hermes_cli.config import load_config
    from hermes_cli.setup import setup_stt

    seen = {}

    def fake_prompt(question, default=None, password=False):
        seen["question"] = question
        return "supplied-key"

    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        _scripted_choice({"STT provider": provider_idx}),
    )
    monkeypatch.setattr("hermes_cli.setup.prompt", fake_prompt)

    setup_stt(load_config())

    assert os.environ.get(env_var) == "supplied-key"
    assert load_config()["stt"]["provider"] == provider_key
    assert prompted_label in seen["question"]


@pytest.mark.parametrize(
    "provider_idx,provider_key",
    [(1, "groq"), (2, "openai"), (3, "mistral"), (4, "xai")],
)
def test_cloud_provider_empty_input_aborts(
    hermes_home, monkeypatch, provider_idx, provider_key,
):
    from hermes_cli.config import load_config, save_config
    from hermes_cli.setup import setup_stt

    cfg = load_config()
    cfg.setdefault("stt", {})["provider"] = "local"
    save_config(cfg)

    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice",
        _scripted_choice({"STT provider": provider_idx}),
    )
    monkeypatch.setattr("hermes_cli.setup.prompt", lambda *a, **kw: "")

    setup_stt(load_config())
    assert load_config()["stt"]["provider"] == "local"


def test_cloud_provider_reuses_existing_key(hermes_home, monkeypatch):
    """If a key is already in env (from a prior section or .env edit), don't
    re-prompt — just switch the provider."""
    from hermes_cli.config import load_config
    from hermes_cli.setup import setup_stt

    monkeypatch.setenv("GROQ_API_KEY", "already-set")

    monkeypatch.setattr(
        "hermes_cli.setup.prompt_choice", _scripted_choice({"STT provider": 1}),
    )
    monkeypatch.setattr(
        "hermes_cli.setup.prompt",
        lambda *a, **kw: pytest.fail("must not re-prompt for existing key"),
    )

    setup_stt(load_config())
    assert load_config()["stt"]["provider"] == "groq"
    assert os.environ.get("GROQ_API_KEY") == "already-set"
