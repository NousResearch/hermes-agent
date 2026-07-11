"""Tests for the Matrix plugin's interactive_setup wizard."""

import hermes_cli.config as config_mod
import hermes_cli.cli_output as cli_output_mod
from plugins.platforms.matrix.adapter import interactive_setup


def _patch_setup_io(monkeypatch, prompts, yes_no_answers, saved):
    prompt_iter = iter(prompts)
    yes_no_iter = iter(yes_no_answers)
    monkeypatch.setattr(config_mod, "get_env_value", lambda key: saved.get(key, ""))
    monkeypatch.setattr(config_mod, "save_env_value", lambda k, v: saved.update({k: v}))
    monkeypatch.setattr(cli_output_mod, "prompt", lambda *_a, **_kw: next(prompt_iter))
    monkeypatch.setattr(
        cli_output_mod,
        "prompt_yes_no",
        lambda *_a, **_kw: next(yes_no_iter),
    )
    for name in ("print_header", "print_info", "print_success", "print_warning"):
        monkeypatch.setattr(cli_output_mod, name, lambda *_a, **_kw: None)


def test_interactive_setup_non_e2ee_only_installs_base_feature(monkeypatch):
    saved = {}
    calls = []
    _patch_setup_io(
        monkeypatch,
        [
            "https://matrix.example.org",
            "syt_token",
            "@bot:example.org",
            "",
            "",
        ],
        [False],
        saved,
    )
    monkeypatch.setattr(
        "tools.lazy_deps.feature_missing",
        lambda feature: ("mautrix==0.21.0",) if feature == "platform.matrix" else (),
    )
    monkeypatch.setattr(
        "tools.lazy_deps.ensure",
        lambda feature, **_kw: calls.append(feature),
    )

    interactive_setup()

    assert calls == ["platform.matrix"]
    assert "MATRIX_ENCRYPTION" not in saved


def test_interactive_setup_e2ee_installs_base_and_e2ee_features(monkeypatch):
    saved = {}
    calls = []
    _patch_setup_io(
        monkeypatch,
        [
            "https://matrix.example.org",
            "syt_token",
            "@bot:example.org",
            "",
            "",
        ],
        [True],
        saved,
    )
    monkeypatch.setattr(
        "tools.lazy_deps.feature_missing",
        lambda feature: ("missing",),
    )
    monkeypatch.setattr(
        "tools.lazy_deps.ensure",
        lambda feature, **_kw: calls.append(feature),
    )
    monkeypatch.setattr(
        "plugins.platforms.matrix.adapter._check_e2ee_deps",
        lambda: False,
    )

    interactive_setup()

    assert calls == ["platform.matrix", "platform.matrix.e2ee"]
    assert saved["MATRIX_ENCRYPTION"] == "true"
