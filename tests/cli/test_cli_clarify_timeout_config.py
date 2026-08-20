"""Integration coverage for clarify timeout resolution through CLI config."""

import yaml


def _load_cli_config(tmp_path, monkeypatch, config):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )

    import cli

    monkeypatch.setattr(cli, "_hermes_home", tmp_path)
    return cli.load_cli_config()


def test_cli_uses_canonical_clarify_timeout(tmp_path, monkeypatch):
    """A CLI-only default must not mask agent.clarify_timeout."""
    from tools.clarify_gateway import resolve_clarify_timeout

    config = _load_cli_config(
        tmp_path,
        monkeypatch,
        {"agent": {"clarify_timeout": 987}},
    )

    assert resolve_clarify_timeout(config) == 987


def test_cli_preserves_explicit_legacy_clarify_timeout(tmp_path, monkeypatch):
    """An explicit clarify.timeout remains the backward-compatible override."""
    from tools.clarify_gateway import resolve_clarify_timeout

    config = _load_cli_config(
        tmp_path,
        monkeypatch,
        {
            "agent": {"clarify_timeout": 987},
            "clarify": {"timeout": 42},
        },
    )

    assert resolve_clarify_timeout(config) == 42
