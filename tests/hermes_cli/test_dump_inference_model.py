"""`hermes dump` must report the EFFECTIVE inference model.

`hermes dump` prints `model:` from config.yaml (`model.default`), but the CLI
runtime resolves ``HERMES_INFERENCE_MODEL`` (env) OVER config — see
``oneshot.py``: ``effective_model = explicit_arg or env_model or cfg_model``.
Since ``run_dump()`` loads .env first, an operator who sets
``HERMES_INFERENCE_MODEL`` saw a dump claiming the config model while the agent
ran the env one — the same misleading-diagnostic class already fixed for the
terminal backend.
"""

from pathlib import Path
from types import SimpleNamespace


def _model_line(out: str) -> str:
    for line in out.splitlines():
        if line.startswith("model:"):
            return line
    raise AssertionError(f"no 'model:' line in dump output:\n{out}")


def _seed(home: Path, *, config_yaml: str, env_text: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config_yaml)
    (home / ".env").write_text(env_text)


def test_dump_surfaces_inference_model_env_override(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    # Keep run_dump's project-.env fallback from touching the real repo.
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    home = get_hermes_home()
    _seed(
        home,
        config_yaml="model:\n  default: hermes-4-405b\n",
        env_text="HERMES_INFERENCE_MODEL=anthropic/claude-opus-4\n",
    )

    dump.run_dump(SimpleNamespace(show_keys=False))

    line = _model_line(capsys.readouterr().out)
    # Effective model (env) is what actually runs, not the config default.
    assert "anthropic/claude-opus-4" in line
    assert "overrides config.yaml" in line
    # The shadowed config value is still shown so the mismatch is obvious.
    assert "model.default=hermes-4-405b" in line


def test_dump_reports_config_model_when_no_override(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    home = get_hermes_home()
    _seed(home, config_yaml="model:\n  default: hermes-4-405b\n", env_text="")

    dump.run_dump(SimpleNamespace(show_keys=False))

    line = _model_line(capsys.readouterr().out)
    assert "hermes-4-405b" in line
    assert "overrides" not in line


def test_dump_no_override_when_env_matches_config(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)
    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")

    home = get_hermes_home()
    # HERMES_INFERENCE_MODEL agrees with config — no spurious "override" note.
    _seed(
        home,
        config_yaml="model:\n  default: hermes-4-405b\n",
        env_text="HERMES_INFERENCE_MODEL=hermes-4-405b\n",
    )

    dump.run_dump(SimpleNamespace(show_keys=False))

    line = _model_line(capsys.readouterr().out)
    assert "hermes-4-405b" in line
    assert "overrides" not in line
