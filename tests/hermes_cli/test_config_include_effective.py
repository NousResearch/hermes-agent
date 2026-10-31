import pytest

from hermes_cli import config
from hermes_cli.config_effective import load_user_config_effective


def test_effective_and_defaulted_caches_have_independent_include_signatures(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    root = tmp_path / "config.yaml"
    inc = tmp_path / "inc.yaml"
    root.write_text("include: inc.yaml\n", encoding="utf-8")
    inc.write_text("model:\n  default: first/model\n", encoding="utf-8")
    assert load_user_config_effective()["model"]["default"] == "first/model"
    assert config.load_config()["model"]["default"] == "first/model"
    assert "terminal" not in load_user_config_effective()
    inc.write_text("model:\n  default: second/model\n", encoding="utf-8")
    assert config.load_config()["model"]["default"] == "second/model"
    assert load_user_config_effective()["model"]["default"] == "second/model"
    inc.write_text("model:\n  default: third/model\n", encoding="utf-8")
    assert load_user_config_effective()["model"]["default"] == "third/model"
    assert config.load_config()["model"]["default"] == "third/model"
    assert config.read_raw_config() == {"include": "inc.yaml"}


def test_effective_invalid_include_keeps_last_good_but_fail_closed_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("include: inc.yaml\n", encoding="utf-8")
    inc = tmp_path / "inc.yaml"
    inc.write_text("model:\n  default: good/model\n", encoding="utf-8")
    assert load_user_config_effective()["model"]["default"] == "good/model"
    inc.write_text("model: [broken\n", encoding="utf-8")
    assert load_user_config_effective()["model"]["default"] == "good/model"
    with pytest.raises(ValueError, match="Invalid config include"):
        load_user_config_effective(fail_closed=True)
    inc.write_text("model:\n  default: repaired/model\n", encoding="utf-8")
    assert load_user_config_effective(fail_closed=True)["model"]["default"] == "repaired/model"
