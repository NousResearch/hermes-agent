"""PM's actual uv environment accepts UTF-8 pip config and tolerates undecodable files."""

from pm.environment import _base_environment


def test_bom_pip_config_reaches_uv_environment(tmp_path, monkeypatch):
    config = tmp_path / "pip.conf"
    monkeypatch.setenv("PIP_CONFIG_FILE", str(config))
    config.write_text("[global]\nindex-url = https://mirror.invalid/simple\n", encoding="utf-8-sig")
    assert _base_environment()["UV_INDEX_URL"] == "https://mirror.invalid/simple"

    config.write_text("[global]\nindex-url = https://plain.invalid/simple\n", encoding="utf-8")
    assert _base_environment()["UV_INDEX_URL"] == "https://plain.invalid/simple"


def test_undecodable_pip_config_does_not_abort_environment_build(tmp_path, monkeypatch):
    config = tmp_path / "pip.conf"
    config.write_bytes(b"[global]\nindex-url = \xff\n")
    monkeypatch.setenv("PIP_CONFIG_FILE", str(config))
    assert "UV_INDEX_URL" not in _base_environment()
