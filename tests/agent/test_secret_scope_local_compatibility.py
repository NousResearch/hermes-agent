"""Scoped dotenv bytes captured from the original local builder, not the launch resolver."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def local_homes(tmp_path, monkeypatch):
    import hermes_constants as hc
    import hermes_cli.env_loader as loader
    import hermes_cli.managed_scope as managed
    from gateway import config_loader

    launch, satellite = tmp_path / "launch", tmp_path / "satellite"
    launch.mkdir()
    satellite.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("HERMES_SECRET_STORE", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "launch-private-marker")
    monkeypatch.setenv("RV10_OUTSIDE", "launch-outside")
    monkeypatch.setattr(hc, "_PINNED_PROCESS_HERMES_HOME", launch)
    monkeypatch.setattr(loader, "get_secret_source_values", lambda home: {})
    monkeypatch.setattr(managed, "get_managed_dir", lambda: None)
    monkeypatch.setattr(managed, "load_managed_env", dict)
    monkeypatch.setattr(config_loader, "bridged_allow_all_users", lambda: None)
    return {"process": launch, "satellite": satellite}


_ROWS = json.loads((Path(__file__).parent / "fixtures/local_dotenv_base.json").read_text(encoding="utf-8"))["rows"]


@pytest.mark.parametrize("row", _ROWS, ids=[str(i) for i in range(len(_ROWS))])
def test_local_scope_matches_original_bytes(local_homes, row):
    from agent.secret_scope import build_profile_secret_scope

    home = local_homes[row["home"]]
    (home / ".env").write_text(row["text"], encoding="utf-8")
    assert build_profile_secret_scope(home)["RV10_VALUE"] == row["base"]


def test_satellite_interpolation_never_reads_launch_credential(local_homes):
    from agent.secret_scope import build_profile_secret_scope, get_secret, set_secret_scope

    home = local_homes["satellite"]
    (home / ".env").write_text("COPY=${OPENAI_API_KEY}\n", encoding="utf-8")
    with set_secret_scope(build_profile_secret_scope(home), profile_home=home):
        # A satellite's references resolve against process-global vars only (recorded from the base).
        assert get_secret("COPY") == ""
        assert "launch-private-marker" not in (get_secret("COPY") or "")
