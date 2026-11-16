"""The remote store's view is installed once at start, as the profile ``.env`` layer.

Precedence is the local store's (the same layer order, compared against a real local ``.env``);
stored strings are literal (never expanded on the way in, though a later layer may expand them like
any value); the one-time import stores what the original ambient load published; a satellite
profile's import never sees a launch credential.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import hermes_secret_routing as routing
from tests.plugins.secret_store.test_remote_secret_store import _boot as _boot_with_sources

pytest_plugins = ["tests.plugins.secret_store.test_remote_secret_store"]

FIXTURE = json.loads((Path(__file__).parents[2] / "agent/fixtures/local_dotenv_ambient_base.json")
                     .read_text(encoding="utf-8"))


def _boot(home, project=None):
    from hermes_cli.env_loader import load_hermes_dotenv
    load_hermes_dotenv(hermes_home=home, project_env=project, load_external_secrets=False)


def _readers(home, name, *, ambient=True):
    """Every served value of *name*: bound get_secret, the scope, the resolved dotenv layers, a
    credential-inheriting child and (process home) ``os.environ``."""
    from agent.secret_scope import build_profile_secret_scope, get_secret, set_secret_scope
    from hermes_cli.env_loader import resolve_home_dotenv
    from tools.environments.local import served_profile_child_env
    scope = build_profile_secret_scope(home)
    with set_secret_scope(scope, profile_home=home):
        got = {"get_secret": get_secret(name), "scope": scope.get(name),
               "child": served_profile_child_env(target_home=home, inherit_credentials=True).get(name)}
    got["resolved"] = resolve_home_dotenv(home).get(name)
    if ambient:
        got["ambient"] = os.environ.get(name)
    return got


PROFILE = {"LAYER_SHELL": "profile-shell", "LAYER_PROJECT": "profile-project",
           "LAYER_MANAGED": "profile-managed", "LAYER_ONLY": "profile-only"}
NAMES = (*PROFILE, "LAYER_PROJECT_ONLY", "LAYER_PROJECT_REF")


@pytest.mark.parametrize("store", ["local", "remote"])
def test_the_view_takes_the_profile_env_files_place_in_the_layer_order(plane, tmp_path, monkeypatch, store):
    """Same layers, same answers: shell exports lose to the profile layer, the project ``.env``
    fills gaps (and may expand a profile value), the managed ``.env`` wins last."""
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAYER_SHELL", "shell")
    project = tmp_path / "project.env"
    project.write_text("LAYER_PROJECT=project\nLAYER_PROJECT_ONLY=project-only\n"
                       "LAYER_PROJECT_REF=${LAYER_ONLY}/ref\n", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("LAYER_MANAGED=managed\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    if store == "local":
        monkeypatch.delenv("HERMES_SECRET_STORE")
        routing.reset_for_tests()
        (plane.home / ".env").write_text("".join(f"{k}={v}\n" for k, v in PROFILE.items()), encoding="utf-8")
    else:
        plane.set_level(plane.profile("default"), env=PROFILE)
    _boot(plane.home, project)
    assert {name: os.environ.get(name) for name in NAMES} == {
        "LAYER_SHELL": "profile-shell", "LAYER_PROJECT": "profile-project", "LAYER_MANAGED": "managed",
        "LAYER_ONLY": "profile-only", "LAYER_PROJECT_ONLY": "project-only", "LAYER_PROJECT_REF": "profile-only/ref"}


LITERALS = ["${OUTSIDE}", "${OUTSIDE:-fallback}", "${:-$}{OUTSIDE}", "$$", "${", "end$", r"\$", "a'b\"c",
            "back\\slash", " spaced ", "#hash", "multi\nline", "${LITERAL_SELF}", "${LITERAL_OTHER}"]


@pytest.mark.parametrize("value", LITERALS)
def test_a_stored_string_is_installed_verbatim(plane, monkeypatch, value):
    monkeypatch.setenv("OUTSIDE", "expanded-marker")
    monkeypatch.delenv("LITERAL_OTHER", raising=False)
    monkeypatch.delenv("LITERAL_SELF", raising=False)
    plane.set_level(plane.profile("default"), env={"LITERAL_SELF": value, "LITERAL_OTHER": "other-marker"})
    _boot(plane.home)
    assert set(_readers(plane.home, "LITERAL_SELF").values()) == {value}
    _boot(plane.home)  # a reload expands nothing either
    assert os.environ["LITERAL_SELF"] == value


def test_a_project_line_expands_a_stored_value_once(plane, tmp_path, monkeypatch):
    """Other layers expand stored values normally; the stored text is never expanded again."""
    monkeypatch.setenv("EXPAND_BASE", "launch")
    plane.set_level(plane.profile("default"), env={"EXPAND_BASE": "profile", "LITERAL": "${EXPAND_BASE}"})
    project = tmp_path / "project.env"
    project.write_text("PROJECT_REF=${EXPAND_BASE}/${LITERAL}\n", encoding="utf-8")
    _boot(plane.home, project)
    assert os.environ["LITERAL"] == "${EXPAND_BASE}"
    assert os.environ["PROJECT_REF"] == "profile/${EXPAND_BASE}"


def test_bootstrap_lines_still_load_from_the_file(plane, monkeypatch):
    monkeypatch.delenv("TERMINAL_BOOT_PARSE", raising=False)
    with open(plane.home / ".env", "a", encoding="utf-8") as f:
        f.write('TERMINAL_BOOT_PARSE="agents\\tquoted"\n')  # process-global: stays local
    _boot(plane.home)
    assert os.environ["TERMINAL_BOOT_PARSE"] == "agents\tquoted"  # the loader's parser: a tab
    assert plane.patches() == []


@pytest.mark.parametrize("row", FIXTURE["rows"], ids=lambda row: row["home"])
def test_import_stores_the_original_ambient_load_result(plane, monkeypatch, row):
    for name, value in FIXTURE["launch"].items():
        monkeypatch.setenv(name, value)
    satellite = row["home"] == "satellite"
    home = plane.home.parent / "satellite" if satellite else plane.home
    home.mkdir(exist_ok=True)
    (home / ".env").write_text(row["text"], encoding="utf-8")
    if satellite:
        from agent.secret_scope import build_profile_secret_scope
        build_profile_secret_scope(home)
    else:
        _boot(home)
    stored = {name: value for name, (value, _version) in plane.profile(home.name)["env"].items()}
    assert stored == row["values"]
    for name, value in row["values"].items():
        assert set(_readers(home, name, ambient=not satellite).values()) == {value}, name


def test_a_satellite_import_never_receives_a_launch_credential(plane, monkeypatch):
    from agent.secret_scope import build_profile_secret_scope
    monkeypatch.setenv("OPENAI_API_KEY", "launch-credential-marker")
    _boot(plane.home)
    home = plane.home.parent / "satellite"
    home.mkdir()
    (home / ".env").write_text("COPY=${OPENAI_API_KEY}\nOWN=${OPENAI_API_KEY:-own-default}\n", encoding="utf-8")
    scope = build_profile_secret_scope(home)
    assert "launch-credential-marker" not in json.dumps(plane.patches())
    assert scope.get("COPY") == "" and scope.get("OWN") == "own-default"


def test_an_imported_launch_reference_is_frozen(plane, monkeypatch):
    """Accepted consequence of literal values: a launch variable is resolved once, at import."""
    monkeypatch.setenv("IMPORT_GLOBAL_SOURCE", "first-marker")
    (plane.home / ".env").write_text(
        'IMPORT_INTERNAL="${IMPORT_GLOBAL_SOURCE}/suffix"\nIMPORT_REF=${IMPORT_INTERNAL}\n', encoding="utf-8")
    _boot_with_sources(plane.home)
    monkeypatch.setenv("IMPORT_GLOBAL_SOURCE", "second-marker")
    _boot(plane.home)
    assert set(_readers(plane.home, "IMPORT_REF").values()) == {"first-marker/suffix"}


_KEPT_NAMES = ("RV_DIR", "RV_A", "RV_B", "RV_LATE", "RV_MISSING", "TERMINAL_RV_REF", "TERMINAL_RV_OVR",
               "TERMINAL_RV_A", "TERMINAL_RV_B")


def _fresh_start(monkeypatch, shell):
    """As a new process: no provider state, no earlier publication, only the shell's exports."""
    from hermes_cli import env_loader
    from plugins.secret_store import remote as remote_pkg
    remote_pkg._reset_for_tests()
    routing.reset_for_tests()
    with env_loader._DOTENV_LOCK:
        env_loader._DOTENV_PUBLISHED.clear()
    for name in _KEPT_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in shell.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("store", ["local", "remote"])
def test_a_satellite_import_sees_process_globals_as_they_are_after_the_launch_load(plane, monkeypatch, store):
    """The original satellite builder resolves a satellite ``.env`` against the process-global names
    of the live environment, so a global the launch ``.env`` set is seen as set (base-executed:
    fixture row 127 after the launch load gives ``launch-dotenv-global/``); a launch credential never."""
    from agent.secret_scope import build_profile_secret_scope
    for name, value in FIXTURE["launch"].items():
        monkeypatch.setenv(name, value)
    if store == "local":
        monkeypatch.delenv("HERMES_SECRET_STORE")
        routing.reset_for_tests()
    (plane.home / ".env").write_text("RV10_VALUE=launch-dotenv-RV10_VALUE\nTERMINAL_RV10_GLOBAL=launch-dotenv-global\n",
                                     encoding="utf-8")
    _boot(plane.home)
    satellite = plane.home.parent / "satellite"
    satellite.mkdir()
    (satellite / ".env").write_text(FIXTURE["rows"][127]["text"], encoding="utf-8")
    assert build_profile_secret_scope(satellite).get("RV10_VALUE") == "launch-dotenv-global/"
    if store == "remote":
        stored = plane.profile("satellite")["env"]
        assert {k: v for k, (v, _version) in stored.items()} == {"RV10_VALUE": "launch-dotenv-global/"}
        assert FIXTURE["launch"]["OPENAI_API_KEY"] not in json.dumps(plane.patches())
