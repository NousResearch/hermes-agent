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
    monkeypatch.setenv("SCOPE_SOURCE", "agents")
    with open(plane.home / ".env", "a", encoding="utf-8") as f:
        f.write('TERMINAL_BOOT_PARSE="${SCOPE_SOURCE}\\tquoted"\n')  # process-global: stays local
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


# Kept-local (bootstrap / process-global) lines stay in the file and are re-read at every start. The
# import freezes each one that references an imported (or any non-kept) name to the value the original
# loader gave that line, as it freezes every imported value; a kept line referencing only kept names
# stays as written. Expected values were produced by executing the base loader (c2784094b6, local
# store) on each text in a fresh process with only the listed shell exports.
KEPT_LOCAL = {
    "backward": ("RV_DIR=/x\nTERMINAL_RV_REF=${RV_DIR}/sub\n", {}, {"TERMINAL_RV_REF": "/x/sub"}),
    "backward_shell": ("RV_DIR=/x\nTERMINAL_RV_REF=${RV_DIR}/sub\n", {"RV_DIR": "shell-marker"},
                       {"TERMINAL_RV_REF": "/x/sub"}),
    "forward": ("TERMINAL_RV_REF=${RV_DIR}/sub\nRV_DIR=/x\n", {}, {"TERMINAL_RV_REF": "/sub"}),
    "forward_shell": ("TERMINAL_RV_REF=${RV_DIR}/sub\nRV_DIR=/x\n", {"RV_DIR": "shell-marker"},
                      {"TERMINAL_RV_REF": "shell-marker/sub"}),
    "duplicate_before": ("RV_DIR=/x\nRV_DIR=/y\nTERMINAL_RV_REF=${RV_DIR}/sub\n", {}, {"TERMINAL_RV_REF": "/y/sub"}),
    "duplicate_spanning": ("RV_DIR=/x\nTERMINAL_RV_REF=${RV_DIR}/sub\nRV_DIR=/y\n", {}, {"TERMINAL_RV_REF": "/x/sub"}),
    "duplicate_spanning_two": ("RV_A=1\nRV_B=1\nTERMINAL_RV_REF=${RV_A}${RV_B}\nRV_A=3\n", {}, {"TERMINAL_RV_REF": "11"}),
    "chain_through_imported": (
        "RV_DIR=/x\nTERMINAL_RV_A=${RV_DIR}/a\nRV_LATE=${TERMINAL_RV_A}/late\nTERMINAL_RV_B=${RV_LATE}/b\n", {},
        {"TERMINAL_RV_A": "/x/a", "TERMINAL_RV_B": "/x/a/late/b"}),
    "chain_through_kept": (
        "RV_DIR=/x\nTERMINAL_RV_A=${RV_DIR}/a\nTERMINAL_RV_B=${TERMINAL_RV_A}/b\nRV_DIR=/y\n", {},
        {"TERMINAL_RV_A": "/x/a", "TERMINAL_RV_B": "/x/a/b"}),
    "default": ("RV_DIR=/x\nTERMINAL_RV_REF=${RV_DIR:-unset}/${RV_MISSING:-dflt}\n", {}, {"TERMINAL_RV_REF": "/x/dflt"}),
    "default_forward": ("TERMINAL_RV_REF=${RV_DIR:-unset}\nRV_DIR=/x\n", {}, {"TERMINAL_RV_REF": "unset"}),
    "kept_line_overriding": ("TERMINAL_RV_OVR=local-wins\nRV_DIR=/x\nTERMINAL_RV_REF=${RV_DIR}/${TERMINAL_RV_OVR}\n", {},
                             {"TERMINAL_RV_REF": "/x/local-wins", "TERMINAL_RV_OVR": "local-wins"}),
    "kept_name_reassigned": ("TERMINAL_RV_REF=${RV_DIR}/first\nRV_DIR=/x\nTERMINAL_RV_REF=${RV_DIR}/second\n", {},
                             {"TERMINAL_RV_REF": "/x/second"}),
    "kept_line_over_shell": ("RV_DIR=/x\nTERMINAL_RV_OVR=${RV_DIR}/o\n", {"TERMINAL_RV_OVR": "shell-marker"},
                             {"TERMINAL_RV_OVR": "/x/o"}),
    "quoted_value": ("RV_DIR=\"/x y #z \\\"q\\\" \\\\b 'c'\"\nTERMINAL_RV_REF=${RV_DIR}/sub\n", {},
                     {"TERMINAL_RV_REF": "/x y #z \"q\" \\b 'c'/sub"}),
    "only_kept_refs": ("TERMINAL_RV_OVR=o\nRV_DIR=/x\nTERMINAL_RV_REF=${TERMINAL_RV_OVR}/k\n", {},
                       {"TERMINAL_RV_REF": "o/k", "TERMINAL_RV_OVR": "o"}),
}
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
@pytest.mark.parametrize("case", KEPT_LOCAL)
def test_a_kept_local_line_resolves_as_the_original_file_did(plane, monkeypatch, store, case):
    """At the import's start and at every later start (the plane holds the values, the file its
    kept lines only), each kept name has the value the base loader gives the original file."""
    from dotenv.variables import Variable, parse_variables
    from hermes_cli.env_loader import _parse_local_dotenv_file
    text, shell, expected = KEPT_LOCAL[case]
    _fresh_start(monkeypatch, shell)
    if store == "local":
        monkeypatch.delenv("HERMES_SECRET_STORE")
        routing.reset_for_tests()
    (plane.home / ".env").write_text(text, encoding="utf-8")
    _boot(plane.home)
    for name, value in expected.items():
        assert set(_readers(plane.home, name).values()) == {value}, (name, store)
    if store == "local":
        assert (plane.home / ".env").read_text(encoding="utf-8") == text
        return
    left = _parse_local_dotenv_file(plane.home / ".env")
    assert all(name.startswith("TERMINAL_") for name, _value in left)
    for name, value in left:   # frozen, or referencing kept names only
        assert all(a.name.startswith("TERMINAL_") for a in parse_variables(value) if isinstance(a, Variable))
    _fresh_start(monkeypatch, shell)   # every later start: same answer
    _boot(plane.home)
    for name, value in expected.items():
        assert set(_readers(plane.home, name).values()) == {value}, (name, "later start")


def test_a_kept_line_referencing_only_kept_names_stays_as_written(plane, monkeypatch):
    text, shell, _expected = KEPT_LOCAL["only_kept_refs"]
    _fresh_start(monkeypatch, shell)
    (plane.home / ".env").write_text("# kept\n" + text, encoding="utf-8")
    _boot(plane.home)
    assert (plane.home / ".env").read_text(encoding="utf-8") == (
        "# kept\nTERMINAL_RV_OVR=o\nTERMINAL_RV_REF=${TERMINAL_RV_OVR}/k\n")


@pytest.mark.parametrize("value", ["plain", "", "a b", "  lead", "x#y", 'q"q', "s'q", "back\\slash", "\\\"",
                                   "tab\tin", "ünïcode", "$NOT_A_REF", "trail\\"])
def test_a_frozen_kept_line_reads_back_as_the_value_it_froze(plane, monkeypatch, value):
    """The frozen line is written as ``save_env_value`` writes a value; the local loader reads back
    exactly the value the original file gave that line."""
    _fresh_start(monkeypatch, {"RV_SHELL": value})
    (plane.home / ".env").write_text("TERMINAL_RV_REF=${RV_SHELL}\nRV_DIR=/x\n", encoding="utf-8")
    _boot(plane.home)
    _fresh_start(monkeypatch, {})
    monkeypatch.delenv("RV_SHELL", raising=False)
    _boot(plane.home)
    assert set(_readers(plane.home, "TERMINAL_RV_REF").values()) == {value}


# A value no single physical line carries through BOTH local readers (python-dotenv for the launch
# load, the one-line tokenizer for scopes and bound get_secret) cannot be frozen: the import leaves
# that kept line exactly as written, and it resolves at each start as before. Expected values were
# produced by executing the base loader (06189b9508, local store) on each text.
MULTILINE = {
    "escaped_newline": ('RV_DIR="l1\\nl2"\nTERMINAL_RV_REF=${RV_DIR}|\n', "l1\nl2|"),
    "physical_newline": ('RV_DIR=/x\nTERMINAL_RV_REF="${RV_DIR}\nsecond"\n', "/x\nsecond"),
}


@pytest.mark.parametrize("store", ["local", "remote"])
@pytest.mark.parametrize("case", MULTILINE)
def test_a_kept_line_resolving_to_a_multiline_value_is_served_whole(plane, monkeypatch, store, case):
    text, expected = MULTILINE[case]
    _fresh_start(monkeypatch, {})
    if store == "local":
        monkeypatch.delenv("HERMES_SECRET_STORE")
        routing.reset_for_tests()
    (plane.home / ".env").write_text(text, encoding="utf-8")
    _boot(plane.home)
    assert set(_readers(plane.home, "TERMINAL_RV_REF").values()) == {expected}
    if store == "remote":   # the kept line is left exactly as written
        assert (plane.home / ".env").read_text(encoding="utf-8") == text.split("\n", 1)[1]
    _fresh_start(monkeypatch, {})
    _boot(plane.home)
    assert set(_readers(plane.home, "TERMINAL_RV_REF").values()) == {expected}


def _generated_values():
    import random
    rng = random.Random(0x5EC2)
    alphabet = ["a", "Z", "0", " ", "\t", "\n", "\r", '"', "'", "\\", "$", "{", "}", "${", "${RV_DIR}",
                "${X:-d}", "#", "=", "n", "\u2028", "\x0c", "\u00e9", "export "]
    values = {"".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6))) for _ in range(160)}
    return sorted(values | {"\n", "\r", "a\r\nb", "\\n", '"\n"', "${", "$", "${RV_DIR}", "x\ty", "\u2028"})


_BREAKS = "\n\r\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"   # str.splitlines() boundaries


@pytest.mark.parametrize("value", _generated_values(), ids=lambda v: repr(v)[1:-1][:40])
def test_a_kept_line_is_frozen_only_when_both_readers_read_back_the_frozen_value(plane, monkeypatch, value):
    """Generated values reach a kept line through a launch reference. Either the line was frozen and
    every reader (the launch load, the scope tokenizer, bound get_secret, ...) serves exactly the
    value at every later start, or the import left the line byte for byte as written. A one-line
    value with no reference in it always freezes."""
    from dotenv.variables import Variable, parse_variables
    _fresh_start(monkeypatch, {"RV_SHELL": value})
    (plane.home / ".env").write_text("TERMINAL_RV_REF=${RV_SHELL}\nRV_DIR=/x\n", encoding="utf-8")
    _boot(plane.home)
    left = (plane.home / ".env").read_text(encoding="utf-8")
    if left == "TERMINAL_RV_REF=${RV_SHELL}\n":
        assert any(c in value for c in _BREAKS) or any(isinstance(a, Variable) for a in parse_variables(value))
        return
    assert "${RV_SHELL}" not in left
    _fresh_start(monkeypatch, {})
    monkeypatch.delenv("RV_SHELL", raising=False)
    _boot(plane.home)
    assert set(_readers(plane.home, "TERMINAL_RV_REF").values()) == {value}


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
