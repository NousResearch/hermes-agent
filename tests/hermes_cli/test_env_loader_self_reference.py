"""Self-referential .env lines (``PATH=/x:${PATH}``) must resolve once per process, not grow on every reload
(#109902: gateway per-turn and cron per-fire reloads grew PATH until child spawns died with E2BIG)."""

import os

import pytest

from hermes_cli.env_loader import load_hermes_dotenv

BASE_PATH = "/usr/bin:/bin"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", BASE_PATH)
    monkeypatch.setenv("HERMES_MULTIPLEX_PROFILES", "0")
    h = tmp_path / "hermes"
    h.mkdir()
    return h


def test_same_file_reloaded_three_times_leaves_path_unchanged(home):
    (home / ".env").write_text("PATH=/x:${PATH}\nPREV=${PATH}\nCHAIN=/c:${PREV}\n")

    seen = []
    for _ in range(3):
        load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
        seen.append((os.environ["PATH"], os.environ["CHAIN"]))

    assert seen == [(f"/x:{BASE_PATH}", f"/c:/x:{BASE_PATH}")] * 3


def test_alternating_project_env_and_none_stays_stable(home, tmp_path):
    """The gateway reloads with a project .env, cron without: both call forms share one process."""
    (home / ".env").write_text("PATH=/x:${PATH}\n")
    project_env = tmp_path / "project.env"
    project_env.write_text("UNRELATED=1\n")

    counts = []
    for i in range(6):
        load_hermes_dotenv(
            hermes_home=home, project_env=project_env if i % 2 else None, load_external_secrets=False
        )
        counts.append(os.environ["PATH"].count("/x:"))

    assert counts == [1] * 6


def test_alternating_homes_each_resolve_against_boot_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", BASE_PATH)
    monkeypatch.setenv("HERMES_MULTIPLEX_PROFILES", "0")
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    for h, tag in ((home_a, "a"), (home_b, "b")):
        h.mkdir()
        (h / ".env").write_text(f"PATH=/{tag}:${{PATH}}\n")

    seen = []
    for h in (home_a, home_b, home_a, home_b, home_a):
        load_hermes_dotenv(hermes_home=h, load_external_secrets=False)
        seen.append(os.environ["PATH"])

    assert seen == [f"/a:{BASE_PATH}", f"/b:{BASE_PATH}", f"/a:{BASE_PATH}", f"/b:{BASE_PATH}", f"/a:{BASE_PATH}"]


def test_variable_unset_at_boot_and_default_syntax_resolve_once(home, monkeypatch):
    monkeypatch.delenv("FOO", raising=False)
    monkeypatch.delenv("DEF", raising=False)
    (home / ".env").write_text("FOO=/f:${FOO}\nDEF=/d:${DEF:-seed}\n")

    for _ in range(3):
        load_hermes_dotenv(hermes_home=home, load_external_secrets=False)

    assert os.environ["FOO"] == "/f:"
    assert os.environ["DEF"] == "/d:seed"


def test_value_changed_outside_the_loader_is_not_frozen(home):
    """Only this process's own dotenv output is peeled; a newer outside value (shell, config bridge,
    secret source) becomes the new baseline instead of being reverted."""
    (home / ".env").write_text("PATH=/x:${PATH}\n")

    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    os.environ["PATH"] = "/opt/new:/bin"
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)
    load_hermes_dotenv(hermes_home=home, load_external_secrets=False)

    assert os.environ["PATH"] == "/x:/opt/new:/bin"


def test_later_layer_in_the_same_load_still_sees_the_earlier_layer(home, tmp_path):
    """Within one load_hermes_dotenv the project layer builds on the user layer, as before; the
    combination is what must not grow across reloads."""
    (home / ".env").write_text("XPATH=/x:${PATH}\n")
    project_env = tmp_path / "project.env"
    project_env.write_text("YPATH=/p:${XPATH}\n")

    for _ in range(3):
        load_hermes_dotenv(hermes_home=home, project_env=project_env, load_external_secrets=False)

    assert os.environ["XPATH"] == f"/x:{BASE_PATH}"
    assert os.environ["YPATH"] == f"/p:/x:{BASE_PATH}"
