"""``${env:VAR}`` refs must not warn before ``.env`` has been loaded.

Several entrypoints import ``hermes_cli.config`` — which expands config
refs as an import side effect (``_inject_profile_env_vars`` →
``providers.list_providers`` → ``plugins._get_enabled_plugins`` →
``load_config``) — strictly BEFORE they call ``load_hermes_dotenv()``.
A variable that lives only in ``~/.hermes/.env`` therefore looked unset at
that moment and printed

    Config ref '${env:X}': X is not set (check ~/.hermes/.env); keeping the
    literal placeholder

on every ``hermes <subcommand>`` invocation, even though the resolved value
was always correct (``load_config()``'s env-ref snapshot re-expands once the
environment changes — #58514).

These tests pin both halves of the contract: silence for a dotenv-only ref,
and a real warning for a genuinely-missing one.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from hermes_cli import env_loader
from hermes_cli.config import _expand_env_vars

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _reset_dotenv_flag(monkeypatch):
    """Each test owns the process-global "which homes have we loaded" set."""
    # raising=True on purpose: if the attribute is ever renamed, this fixture
    # must fail loudly rather than silently stop isolating the global.
    monkeypatch.setattr(env_loader, "_DOTENV_LOADED_HOMES", set(), raising=True)


# ---------------------------------------------------------------------------
# dotenv_pending() — the predicate the warning is gated on
# ---------------------------------------------------------------------------


def test_dotenv_pending_true_when_env_file_exists_and_unloaded(
    monkeypatch, tmp_path
):
    (tmp_path / ".env").write_text("FOO=bar\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert env_loader.dotenv_pending() is True


def test_dotenv_pending_false_when_no_env_file(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert env_loader.dotenv_pending() is False


def test_dotenv_pending_false_once_loaded(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("FOO=bar\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    env_loader._mark_dotenv_loaded(tmp_path)
    assert env_loader.dotenv_pending() is False


def test_load_hermes_dotenv_sets_the_loaded_flag(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("ORDERING_PROBE=1\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert env_loader.dotenv_pending() is True
    env_loader.load_hermes_dotenv(
        hermes_home=tmp_path, load_external_secrets=False
    )
    assert env_loader.dotenv_loaded() is True
    assert env_loader.dotenv_pending() is False


# ---------------------------------------------------------------------------
# The predicate and the loader must resolve the SAME home
# (@kyssta-exe, review of #117476)
# ---------------------------------------------------------------------------


def test_predicate_and_loader_resolve_the_same_home(monkeypatch, tmp_path):
    """Both sides must route through the one shared resolver.

    ``dotenv_pending()`` originally spelled the home resolution out for
    itself. Two independent spellings of "which .env do we mean" drift, and
    when they do the predicate reports on a different file than the loader
    reads — the gate then misfires or goes silent for the wrong home.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert env_loader._resolve_dotenv_home() == tmp_path
    # An explicit argument wins, exactly as load_hermes_dotenv() treats it.
    other = tmp_path / "explicit"
    assert env_loader._resolve_dotenv_home(other) == other


@pytest.mark.parametrize(
    "home_spec",
    ["~/probe_home", "$PROBE_ROOT/probe_home"],
    ids=["tilde", "envvar"],
)
def test_dotenv_pending_expands_hermes_home_like_the_loader(
    monkeypatch, tmp_path, caplog, home_spec
):
    """``HERMES_HOME`` may contain ``~`` or ``$VAR`` — both are expanded by
    ``get_process_hermes_home()``, the resolver ``load_hermes_dotenv()`` uses.

    A raw ``Path(os.getenv("HERMES_HOME"))`` treats ``~/probe_home`` as a
    literal relative directory, finds no ``.env`` there, returns ``False``,
    and re-arms the exact false warning this gate exists to suppress.
    """
    home = tmp_path / "probe_home"
    home.mkdir()
    (home / ".env").write_text("ONLY_IN_DOTENV=x\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Path.home() on Windows
    monkeypatch.setenv("PROBE_ROOT", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", home_spec)

    # Precondition: the two resolvers must agree on the path itself.
    from hermes_constants import get_process_hermes_home

    assert get_process_hermes_home() == home

    assert env_loader.dotenv_pending() is True

    # …and the expander must therefore stay silent.
    monkeypatch.delenv("ONLY_IN_DOTENV", raising=False)
    with caplog.at_level("WARNING"):
        assert _expand_env_vars("${env:ONLY_IN_DOTENV}") == "${env:ONLY_IN_DOTENV}"
    assert "ONLY_IN_DOTENV is not set" not in caplog.text


def test_loading_one_home_does_not_silence_another(monkeypatch, tmp_path):
    """``load_hermes_dotenv(hermes_home=...)`` takes an explicit home.

    Loading home B must not make the predicate report "already loaded" for
    home A, whose ``.env`` this process has never read — that would suppress
    the warning for A while A's vars are genuinely absent from ``os.environ``.
    """
    home_a = tmp_path / "a"
    home_a.mkdir()
    (home_a / ".env").write_text("A_ONLY=1\n", encoding="utf-8")
    home_b = tmp_path / "b"
    home_b.mkdir()
    (home_b / ".env").write_text("B_ONLY=1\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.delenv("A_ONLY", raising=False)  # a sibling test may have loaded it
    assert env_loader.dotenv_pending() is True

    env_loader.load_hermes_dotenv(
        hermes_home=home_b, load_external_secrets=False
    )

    # B is loaded; A is still unread, so A is still pending.
    assert env_loader.dotenv_loaded() is True
    assert env_loader.dotenv_pending() is True
    assert os.environ.get("A_ONLY") is None

    # Now load A for real — the predicate closes.
    env_loader.load_hermes_dotenv(
        hermes_home=home_a, load_external_secrets=False
    )
    assert env_loader.dotenv_pending() is False


def test_unresolved_ref_still_warns_after_a_DIFFERENT_home_loaded(
    monkeypatch, tmp_path, caplog
):
    """The end-to-end consequence of the cross-home bug.

    A single process-global "loaded" bool let home B's load silence the
    warning for home A — whose ``.env`` was never read — so a genuinely
    missing var under A went unreported. Only B is loaded first on purpose:
    loading A too would set the old bool for the right reason and the test
    would pass against the bug.
    """
    home_a = tmp_path / "a"
    home_a.mkdir()
    (home_a / ".env").write_text("A_ONLY=1\n", encoding="utf-8")
    home_b = tmp_path / "b"
    home_b.mkdir()
    (home_b / ".env").write_text("B_ONLY=1\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    env_loader.load_hermes_dotenv(
        hermes_home=home_b, load_external_secrets=False
    )
    monkeypatch.delenv("TRULY_MISSING_UNDER_A", raising=False)

    with caplog.at_level("WARNING"):
        _expand_env_vars("${env:TRULY_MISSING_UNDER_A}")

    # A's .env is still unread, so A is still pending -> correctly silent.
    assert "TRULY_MISSING_UNDER_A is not set" not in caplog.text

    # Once A is genuinely loaded, the missing var IS reported.
    env_loader.load_hermes_dotenv(
        hermes_home=home_a, load_external_secrets=False
    )
    with caplog.at_level("WARNING"):
        _expand_env_vars("${env:TRULY_MISSING_UNDER_A}")
    assert "TRULY_MISSING_UNDER_A is not set" in caplog.text


# ---------------------------------------------------------------------------
# The expander's warning behavior
# ---------------------------------------------------------------------------


def test_unresolved_ref_is_silent_while_dotenv_pending(
    monkeypatch, tmp_path, caplog
):
    """A pending .env suppresses the warning but NOT the placeholder."""
    (tmp_path / ".env").write_text("ONLY_IN_DOTENV=x\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("ONLY_IN_DOTENV", raising=False)

    with caplog.at_level("WARNING"):
        out = _expand_env_vars("${env:ONLY_IN_DOTENV}")

    assert out == "${env:ONLY_IN_DOTENV}"
    assert "keeping the literal placeholder" not in caplog.text


def test_unresolved_ref_still_warns_once_dotenv_is_loaded(
    monkeypatch, tmp_path, caplog
):
    """The real signal survives: after .env load, a missing var warns."""
    (tmp_path / ".env").write_text("SOMETHING_ELSE=x\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    env_loader._mark_dotenv_loaded(tmp_path)
    monkeypatch.delenv("TRULY_MISSING_REF", raising=False)

    with caplog.at_level("WARNING"):
        out = _expand_env_vars("${env:TRULY_MISSING_REF}")

    assert out == "${env:TRULY_MISSING_REF}"
    assert "TRULY_MISSING_REF is not set" in caplog.text


def test_unresolved_ref_warns_when_no_dotenv_exists_at_all(
    monkeypatch, tmp_path, caplog
):
    """No .env on disk → nothing is pending → warn as before."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("NO_DOTENV_REF", raising=False)

    with caplog.at_level("WARNING"):
        _expand_env_vars("${env:NO_DOTENV_REF}")

    assert "NO_DOTENV_REF is not set" in caplog.text


# ---------------------------------------------------------------------------
# End-to-end: a FRESH process on the real CLI import ordering
# ---------------------------------------------------------------------------


_E2E_PROGRAM = textwrap.dedent(
    """
    # Exactly the CLI's ordering: hermes_cli.config is imported (and expands
    # config refs as an import side effect) BEFORE load_hermes_dotenv() runs.
    import hermes_cli.config as _cfg
    from hermes_cli.env_loader import load_hermes_dotenv

    load_hermes_dotenv(load_external_secrets=False)

    from hermes_cli.config import load_config

    c = load_config()
    w = c["gateway"]["webhook"]
    print("SECRET=" + w["secret"])
    print("OTHER=" + w["other"])
    """
)


def _run_fresh_cli_process(home: Path) -> subprocess.CompletedProcess:
    env = {
        "HOME": str(home),
        "HERMES_HOME": str(home / ".hermes"),
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run(
        [sys.executable, "-c", _E2E_PROGRAM],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_fresh_process_resolves_dotenv_only_ref_without_warning(tmp_path):
    """The reported bug, end to end, in a real subprocess.

    ``secret`` reads a var that exists ONLY in ``.env``; ``other`` reads a
    var that exists nowhere. The first must resolve silently, the second
    must still warn — one run proves both directions.
    """
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        textwrap.dedent(
            """\
            model:
              default: claude-opus-4-6
            gateway:
              webhook:
                secret: '${env:ORDERING_DOTENV_ONLY}'
                other: '${env:ORDERING_NOWHERE}'
            """
        ),
        encoding="utf-8",
    )
    (hermes_home / ".env").write_text(
        "ORDERING_DOTENV_ONLY=resolved-from-dotenv\n", encoding="utf-8"
    )

    proc = _run_fresh_cli_process(home)
    combined = proc.stdout + proc.stderr

    assert proc.returncode == 0, combined
    assert "SECRET=resolved-from-dotenv" in proc.stdout, combined
    assert "OTHER=${env:ORDERING_NOWHERE}" in proc.stdout, combined
    # The dotenv-only var must never be reported as unset...
    assert "ORDERING_DOTENV_ONLY is not set" not in combined, combined
    # ...while the genuinely-missing one still is.
    assert "ORDERING_NOWHERE is not set" in combined, combined
