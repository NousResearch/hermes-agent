"""Containment checks that survive case-insensitive filesystems (#127873).

``Path.is_relative_to`` is a purely lexical component compare, so on a case-insensitive volume
(macOS APFS, Windows) two spellings of the same tree compare unequal — ``resolve()`` preserves
the case that was passed in, and ``os.path.normcase`` folds only on Windows. A dependency
environment recorded in ``facts.json`` with one casing then reads as "outside this install" and
every ``no_agent=True`` script job fails with a misleading "missing" error.

Case-insensitivity is a filesystem property, so it is simulated through ``os.path.samefile`` —
the seam the identity fallback uses — rather than by patching platform identifiers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pm.environments import _is_within, _recorded_venv, install_state_dir


@pytest.fixture
def case_insensitive_volume(monkeypatch):
    """Simulate APFS/Windows: two paths are the same file iff they match case-insensitively."""
    def samefile(a, b):
        if str(a).lower() == str(b).lower():
            return True
        raise OSError(f"not the same file: {a} != {b}")

    monkeypatch.setattr(os.path, "samefile", samefile)


# ── the containment contract ─────────────────────────────────────────────────


def test_containment_is_lexical_when_the_case_agrees(tmp_path):
    """The fast path needs no stat call: identical spellings compare lexically."""
    parent, child = tmp_path / "environments", tmp_path / "environments" / "E1" / "venv"
    child.mkdir(parents=True)
    assert _is_within(child, parent) is True
    assert _is_within(parent, parent) is True


def test_case_mismatched_spellings_match_by_identity(tmp_path, case_insensitive_volume):
    """The reported shape: the recorded path and the derived path differ only in case."""
    parent = tmp_path / "installs" / "abc" / "environments"
    child = tmp_path / "INSTALLS" / "ABC" / "environments" / "E1" / "venv"
    assert _is_within(child, parent) is True


def test_a_path_outside_the_parent_is_still_rejected(tmp_path, case_insensitive_volume):
    """The fallback must not accept everything: a sibling tree is not the parent."""
    parent = tmp_path / "installs" / "abc" / "environments"
    assert _is_within(tmp_path / "elsewhere" / "venv", parent) is False
    # …and a path that merely shares a prefix (the classic startswith trap) is outside.
    assert _is_within(tmp_path / "installs" / "abc" / "environments-other" / "venv", parent) is False


def test_missing_parent_does_not_raise(tmp_path):
    """Nothing on disk: the ancestor walk must degrade to a plain False, not an OSError."""
    assert _is_within(tmp_path / "gone" / "venv", tmp_path / "never-existed") is False


# ── the recorded environment end to end ──────────────────────────────────────


def _write_facts(state_dir: Path, environment: Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "facts.json").write_text(
        json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8"
    )


def test_recorded_venv_accepts_a_case_mismatched_environment(tmp_path, monkeypatch, case_insensitive_volume):
    """The full gate: facts.json records one spelling, the install derives another, the venv is
    intact — the environment must load instead of raising "missing or outside this install"."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = tmp_path / "project"
    root.mkdir()
    derived = install_state_dir(root) / "environments"
    recorded = Path(str(derived).replace("/installs/", "/INSTALLS/")) / "E1" / "venv"
    recorded.mkdir(parents=True)
    (recorded / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    _write_facts(install_state_dir(root), recorded)

    assert _recorded_venv(root) == recorded.resolve()


def test_recorded_venv_still_rejects_an_environment_outside_the_install(tmp_path, monkeypatch):
    """The security side of the gate survives: an environment under another root still raises."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "somewhere-else" / "venv"
    outside.mkdir(parents=True)
    (outside / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    _write_facts(install_state_dir(root), outside)

    with pytest.raises(RuntimeError, match="missing or outside this install"):
        _recorded_venv(root)
