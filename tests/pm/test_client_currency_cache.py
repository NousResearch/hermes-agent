"""The launch-time currency probe must not spawn a PM worker for an unchanged install.

``venv_is_current`` answers a launch-time question on every ``hermes`` invocation, twice per boot
(the launcher re-execs into the store interpreter). Each answer used to cost a PM worker
(runtime validation plus start-up: ~10 s on a large NTFS home), so the probe missed the desktop's
15 s budget and the app fell through to the first-run bootstrap path. The memo is keyed on the
files that decide the dependency set, so it must be dropped the moment one of them changes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pm import client, environments


@pytest.fixture
def install(tmp_path, monkeypatch):
    state = tmp_path / "installs" / "key"
    state.mkdir(parents=True)
    environment = tmp_path / "venv"
    (environment / "Scripts").mkdir(parents=True)
    (environment / "Scripts" / "python.exe").write_bytes(b"x")
    (environment / "Lib" / "site-packages").mkdir(parents=True)
    (state / "facts.json").write_text(
        json.dumps({"packages": {"venv": {"environment": str(environment)}}}), encoding="utf-8")

    root = tmp_path / "checkout"
    (root / "pm").mkdir(parents=True)
    for name in ("pyproject.toml", "uv.lock"):
        (root / name).write_text("x", encoding="utf-8")
    (root / "pm" / "lock.json").write_text("x", encoding="utf-8")

    monkeypatch.setattr(environments, "install_state_dir", lambda project_root: state)
    # The verdict is about the *selected* environment; the real resolver reads the recorded
    # environment out of facts.json, which this fixture writes in its own minimal shape.
    monkeypatch.setattr(environments, "selected_venv", lambda project_root: environment)
    calls: list[tuple] = []
    monkeypatch.setattr(client, "_request", lambda operation, *args, **kwargs: calls.append(operation) or True)
    return root, state, environment, calls


def test_repeated_probe_reuses_the_memo(install):
    root, _state, _environment, calls = install

    assert client.venv_is_current(project_root=root) is True
    assert len(calls) == 1
    assert client.venv_is_current(project_root=root) is True
    assert len(calls) == 1, "an unchanged install must not spawn a second worker"
    assert client._cache_path(root).is_file()


def test_changed_environment_is_probed_again(install):
    root, _state, environment, calls = install
    client.venv_is_current(project_root=root)

    (environment / "Lib" / "site-packages" / "newdist").mkdir()
    assert client.venv_is_current(project_root=root) is True
    assert len(calls) == 2, "a distribution added outside PM must invalidate the memo"


def test_changed_activation_input_is_probed_again(install):
    root, _state, _environment, calls = install
    client.venv_is_current(project_root=root)

    lock = Path(root) / "uv.lock"
    lock.write_text("y" * 10, encoding="utf-8")
    assert client.venv_is_current(project_root=root) is True
    assert len(calls) == 2


def test_expired_memo_is_probed_again(install):
    root, _state, _environment, calls = install
    client.venv_is_current(project_root=root)

    memo = client._cache_path(root)
    payload = json.loads(memo.read_text(encoding="utf-8"))
    payload["at"] = time.time() - client._CURRENCY_CACHE_TTL_SECONDS - 1
    memo.write_text(json.dumps(payload), encoding="utf-8")
    assert client.venv_is_current(project_root=root) is True
    assert len(calls) == 2


def test_unreadable_facts_disable_the_memo(install):
    root, _state, _environment, calls = install
    (_state / "facts.json").write_text("not json", encoding="utf-8")

    assert client.venv_is_current(project_root=root) is True
    assert len(calls) == 1
    assert not client._cache_path(root).exists(), "no memo without a signature to key it on"


def test_plugin_members_join_the_signature(install):
    """``plugin_inputs.encode`` answers a dict, not a string: it must be hashed, not joined raw,
    and two different member sets must not share one memo entry."""
    from pm.plugin_inputs import Members

    root, _state, _environment, calls = install
    alpha = Members([Path("/tmp/alpha")])
    beta = Members([Path("/tmp/beta")])

    assert client.venv_is_current(project_root=root, plugins=alpha) is True
    assert len(calls) == 1
    assert client.venv_is_current(project_root=root, plugins=alpha) is True
    assert len(calls) == 1, "the same plugin members must reuse the memo"

    assert client.venv_is_current(project_root=root, plugins=beta) is True
    assert len(calls) == 2, "different plugin members ask a different question"


def test_no_members_is_a_different_question_than_empty_members(install):
    """``plugins=None`` lets the engine derive members from the enabled extras; ``Members([])``
    asks about no members at all. Collapsing both to one digest would answer one with the other."""
    from pm.plugin_inputs import Members

    root, _state, _environment, calls = install

    assert client.venv_is_current(project_root=root) is True
    assert len(calls) == 1
    assert client.venv_is_current(project_root=root, plugins=Members([])) is True
    assert len(calls) == 2, "an explicit empty member set is a different query"


def test_alternative_environment_selection_is_probed_again(install, monkeypatch):
    """A selection can move (recorded environment replaced, in-tree tree appearing) without any
    fingerprinted file changing, so the memo has to cover the selected environment too."""
    root, _state, environment, calls = install

    assert client.venv_is_current(project_root=root) is True
    assert len(calls) == 1
    other = environment.parent / "venv-other"
    (other / "Lib" / "site-packages").mkdir(parents=True)
    (other / "pyvenv.cfg").write_text("home = other\n", encoding="utf-8")
    monkeypatch.setattr(environments, "selected_venv", lambda project_root: other)

    assert client.venv_is_current(project_root=root) is True
    assert len(calls) == 2, "a different selected environment is a different query"


def test_unfingerprintable_query_never_answers_from_the_memo(install):
    """A ``Selection`` is a proposed plugin state only a sync may carry: the engine decides what it
    means, and a memo keyed on nothing must not answer for it. No signature means probe."""
    from pm.plugin_inputs import Selection

    root, _state, _environment, calls = install

    for _ in range(2):
        assert client.venv_is_current(project_root=root, plugins=Selection({"enabled": ["demo"]})) is True
    assert len(calls) == 2, "an unfingerprintable query is probed every time"
    assert not client._cache_path(root).exists(), "no memo without a question it can key on"
