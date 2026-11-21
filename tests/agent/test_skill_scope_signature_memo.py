"""The change watcher's skills signature: one walk per root per pass, not one per scope.

The gateway signs every scope a session published commands for on one poll
(``agent.skill_commands.published_scopes_signature``), and scopes of the same profile share
that profile's skills root. The walk was repeated per scope: measured on the live dashboard,
32 watched scopes resolved to 1 unique root and the same tree was walked 27x over — 0.76s of
stat work per 2.0s poll, so the watcher never finished a pass and the loop never slept
(#146538).

These tests pin the memo's contract:

  * a root referenced by several scopes is walked ONCE per pass;
  * the signature it produces is IDENTICAL to the un-memoized one;
  * the memo does not outlive the pass, so a change between passes is still seen;
  * a change inside a shared root still moves the signature.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import agent.skill_commands as sc
import tools.skills_tool as st


def _make_skill(skills_dir, name):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}.\n---\n\nDo the thing.\n",
        encoding="utf-8")
    return skill_dir


def _counting_walk(monkeypatch):
    """Count the per-root walks the signature performs, and record which roots were walked."""
    walked: list[str] = []
    original = st._dir_tree_signature

    def counting(root, sig, depth=st._SIGNATURE_MAX_DEPTH):
        walked.append(str(root))
        return original(root, sig, depth)

    monkeypatch.setattr(st, "_dir_tree_signature", counting)
    return walked


def test_a_root_shared_by_many_scopes_is_walked_once_per_pass(tmp_path, monkeypatch):
    """The whole point: the shared tree is stat'ed once, not once per scope."""
    root = tmp_path / "skills"
    root.mkdir()
    _make_skill(root, "alpha")

    walked = _counting_walk(monkeypatch)
    with st.root_signature_memo():
        for _ in range(8):  # eight scopes of the same profile
            st._skills_scan_signature([root], set())

    # The walk recurses into the skill directory, so both paths appear — once each.
    assert walked.count(str(root)) == 1
    assert walked.count(str(root / "alpha")) == 1
    assert len(walked) == 2


def test_the_memoized_signature_is_identical_to_the_unmemoized_one(tmp_path, monkeypatch):
    """The memo is a speedup, never a different answer: same tuple, same order, same entries."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for root, name in ((root_a, "alpha"), (root_b, "beta")):
        root.mkdir()
        _make_skill(root, name)

    monkeypatch.setattr(st, "_dir_tree_signature", st._dir_tree_signature)  # keep the original
    plain = st._skills_scan_signature([root_a, root_b], {"alpha"})
    with st.root_signature_memo():
        memoized = st._skills_scan_signature([root_a, root_b], {"alpha"})
        # and again, now that both roots are in the memo
        again = st._skills_scan_signature([root_a, root_b], {"alpha"})

    assert memoized == plain
    assert again == plain


def test_the_memo_does_not_outlive_its_pass(tmp_path, monkeypatch):
    """A memo that outlived the pass would hide a change between passes — the whole detection."""
    root = tmp_path / "skills"
    root.mkdir()
    _make_skill(root, "alpha")

    walked = _counting_walk(monkeypatch)
    with st.root_signature_memo():
        st._skills_scan_signature([root], set())
    st._skills_scan_signature([root], set())  # a second pass: outside the memo

    assert walked.count(str(root)) == 2  # the second pass walked it again


def test_a_change_inside_a_shared_root_still_moves_the_signature(tmp_path, monkeypatch):
    """The end-to-end detection: install a skill, the next pass sees it."""
    root = tmp_path / "skills"
    root.mkdir()
    _make_skill(root, "alpha")

    def signatures():
        with st.root_signature_memo():
            return st._skills_scan_signature([root], set())

    before = signatures()
    _make_skill(root, "beta")  # installed between two passes
    after = signatures()

    assert before != after


def test_published_scopes_signature_walks_a_shared_root_once(monkeypatch, tmp_path):
    """The watcher's own entry point: several published scopes, one shared profile root."""
    home = tmp_path / "profile"
    root = home / "skills"
    root.mkdir(parents=True)
    _make_skill(root, "alpha")

    monkeypatch.setattr(sc, "_skill_commands_by_key", {
        ("local", str(home), f"/repo/{i}"): {} for i in range(6)
    })
    monkeypatch.setattr(sc, "_skill_commands_sig_by_key", {})
    monkeypatch.setattr(sc, "_scope_recency", {})

    roots = [root]
    monkeypatch.setattr(st, "skill_search_dirs", lambda: ([(0, r) for r in roots], root))

    walked = _counting_walk(monkeypatch)
    signature = sc.published_scopes_signature()

    assert len(signature) == 7  # the ambient scope plus the six published ones
    assert walked.count(str(root)) == 1  # walked ONCE for the whole pass


def test_the_memo_is_dropped_when_the_pass_raises(tmp_path, monkeypatch):
    """A pass that blows up must not leave a stale memo behind for the next one."""
    root = tmp_path / "skills"
    root.mkdir()
    _make_skill(root, "alpha")

    with pytest.raises(RuntimeError):
        with st.root_signature_memo():
            raise RuntimeError("boom")

    assert st._ROOT_SIG_MEMO.get() is None
    walked = _counting_walk(monkeypatch)
    st._skills_scan_signature([root], set())
    assert walked.count(str(root)) == 1


def test_the_memo_is_per_pass_not_shared_between_threads(tmp_path, monkeypatch):
    """A contextvar, not a module global: one pass's memo is not another pass's."""
    import threading

    root = tmp_path / "skills"
    root.mkdir()
    _make_skill(root, "alpha")

    seen: list[bool] = []

    def worker():
        seen.append(st._ROOT_SIG_MEMO.get() is not None)

    with st.root_signature_memo():
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)

    assert seen == [False]  # the worker's own context has no memo
    assert st._ROOT_SIG_MEMO.get() is None
