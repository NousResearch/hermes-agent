"""`skills.signature_ttl_s`: the catalog walk is exact by default and reusable when opted in.

The freshness contract is the default: 0 means every call re-walks, so a skill written by another
process appears on the next palette open. Above zero, one walk serves the window — and every
in-process invalidation (mutation, reload) still forces an immediate re-walk.
"""
import os

import pytest


def _skill(root, name):
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\n---\n")
    return d


@pytest.fixture
def skills_root(tmp_path):
    import tools.skills_tool as st
    root = tmp_path / "skills"
    root.mkdir()
    _skill(root, "alpha")
    st._clear_signature_memo()
    yield root
    st._clear_signature_memo()


def _count_walks(monkeypatch, st):
    calls = []
    real = st._dir_tree_signature

    def counting(root, sig, depth=st._SIGNATURE_MAX_DEPTH):
        calls.append(root)
        return real(root, sig, depth)

    monkeypatch.setattr(st, "_dir_tree_signature", counting)
    return calls


def test_default_ttl_is_exact(skills_root, monkeypatch):
    """No config = today's contract: every call walks, so an external write is never missed."""
    import tools.skills_tool as st

    monkeypatch.setattr(st, "_signature_ttl_s", lambda: 0.0)
    calls = _count_walks(monkeypatch, st)
    st._compute_dirs_signature([skills_root])
    first = len(calls)
    st._compute_dirs_signature([skills_root])
    assert len(calls) == first * 2, "ttl=0 must not reuse a walk"


def test_ttl_reuses_one_walk(skills_root, monkeypatch):
    import tools.skills_tool as st

    monkeypatch.setattr(st, "_signature_ttl_s", lambda: 5.0)
    calls = _count_walks(monkeypatch, st)
    first = st._compute_dirs_signature([skills_root])
    walked = len(calls)
    assert st._compute_dirs_signature([skills_root]) == first
    assert len(calls) == walked, "inside the window the walk must be reused"


def test_ttl_expiry_re_walks(skills_root, monkeypatch):
    import tools.skills_tool as st

    monkeypatch.setattr(st, "_signature_ttl_s", lambda: 2.0)
    before = st._compute_dirs_signature([skills_root])
    _skill(skills_root, "beta")
    real = st.time.monotonic()
    monkeypatch.setattr(st.time, "monotonic", lambda: real + 3.0)
    assert st._compute_dirs_signature([skills_root]) != before


def test_a_reload_re_walks_inside_the_window(skills_root, monkeypatch):
    import tools.skills_tool as st

    monkeypatch.setattr(st, "_signature_ttl_s", lambda: 30.0)
    before = st._compute_dirs_signature([skills_root])
    _skill(skills_root, "gamma")
    st.clear_skills_cache()
    assert st._compute_dirs_signature([skills_root]) != before


def test_a_skill_write_re_walks_inside_the_window(skills_root, monkeypatch):
    """Mutations clear the prompt cache; the signature memo must die with it."""
    import tools.skills_tool as st
    from agent.prompt_builder import clear_skills_system_prompt_cache

    monkeypatch.setattr(st, "_signature_ttl_s", lambda: 30.0)
    before = st._compute_dirs_signature([skills_root])
    _skill(skills_root, "delta")
    clear_skills_system_prompt_cache(clear_snapshot=True)
    assert st._compute_dirs_signature([skills_root]) != before


def test_the_config_key_is_honored(monkeypatch):
    import agent.skill_utils as su
    import tools.skills_tool as st

    monkeypatch.setattr(su, "_load_raw_config", lambda: {"skills": {"signature_ttl_s": 3}})
    st._SIG_TTL.clear()
    assert st._signature_ttl_s() == 3.0


def test_a_broken_config_falls_back_to_exact(monkeypatch):
    import agent.skill_utils as su
    import tools.skills_tool as st

    def boom():
        raise RuntimeError("config unavailable")

    st._SIG_TTL.clear()
    monkeypatch.setattr(su, "_load_raw_config", boom)
    assert st._signature_ttl_s() == 0.0


def test_negative_values_are_treated_as_exact(monkeypatch):
    import agent.skill_utils as su
    import tools.skills_tool as st

    st._SIG_TTL.clear()
    monkeypatch.setattr(su, "_load_raw_config", lambda: {"skills": {"signature_ttl_s": -5}})
    assert st._signature_ttl_s() == 0.0


def test_the_ttl_knob_is_per_profile_home(tmp_path):
    """One process serves several profiles and each owns its own reuse window: a process-wide
    memo would serve the first profile read to every other one (a 60 s window to a profile that
    opted into exact freshness, and vice versa). Both orderings must hold."""
    import hermes_constants as hc
    import tools.skills_tool as st

    home_a = tmp_path / "cache" / "profile-a"  # a cache-named ancestor is just a path
    home_b = tmp_path / "profiles" / "profile-b"
    for home, ttl in ((home_a, 60), (home_b, 0)):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(f"skills:\n  signature_ttl_s: {ttl}\n", encoding="utf-8")

    st._SIG_TTL.clear()
    for home, expected in ((home_a, 60.0), (home_b, 0.0), (home_a, 60.0), (home_b, 0.0)):
        token = hc.set_hermes_home_override(home)
        try:
            assert st._signature_ttl_s() == expected, (
                f"{home.name}: the TTL memo crossed profile boundaries"
            )
        finally:
            hc.reset_hermes_home_override(token)


def test_a_skills_own_tree_stays_out_of_the_walk(skills_root):
    import tools.skills_tool as st

    d = _skill(skills_root, "zeta")
    (d / "references").mkdir()
    (d / "references" / "notes.md").write_text("x")
    walked = {os.path.basename(entry[0]) for entry in st._compute_dirs_signature([skills_root])}
    assert "references" not in walked
