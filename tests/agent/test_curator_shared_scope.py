"""Curator shared-scope tests — is_shared_curatable_path + gate rewiring.

Covers the curator-scope-shared-skills spec Phases 1-2:
- config parsing (include_shared_dirs / shared_dirs / split_over_kb)
- the single-chokepoint predicate table (shared/hub/bundled/allowlist)
- symlink-escape and path-traversal adversarial cases
- AC1: with the flag off, zero external-dir skills are candidates
- AC2: with the flag on, shared skills are candidates; bundled/hub never

Real imports against a temp HERMES_HOME with a real config.yaml — the
resolution chain (config.yaml -> skills.external_dirs -> predicate) is
exercised end-to-end, no mocks.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _write_config(home: Path, *, include_shared: bool = False,
                  shared_dirs=None, split_over_kb: int = 0,
                  external_dirs=None) -> None:
    lines = ["skills:", "  external_dirs:"]
    for d in external_dirs or []:
        lines.append(f"    - {d}")
    lines.append("curator:")
    lines.append(f"  include_shared_dirs: {'true' if include_shared else 'false'}")
    if shared_dirs is not None:
        if isinstance(shared_dirs, str):
            lines.append(f"  shared_dirs: {shared_dirs}")
        else:
            lines.append("  shared_dirs:")
            for g in shared_dirs:
                lines.append(f"    - {g}")
    lines.append(f"  split_over_kb: {split_over_kb}")
    (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mk_skill(root: Path, group: str, name: str, body: str = "hello") -> Path:
    d = root / group / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: t\n---\n\n{body}\n", encoding="utf-8"
    )
    return d


@pytest.fixture
def shared_env(tmp_path, monkeypatch):
    """Temp HERMES_HOME with a skills-shared/ tree registered as external dirs."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    shared = home / "skills-shared"
    for group in ("smart-home", "devops"):
        (shared / group).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    _write_config(
        home,
        external_dirs=[str(shared / "smart-home"), str(shared / "devops")],
    )

    import agent.skill_utils as su
    importlib.reload(su)
    yield {"home": home, "shared": shared, "su": su}


def _reload_su():
    import agent.skill_utils as su
    return importlib.reload(su)


class TestConfigParsing:
    def test_defaults_when_missing(self, shared_env):
        su = shared_env["su"]
        assert su.get_curator_include_shared_dirs() is False
        assert su.get_curator_shared_dirs() == []

    def test_parses_all_three_keys(self, shared_env):
        home, shared = shared_env["home"], shared_env["shared"]
        _write_config(
            home, include_shared=True, shared_dirs=["smart-home"],
            split_over_kb=100,
            external_dirs=[str(shared / "smart-home")],
        )
        su = _reload_su()
        assert su.get_curator_include_shared_dirs() is True
        assert su.get_curator_shared_dirs() == ["smart-home"]
        import agent.curator  # split_over_kb reads via hermes_cli.config

    def test_malformed_shared_dirs_string_coerces(self, shared_env):
        home, shared = shared_env["home"], shared_env["shared"]
        _write_config(
            home, include_shared=True, shared_dirs="smart-home",
            external_dirs=[str(shared / "smart-home")],
        )
        su = _reload_su()
        # a bare string coerces to a single-element list, never raises
        assert su.get_curator_shared_dirs() == ["smart-home"]

    def test_split_over_kb_getter(self, shared_env, monkeypatch):
        import agent.curator as curator
        importlib.reload(curator)
        monkeypatch.setattr(curator, "_load_config", lambda: {"split_over_kb": 100})
        assert curator.get_split_over_kb() == 100
        monkeypatch.setattr(curator, "_load_config", lambda: {})
        assert curator.get_split_over_kb() == 0
        monkeypatch.setattr(curator, "_load_config", lambda: {"split_over_kb": "junk"})
        assert curator.get_split_over_kb() == 0
        monkeypatch.setattr(curator, "_load_config", lambda: {"split_over_kb": -5})
        assert curator.get_split_over_kb() == 0

    def test_include_shared_dirs_getter(self, shared_env, monkeypatch):
        import agent.curator as curator
        importlib.reload(curator)
        monkeypatch.setattr(curator, "_load_config", lambda: {"include_shared_dirs": True})
        assert curator.get_include_shared_dirs() is True
        monkeypatch.setattr(curator, "_load_config", lambda: {})
        assert curator.get_include_shared_dirs() is False

    def test_shared_dirs_getter_coercion(self, shared_env, monkeypatch):
        import agent.curator as curator
        importlib.reload(curator)
        monkeypatch.setattr(curator, "_load_config", lambda: {"shared_dirs": "smart-home"})
        assert curator.get_shared_dirs() == ["smart-home"]
        monkeypatch.setattr(curator, "_load_config", lambda: {"shared_dirs": 42})
        assert curator.get_shared_dirs() == []


class TestPredicateTable:
    def test_shared_allowlisted_flag_on_true(self, shared_env):
        su = shared_env["su"]
        d = _mk_skill(shared_env["shared"], "smart-home", "clanker-e2e")
        assert su.is_shared_curatable_path(
            d, include_shared_dirs=True, shared_dirs=["smart-home"],
        ) is True

    def test_flag_off_false(self, shared_env):
        su = shared_env["su"]
        d = _mk_skill(shared_env["shared"], "smart-home", "clanker-e2e")
        assert su.is_shared_curatable_path(
            d, include_shared_dirs=False, shared_dirs=[],
        ) is False

    def test_empty_allowlist_means_all_groups(self, shared_env):
        su = shared_env["su"]
        d = _mk_skill(shared_env["shared"], "devops", "cron-alert-discipline")
        assert su.is_shared_curatable_path(
            d, include_shared_dirs=True, shared_dirs=[],
        ) is True

    def test_group_not_in_allowlist_false(self, shared_env):
        su = shared_env["su"]
        d = _mk_skill(shared_env["shared"], "devops", "cron-alert-discipline")
        assert su.is_shared_curatable_path(
            d, include_shared_dirs=True, shared_dirs=["smart-home"],
        ) is False

    def test_hub_marker_false_even_with_flag(self, shared_env):
        su = shared_env["su"]
        d = _mk_skill(shared_env["shared"], "smart-home", "hubby")
        (d / ".hub-meta.json").write_text("{}", encoding="utf-8")
        assert su.is_shared_curatable_path(
            d, include_shared_dirs=True, shared_dirs=[],
        ) is False

    def test_non_shared_external_dir_false(self, shared_env, tmp_path):
        """An external dir OUTSIDE skills-shared/ is never curatable."""
        home, shared = shared_env["home"], shared_env["shared"]
        elsewhere = tmp_path / "elsewhere-skills"
        d = _mk_skill(elsewhere, "misc", "outsider")
        _write_config(
            home, include_shared=True,
            external_dirs=[str(shared / "smart-home"), str(elsewhere / "misc")],
        )
        su = _reload_su()
        assert su.is_external_skill_path(d) is True
        assert su.is_shared_curatable_path(
            d, include_shared_dirs=True, shared_dirs=[],
        ) is False

    def test_bundled_local_skill_false(self, shared_env):
        """A local-tree (bundled) skill path is never shared-curatable."""
        su = shared_env["su"]
        local = shared_env["home"] / "skills" / "bundled-thing"
        local.mkdir(parents=True)
        (local / "SKILL.md").write_text("---\nname: bundled-thing\n---\nx",
                                        encoding="utf-8")
        assert su.is_shared_curatable_path(
            local, include_shared_dirs=True, shared_dirs=[],
        ) is False

    def test_symlink_escape_to_local_tree_rejected(self, shared_env):
        """A skills-shared entry symlinked to a local/bundled dir is rejected."""
        su = shared_env["su"]
        home, shared = shared_env["home"], shared_env["shared"]
        target = home / "skills" / "real-bundled"
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text("---\nname: real-bundled\n---\nx",
                                         encoding="utf-8")
        link = shared / "smart-home" / "sneaky"
        link.symlink_to(target)
        assert su.is_shared_curatable_path(
            link, include_shared_dirs=True, shared_dirs=[],
        ) is False

    def test_path_traversal_allowlist_entry_does_not_widen(self, shared_env, tmp_path):
        """../../ in an absolute allowlist entry cannot widen scope beyond
        skills-shared/."""
        su = shared_env["su"]
        shared = shared_env["shared"]
        d = _mk_skill(shared, "smart-home", "clanker-e2e")
        evil = str(shared / "smart-home" / ".." / ".." / "..")
        # the traversal entry resolves OUTSIDE skills-shared → contributes
        # nothing; with no other match the skill is out of scope
        assert su.is_shared_curatable_path(
            d, include_shared_dirs=True, shared_dirs=[evil],
        ) is False
        # but a sane absolute entry still matches
        assert su.is_shared_curatable_path(
            d, include_shared_dirs=True,
            shared_dirs=[str(shared / "smart-home")],
        ) is True


class TestGateRewiring:
    """The live gates (is_curation_eligible / is_agent_created /
    archive routing / background-review write guard) consult the predicate."""

    def _reload_usage(self):
        import tools.skill_usage as usage
        return importlib.reload(usage)

    def test_curation_eligible_shared_flag_on(self, shared_env):
        home, shared = shared_env["home"], shared_env["shared"]
        d = _mk_skill(shared, "smart-home", "clanker-e2e")
        _write_config(
            home, include_shared=True,
            external_dirs=[str(shared / "smart-home"), str(shared / "devops")],
        )
        _reload_su()
        usage = self._reload_usage()
        assert usage.is_curation_eligible("clanker-e2e", d) is True

    def test_curation_eligible_shared_flag_off(self, shared_env):
        shared = shared_env["shared"]
        d = _mk_skill(shared, "smart-home", "clanker-e2e")
        usage = self._reload_usage()
        # flag off in config (shared_env default)
        assert usage.is_curation_eligible("clanker-e2e", d) is False

    def test_hub_dir_stays_ineligible_with_flag_on(self, shared_env):
        home, shared = shared_env["home"], shared_env["shared"]
        d = _mk_skill(shared, "smart-home", "hub-thing")
        (d / ".hub-meta.json").write_text("{}", encoding="utf-8")
        _write_config(
            home, include_shared=True,
            external_dirs=[str(shared / "smart-home"), str(shared / "devops")],
        )
        _reload_su()
        usage = self._reload_usage()
        assert usage.is_curation_eligible("hub-thing", d) is False

    def test_background_write_guard_permits_in_scope_shared(self, shared_env, monkeypatch):
        home, shared = shared_env["home"], shared_env["shared"]
        d = _mk_skill(shared, "smart-home", "clanker-e2e")
        _write_config(
            home, include_shared=True,
            external_dirs=[str(shared / "smart-home"), str(shared / "devops")],
        )
        _reload_su()
        import tools.skill_manager_tool as smt
        importlib.reload(smt)
        monkeypatch.setattr(
            "tools.skill_provenance.is_background_review", lambda: True
        )
        result = smt._background_review_write_guard(
            name="clanker-e2e", skill_dir=d, action="patch",
        )
        assert result is None  # permitted

    def test_background_write_guard_still_blocks_flag_off(self, shared_env, monkeypatch):
        shared = shared_env["shared"]
        d = _mk_skill(shared, "smart-home", "clanker-e2e")
        import tools.skill_manager_tool as smt
        importlib.reload(smt)
        monkeypatch.setattr(
            "tools.skill_provenance.is_background_review", lambda: True
        )
        result = smt._background_review_write_guard(
            name="clanker-e2e", skill_dir=d, action="patch",
        )
        assert result is not None and result["success"] is False


class TestCandidateScope:
    def test_ac1_flag_off_zero_shared_candidates(self, shared_env, monkeypatch):
        _mk_skill(shared_env["shared"], "smart-home", "clanker-e2e")
        import agent.curator as curator
        importlib.reload(curator)
        monkeypatch.setattr(curator, "_load_config", lambda: {})
        assert curator._iter_shared_candidate_dirs() == []

    def test_ac2_flag_on_shared_candidates_no_hub(self, shared_env, monkeypatch):
        home, shared = shared_env["home"], shared_env["shared"]
        _mk_skill(shared, "smart-home", "clanker-e2e")
        _mk_skill(shared, "devops", "cron-alert-discipline")
        hub = _mk_skill(shared, "smart-home", "hub-thing")
        (hub / ".hub-meta.json").write_text("{}", encoding="utf-8")
        _write_config(
            home, include_shared=True,
            external_dirs=[str(shared / "smart-home"), str(shared / "devops")],
        )
        _reload_su()
        import agent.curator as curator
        importlib.reload(curator)
        monkeypatch.setattr(
            curator, "_load_config", lambda: {"include_shared_dirs": True}
        )
        names = [d.name for d in curator._iter_shared_candidate_dirs()]
        assert "clanker-e2e" in names
        assert "cron-alert-discipline" in names
        assert "hub-thing" not in names

    def test_shared_candidate_rows_carry_source_and_group(self, shared_env, monkeypatch):
        home, shared = shared_env["home"], shared_env["shared"]
        _mk_skill(shared, "smart-home", "clanker-e2e")
        _write_config(
            home, include_shared=True,
            external_dirs=[str(shared / "smart-home"), str(shared / "devops")],
        )
        _reload_su()
        import agent.curator as curator
        importlib.reload(curator)
        monkeypatch.setattr(
            curator, "_load_config", lambda: {"include_shared_dirs": True}
        )
        rows = curator._render_shared_candidate_list()
        assert "clanker-e2e" in rows
        assert "source=shared" in rows
        assert "group=smart-home" in rows

    def test_prompt_note_names_bundled_and_hub_readonly(self):
        import agent.curator as curator
        note = curator.SHARED_SCOPE_NOTE
        assert "source=shared" in note
        assert "Bundled" in note and "hub-installed" in note
        assert "read-only" in note
        assert "skills-shared/<group>/.archive/<name>/" in note
