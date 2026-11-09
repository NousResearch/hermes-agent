"""The skills index port as a plugin kind, the local backend's pin, and its level refusals.

``resolve_skills_index(name)`` picks a backend by explicit name (no caller is wired to it yet):
``"local"`` is today's per-profile ``skills/`` directory; another name loads a provider through
real discovery (``$HERMES_HOME/plugins/<name>/``), and a missing one is an error, never a
fallback to the local index. The local backend serves only the local tenant and its own profile,
and stores the agent level only.
"""
import textwrap
from pathlib import Path

import pytest

from agent.runtime_scope import bind_profile_scope, bind_state
from hermes_scope import Group, Principal, Scope, push_state
from hermes_skills_port import (
    SKILL_LEVELS, SkillsIndexProvider, SkillsLevelRefused, StoragePinRefused, resolution_order,
    resolve_skills_index,
)
from hermes_skills_port_local import LocalSkillsIndexProvider

_PLUGIN = '''
from hermes_skills_port import SkillsIndexProvider


class Probe(SkillsIndexProvider):
    @property
    def name(self):
        return "probe"

    def open(self, skills_dir=None):
        raise NotImplementedError

    def assert_isolating(self):
        raise RuntimeError("probe role bypasses row security")


def register(ctx):
    ctx.register_skills_index(Probe())
'''


def _skill_md(name, description, body="Instructions."):
    return f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"


@pytest.fixture
def index(tmp_path):
    return LocalSkillsIndexProvider().open(tmp_path / "skills").index


# --- the plugin kind


def test_local_names_the_local_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("local", ""):
        provider = resolve_skills_index(name)
        assert isinstance(provider, LocalSkillsIndexProvider) and provider.levels == frozenset({"agent"})
        provider.assert_isolating()


def test_a_provider_that_does_not_check_isolation_cannot_be_built():
    class Unchecked(SkillsIndexProvider):
        @property
        def name(self):
            return "unchecked"

        def open(self, skills_dir=None):
            raise NotImplementedError

    with pytest.raises(TypeError, match="assert_isolating"):
        Unchecked()


def test_a_named_provider_loads_through_discovery_and_runs_its_isolation_check(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "probe"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN))
    provider = resolve_skills_index("probe")
    assert isinstance(provider, SkillsIndexProvider) and provider.name == "probe"
    with pytest.raises(RuntimeError, match="bypasses row security"):
        provider.assert_isolating()


def test_an_unavailable_named_provider_is_an_error_not_the_local_index(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin = tmp_path / "plugins" / "offline"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(textwrap.dedent(_PLUGIN).replace(
        "    def assert_isolating(self):", "    def is_available(self):\n        return False\n\n    def assert_isolating(self):"))
    with pytest.raises(LookupError, match="not available"):
        resolve_skills_index("offline")


def test_a_missing_named_provider_is_an_error_not_the_local_index(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with pytest.raises(LookupError, match="postgres"):
        resolve_skills_index("postgres")


# --- the local backend is today's store, at today's paths


def test_the_local_backend_reads_what_today_puts_there(index, tmp_path):
    (tmp_path / "skills" / "handmade").mkdir(parents=True)
    (tmp_path / "skills" / "handmade" / "SKILL.md").write_text(
        _skill_md("handmade", "Written before the port existed."), encoding="utf-8")
    with bind_profile_scope(tmp_path):
        row = index.get("handmade")
        assert row["name"] == "handmade" and row["load_name"] == "handmade"
        assert row["description"] == "Written before the port existed."
        assert row["category"] is None and row["level"] == "agent" and row["status"] == "unique"
        assert "before the port" in index.read_body("handmade")


def test_the_local_backend_writes_today_paths(index, tmp_path):
    with bind_profile_scope(tmp_path):
        row = index.put("fresh", _skill_md("fresh", "A new skill."), category="mlops")
        assert row["category"] == "mlops"
    skill_md = tmp_path / "skills" / "mlops" / "fresh" / "SKILL.md"
    assert skill_md.read_text(encoding="utf-8") == _skill_md("fresh", "A new skill.")
    # today's scanner sees it: the same file the prompt builder's scan walks
    from agent.skill_utils import iter_skill_index_files
    assert skill_md in list(iter_skill_index_files(tmp_path / "skills", "SKILL.md"))


def test_a_replace_lands_where_the_skill_lives(index, tmp_path):
    with bind_profile_scope(tmp_path):
        index.put("fresh", _skill_md("fresh", "First."), category="mlops")
        index.put("fresh", _skill_md("fresh", "Second."))
    assert (tmp_path / "skills" / "mlops" / "fresh" / "SKILL.md").read_text(
        encoding="utf-8") == _skill_md("fresh", "Second.")


def test_a_name_resolves_by_its_declared_name_its_directory_and_its_path(index, tmp_path):
    (tmp_path / "skills" / "renamed").mkdir(parents=True)
    (tmp_path / "skills" / "renamed" / "SKILL.md").write_text(
        _skill_md("display-name", "Declared in frontmatter."), encoding="utf-8")
    with bind_profile_scope(tmp_path):
        for name in ("display-name", "renamed"):
            assert index.get(name)["load_name"] == "display-name"
        assert index.get("missing") is None


def test_a_name_two_skills_share_resolves_only_by_its_exact_path(index, tmp_path):
    for category in ("one", "two"):
        skill_dir = tmp_path / "skills" / category / "alpha"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            _skill_md("alpha", f"The {category} copy."), encoding="utf-8")
    with bind_profile_scope(tmp_path):
        listed = {r["load_name"]: r["status"] for r in index.list()}
        assert listed == {"one/alpha": "ambiguous", "two/alpha": "ambiguous"}
        assert index.get("alpha") is None  # two skills share it: no guess
        assert index.get("one/alpha")["description"] == "The one copy."
        assert index.read_body("alpha") is None
        with pytest.raises(ValueError, match="Ambiguous skill name"):
            index.put("alpha", _skill_md("alpha", "A third copy."))  # never guess which to write


def test_supporting_files_live_behind_the_body_operation(index, tmp_path):
    with bind_profile_scope(tmp_path):
        index.put("alpha", _skill_md("alpha", "Holds files."))
        index.put_file("alpha", "references/notes.md", "notes")
        assert index.read_body("alpha", "references/notes.md") == "notes"
        assert index.read_body("alpha", "scripts/run.py") is None  # not held
        assert index.delete_file("alpha", "references/notes.md") is True
        assert index.delete_file("alpha", "references/notes.md") is False
        assert index.read_body("alpha", "../escape.md") is None  # an escape reads like a missing file
        with pytest.raises(ValueError, match="traversal"):
            index.put_file("alpha", "../escape.md", "no")
        with pytest.raises(LookupError):
            index.put_file("missing", "references/notes.md", "no")


def test_delete_removes_the_skill_and_its_emptied_category(index, tmp_path):
    with bind_profile_scope(tmp_path):
        index.put("beta", _skill_md("beta", "Bye."), category="ops")
        assert index.delete("beta") is True
        assert index.delete("beta") is False  # already gone, like a foreign skill
    assert not (tmp_path / "skills" / "ops").exists()


def test_the_index_refuses_a_handle_that_is_not_today_directory(tmp_path):
    with pytest.raises(StoragePinRefused, match="absolute"):
        LocalSkillsIndexProvider().open(Path("skills"))
    elsewhere = tmp_path / "elsewhere" / "skills"
    elsewhere.mkdir(parents=True)
    link = tmp_path / "linked-skills"
    link.symlink_to(elsewhere)
    with pytest.raises(StoragePinRefused, match="link into"):
        LocalSkillsIndexProvider().open(link)


# --- the pin


@pytest.mark.parametrize("pin", [
    # A hosted tenant naming this very path as its agent is still not this local index's.
    lambda home: Scope("acme", str(home), Principal("u"), owner=Group("support"), can_administer=False),
    lambda home: Scope("local", str(home / "elsewhere"), Principal("owner")),
], ids=["another_tenant", "another_profile"])
def test_the_local_backend_refuses_any_pin_but_its_own_profile(index, tmp_path, pin):
    scope = pin(tmp_path)
    with bind_state(scope) if not scope.is_local else push_state(scope=scope):
        for call in (lambda: index.list(), lambda: index.get("alpha"),
                     lambda: index.read_body("alpha"),
                     lambda: index.put("alpha", _skill_md("alpha", "x")),
                     lambda: index.delete("alpha")):
            with pytest.raises(StoragePinRefused):
                call()
    assert not (tmp_path / "skills").exists() or not any((tmp_path / "skills").iterdir())


def test_with_no_scope_bound_the_pin_is_the_ambient_profile(index, tmp_path, monkeypatch):
    (tmp_path / "skills" / "handmade").mkdir(parents=True)
    (tmp_path / "skills" / "handmade" / "SKILL.md").write_text(
        _skill_md("handmade", "Made today."), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert index.get("handmade") is not None  # a local caller with no scope, as today
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))
    with pytest.raises(StoragePinRefused):
        index.get("handmade")


# --- levels


def test_a_local_index_stores_the_agent_level_only(index, tmp_path):
    with bind_profile_scope(tmp_path):
        index.put("alpha", _skill_md("alpha", "The agent's own."))
        for write in (lambda: index.put("fleet", _skill_md("fleet", "x"), level="owner"),
                      lambda: index.put("org", _skill_md("org", "x"), level="tenant"),
                      lambda: index.delete("alpha", level="owner"),
                      lambda: index.put_file("alpha", "references/x.md", "x", level="tenant"),
                      lambda: index.delete_file("alpha", "references/x.md", level="tenant")):
            with pytest.raises(SkillsLevelRefused):
                write()
        assert index.get("alpha") is not None  # no refused write changed anything
        assert index.list() and not (tmp_path / "skills" / "fleet").exists()


def test_a_level_no_skills_index_stores_is_refused(index, tmp_path):
    with bind_profile_scope(tmp_path):
        with pytest.raises(SkillsLevelRefused):
            index.put("alpha", _skill_md("alpha", "x"), level="participant")


# --- the one resolution policy


def test_reads_resolve_the_agent_then_its_owner_then_the_tenant():
    scope = Scope("acme", "agent-1", Principal("member"), owner=Group("support"), can_administer=False)
    assert resolution_order(scope) == (("agent", None), ("owner", Group("support")), ("tenant", None))
    assert SKILL_LEVELS == frozenset({"agent", "owner", "tenant"})
