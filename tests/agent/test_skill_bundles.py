"""Tests for agent/skill_bundles.py — YAML-defined skill bundles."""

import dis
import os
import sys
import threading
from pathlib import Path

import pytest

from agent.skill_bundles import (
    _slugify,
    build_bundle_invocation_message,
    delete_bundle,
    get_bundle,
    get_skill_bundles,
    list_bundles,
    reload_bundles,
    resolve_bundle_command_key,
    resolve_cached_bundle_command_key,
    save_bundle,
    scan_bundles,
)


def _make_bundle_yaml(
    bundles_dir: Path, slug: str, skills: list[str],
    description: str = "", instruction: str = "", name: str | None = None,
) -> Path:
    bundles_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    if name is not None:
        lines.append(f"name: {name}")
    else:
        lines.append(f"name: {slug}")
    if description:
        lines.append(f"description: {description}")
    lines.append("skills:")
    for s in skills:
        lines.append(f"  - {s}")
    if instruction:
        lines.append("instruction: |")
        for ln in instruction.splitlines():
            lines.append(f"  {ln}")
    path = bundles_dir / f"{slug}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_skill(skills_dir: Path, name: str, body: str = "Do the thing.") -> Path:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}\n---\n\n# {name}\n\n{body}\n"
    )
    return skill_dir


@pytest.fixture
def bundles_env(tmp_path, monkeypatch):
    """Isolated bundles dir + skills dir."""
    bundles_dir = tmp_path / "skill-bundles"
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    # Patch SKILLS_DIR so skill loading hits our temp tree.
    import tools.skills_tool as skills_tool_module
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", skills_dir)
    # Reset module-level cache between tests.
    import agent.skill_bundles as mod
    mod._bundles_cache = {}
    mod._bundles_cache_mtime = None
    mod._bundles_cache_dir = None
    return bundles_dir, skills_dir


class TestSlugify:
    def test_basic(self):
        assert _slugify("Backend Dev") == "backend-dev"




    def test_empty(self):
        assert _slugify("") == ""
        assert _slugify("!!!") == ""


class TestScanBundles:

    def test_finds_bundle(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "backend", ["skill-a", "skill-b"])
        result = scan_bundles()
        assert "/backend" in result
        assert result["/backend"]["name"] == "backend"
        assert result["/backend"]["skills"] == ["skill-a", "skill-b"]

    def test_skips_invalid_yaml(self, bundles_env):
        bundles_dir, _ = bundles_env
        bundles_dir.mkdir(parents=True)
        (bundles_dir / "broken.yaml").write_text("{not: valid yaml: [")
        _make_bundle_yaml(bundles_dir, "good", ["skill-a"])
        result = scan_bundles()
        assert "/good" in result
        assert "/broken" not in result





class TestGetSkillBundles:
    def test_returns_cache(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "a", ["s1"])
        first = get_skill_bundles()
        # Second call should hit cache (no rescan unless mtime changed).
        second = get_skill_bundles()
        assert first is second or first == second

    def test_rescans_on_change(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "a", ["s1"])
        assert "/a" in get_skill_bundles()
        # Add a second bundle and bump mtime.
        import time as _t
        _t.sleep(0.05)  # ensure mtime granularity is exceeded
        _make_bundle_yaml(bundles_dir, "b", ["s2"])
        os.utime(bundles_dir, None)
        result = get_skill_bundles()
        assert "/a" in result
        assert "/b" in result


class TestResolveBundleCommandKey:
    def test_exact_match(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "my-bundle", ["s1"])
        scan_bundles()
        assert resolve_bundle_command_key("my-bundle") == "/my-bundle"


    def test_unknown(self, bundles_env):
        scan_bundles()
        assert resolve_bundle_command_key("missing") is None

    def test_empty(self, bundles_env):
        assert resolve_bundle_command_key("") is None

    def test_cached_resolver_rejects_another_profile(self, bundles_env, monkeypatch):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "only-a", ["s1"])
        scan_bundles()
        import agent.skill_bundles as bundle_mod
        from hermes_constants import get_hermes_home, hermes_home_key

        current_home_key = hermes_home_key(get_hermes_home())
        assert resolve_cached_bundle_command_key(
            "only-a", current_home_key, bundle_mod._resolve_bundle_platform()
        ) == "/only-a"

        other_home_key = hermes_home_key(bundles_dir.parent / "other-home")
        assert resolve_cached_bundle_command_key(
            "only-a", other_home_key, bundle_mod._resolve_bundle_platform()
        ) is None

    def test_prepared_identity_lookup_is_filesystem_free(self, monkeypatch):
        import agent.skill_bundles as bundle_mod
        from hermes_constants import hermes_home_key

        home = Path("/profiles") / "telegram" / ".." / "telegram"
        home_key = hermes_home_key(home)
        monkeypatch.setattr(
            bundle_mod,
            "_bundle_cache_snapshots",
            {
                (home_key, "telegram"): {
                    "/only-a": {"name": "only-a"},
                }
            },
            raising=False,
        )
        original_exists = Path.exists
        original_read_text = Path.read_text
        monkeypatch.setattr(
            Path,
            "exists",
            lambda _path: (_ for _ in ()).throw(
                AssertionError("cache-only lookup touched the filesystem")
            ),
        )
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda _path, *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("cache-only lookup read the filesystem")
            ),
        )
        try:
            assert bundle_mod.resolve_cached_bundle_command_key(
                "only-a", home_key, " TELEGRAM "
            ) == "/only-a"
        finally:
            monkeypatch.setattr(Path, "exists", original_exists)
            monkeypatch.setattr(Path, "read_text", original_read_text)

    def test_scan_and_lookup_share_canonical_home_and_platform_identity(
        self, bundles_env, tmp_path, monkeypatch
    ):
        bundles_dir, _ = bundles_env
        import agent.skill_bundles as bundle_mod
        from hermes_constants import hermes_home_key

        profile_home = tmp_path / "profile"
        (profile_home / "nested").mkdir(parents=True)
        equivalent_home = profile_home / "nested" / ".."
        _make_bundle_yaml(bundles_dir, "canonical-bundle", ["skill"])

        monkeypatch.setattr(bundle_mod, "_bundle_cache_snapshots", {})
        monkeypatch.setattr(
            bundle_mod, "_resolve_bundle_home", lambda: equivalent_home
        )
        monkeypatch.setattr(
            bundle_mod, "_resolve_bundle_platform", lambda: "telegram"
        )

        bundle_mod.scan_bundles()

        assert bundle_mod.resolve_cached_bundle_command_key(
            "canonical-bundle", hermes_home_key(profile_home), " TELEGRAM "
        ) == "/canonical-bundle"

    def test_interleaved_scans_publish_complete_profile_snapshots(self, monkeypatch):
        import agent.skill_bundles as bundle_mod

        first_ready = threading.Event()
        allow_first = threading.Event()
        identity = threading.local()
        bundle_a = Path("/profiles/a/a.yaml")
        bundle_b = Path("/profiles/b/b.yaml")

        def iter_files():
            if identity.name == "a":
                first_ready.set()
                assert allow_first.wait(2)
                return [bundle_a]
            return [bundle_b]

        def load_bundle(path):
            name = "bundle-a" if path == bundle_a else "bundle-b"
            return {"name": name, "slug": name, "skills": ["skill"]}

        monkeypatch.setattr(bundle_mod, "_bundle_cache_snapshots", {}, raising=False)
        monkeypatch.setattr(bundle_mod, "_bundles_cache", {})
        monkeypatch.setattr(bundle_mod, "_bundles_cache_mtime", None)
        monkeypatch.setattr(bundle_mod, "_bundles_cache_dir", None)
        monkeypatch.setattr(bundle_mod, "_iter_bundle_files", iter_files)
        monkeypatch.setattr(bundle_mod, "_load_bundle_file", load_bundle)
        monkeypatch.setattr(bundle_mod, "_max_mtime", lambda _files: 1.0)
        monkeypatch.setattr(bundle_mod, "_bundles_dir", lambda: Path("/profiles/") / identity.name)
        monkeypatch.setattr(bundle_mod, "_resolve_bundle_platform", lambda: "telegram", raising=False)
        monkeypatch.setattr(bundle_mod, "_resolve_bundle_home", lambda: "/profiles/" + identity.name, raising=False)

        failures = []
        probe_results = []

        def run(name):
            identity.name = name
            try:
                bundle_mod.scan_bundles()
                if name == "b":
                    probe_results.append(
                        bundle_mod.resolve_cached_bundle_command_key(
                            "bundle-b", "/profiles/b", "telegram"
                        )
                    )
            except BaseException as exc:  # pragma: no cover - surfaced below
                failures.append(exc)

        thread_a = threading.Thread(target=run, args=("a",))
        thread_b = threading.Thread(target=run, args=("b",))
        thread_a.start()
        assert first_ready.wait(2)
        thread_b.start()
        thread_b.join(2)
        allow_first.set()
        thread_a.join(2)

        assert not failures
        assert not thread_a.is_alive() and not thread_b.is_alive()
        assert probe_results == ["/bundle-b"]
        assert set(bundle_mod._bundles_cache) in ({"/bundle-a"}, {"/bundle-b"})

    def test_get_returns_its_selected_snapshot_when_compat_alias_changes(
        self, monkeypatch
    ):
        """A caller returns its keyed snapshot even if another caller wins the alias."""
        import agent.skill_bundles as bundle_mod
        from hermes_constants import hermes_home_key

        identity = threading.local()
        home_a = hermes_home_key("/profiles/a")
        home_b = hermes_home_key("/profiles/b")
        dir_a = hermes_home_key("/profiles/a/skill-bundles")
        dir_b = hermes_home_key("/profiles/b/skill-bundles")
        bundles_a = {"/bundle-a": {"name": "bundle-a", "skills": ["skill"]}}
        bundles_b = {"/bundle-b": {"name": "bundle-b", "skills": ["skill"]}}
        monkeypatch.setattr(
            bundle_mod,
            "_bundle_cache_snapshots",
            {
                (home_a, "telegram"): bundle_mod._BundleCacheSnapshot(
                    bundles=bundles_a,
                    home_key=home_a,
                    platform="telegram",
                    directory_key=dir_a,
                    mtime=1.0,
                ),
                (home_b, "telegram"): bundle_mod._BundleCacheSnapshot(
                    bundles=bundles_b,
                    home_key=home_b,
                    platform="telegram",
                    directory_key=dir_b,
                    mtime=1.0,
                ),
            },
        )
        monkeypatch.setattr(bundle_mod, "_bundles_cache", {"/seed": {}})
        monkeypatch.setattr(
            bundle_mod, "_resolve_bundle_home", lambda: f"/profiles/{identity.name}"
        )
        monkeypatch.setattr(
            bundle_mod, "_bundles_dir", lambda: Path(f"/profiles/{identity.name}/skill-bundles")
        )
        monkeypatch.setattr(bundle_mod, "_resolve_bundle_platform", lambda: "telegram")
        monkeypatch.setattr(bundle_mod, "_iter_bundle_files", lambda: [])
        monkeypatch.setattr(bundle_mod, "_max_mtime", lambda _files: 1.0)

        final_line = max(line for _, line in dis.findlinestarts(bundle_mod.get_skill_bundles.__code__))
        a_at_return = threading.Event()
        release_a = threading.Event()
        blocked = False
        failures = []
        results = {}

        def trace(frame, event, _arg):
            nonlocal blocked
            if (
                not blocked
                and frame.f_code is bundle_mod.get_skill_bundles.__code__
                and event == "line"
                and frame.f_lineno == final_line
            ):
                blocked = True
                a_at_return.set()
                release_a.wait(2)
            return trace

        def run_a():
            identity.name = "a"
            sys.settrace(trace)
            try:
                results["a"] = bundle_mod.get_skill_bundles()
            except BaseException as exc:  # pragma: no cover - surfaced below
                failures.append(exc)
            finally:
                sys.settrace(None)

        thread_a = threading.Thread(target=run_a)
        thread_a.start()
        assert a_at_return.wait(2)

        identity.name = "b"
        results["b"] = bundle_mod.get_skill_bundles()
        release_a.set()
        thread_a.join(2)

        assert not failures
        assert not thread_a.is_alive()
        assert set(results["a"]) == {"/bundle-a"}
        assert set(results["b"]) == {"/bundle-b"}

    def test_cached_snapshot_is_deeply_immutable(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "immutable", ["skill-a"])

        import agent.skill_bundles as bundle_mod
        from hermes_constants import get_hermes_home, hermes_home_key

        bundle_mod.scan_bundles()
        cached = bundle_mod.get_cached_skill_bundles(
            hermes_home_key(get_hermes_home()),
            bundle_mod._resolve_bundle_platform(),
        )

        assert cached is not None
        with pytest.raises(TypeError):
            cached["/immutable"]["skills"][0] = "mutated"
        assert cached["/immutable"]["skills"] == ("skill-a",)


class TestBuildBundleInvocationMessage:
    def test_loads_all_skills(self, bundles_env):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a", body="Skill A content.")
        _make_skill(skills_dir, "skill-b", body="Skill B content.")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-b"])
        scan_bundles()

        result = build_bundle_invocation_message("/combo")
        assert result is not None
        msg, loaded, missing = result
        assert set(loaded) == {"skill-a", "skill-b"}
        assert missing == []
        assert "Skill A content." in msg
        assert "Skill B content." in msg
        assert "combo" in msg

    def test_forwards_task_id_to_each_loaded_skill(self, bundles_env, monkeypatch):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a")
        _make_skill(skills_dir, "skill-b")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-b"])
        scan_bundles()
        calls = []
        monkeypatch.setattr(
            "tools.skill_usage.bump_use",
            lambda skill_name, **kwargs: calls.append((skill_name, kwargs)),
        )

        result = build_bundle_invocation_message(
            "/combo",
            task_id="task-bundle",
        )

        assert result is not None
        assert calls == [
            ("skill-a", {"task_id": "task-bundle"}),
            ("skill-b", {"task_id": "task-bundle"}),
        ]

    def test_skips_missing_skills(self, bundles_env):
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-ghost"])
        scan_bundles()

        result = build_bundle_invocation_message("/combo")
        assert result is not None
        msg, loaded, missing = result
        assert loaded == ["skill-a"]
        assert missing == ["skill-ghost"]
        assert "skill-ghost" in msg  # called out in header

    def test_skips_platform_disabled_skills(self, bundles_env, monkeypatch):
        """A skill disabled for the invoking platform must not be injected
        via a bundle (mirrors the stacked-skill gate, #58888)."""
        bundles_dir, skills_dir = bundles_env
        _make_skill(skills_dir, "skill-a", body="Skill A content.")
        _make_skill(skills_dir, "skill-b", body="SECRET DISABLED CONTENT.")
        _make_bundle_yaml(bundles_dir, "combo", ["skill-a", "skill-b"])
        scan_bundles()

        def _fake_disabled(platform=None):
            return {"skill-b"} if platform == "telegram" else set()

        import agent.skill_utils as su_module
        monkeypatch.setattr(
            su_module, "get_disabled_skill_names", _fake_disabled
        )

        result = build_bundle_invocation_message("/combo", platform="telegram")
        assert result is not None
        msg, loaded, missing = result
        assert loaded == ["skill-a"]
        assert "SECRET DISABLED CONTENT." not in msg
        assert "skill-b" in msg  # called out in the disabled-skipped header line
        assert "disabled" in msg.lower()

        # Positive control: without the platform the skill loads normally.
        result2 = build_bundle_invocation_message("/combo")
        assert result2 is not None
        msg2, loaded2, _ = result2
        assert set(loaded2) == {"skill-a", "skill-b"}
        assert "SECRET DISABLED CONTENT." in msg2








class TestSaveAndDeleteBundle:
    def test_save_creates_file(self, bundles_env):
        bundles_dir, _ = bundles_env
        path = save_bundle("test-bundle", ["s1", "s2"], description="d", instruction="i")
        assert path.exists()
        assert path.parent == bundles_dir
        content = path.read_text()
        assert "test-bundle" in content
        assert "s1" in content
        assert "s2" in content
        assert "description: d" in content


    def test_save_overwrites_with_force(self, bundles_env):
        save_bundle("dup", ["s1"])
        save_bundle("dup", ["s2"], overwrite=True)
        info = get_bundle("dup")
        assert info is not None
        assert info["skills"] == ["s2"]



    def test_delete_removes_file(self, bundles_env):
        bundles_dir, _ = bundles_env
        save_bundle("doomed", ["s1"])
        assert get_bundle("doomed") is not None
        delete_bundle("doomed")
        assert get_bundle("doomed") is None



class TestReloadBundles:
    def test_reports_added_and_removed(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "old", ["s1"])
        scan_bundles()  # populate cache with {old}

        # Mutate the disk WITHOUT going through save/delete helpers (which
        # would refresh the cache mid-way). reload_bundles() diffs the
        # in-memory cache against the freshly-scanned disk state.
        (bundles_dir / "old.yaml").unlink()
        _make_bundle_yaml(bundles_dir, "new", ["s2"])

        diff = reload_bundles()
        added_names = {e["name"] for e in diff["added"]}
        removed_names = {e["name"] for e in diff["removed"]}
        assert "new" in added_names
        assert "old" in removed_names
        assert diff["total"] == 1


class TestListBundles:
    def test_sorted_by_slug(self, bundles_env):
        bundles_dir, _ = bundles_env
        _make_bundle_yaml(bundles_dir, "zebra", ["s1"])
        _make_bundle_yaml(bundles_dir, "apple", ["s2"])
        _make_bundle_yaml(bundles_dir, "mango", ["s3"])
        scan_bundles()
        info_list = list_bundles()
        slugs = [b["slug"] for b in info_list]
        assert slugs == sorted(slugs)
