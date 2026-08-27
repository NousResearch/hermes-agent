"""Tests for agent/skill_bundles.py — YAML-defined skill bundles."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from agent.skill_bundles import (
    _slugify,
    build_bundle_invocation_message,
    delete_bundle,
    get_bundle,
    get_skill_bundles,
    list_bundles,
    reload_bundles,
    resolve_bundle_command_key,
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


def _strict_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    manifest_path = root / "persona.yaml"
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path == manifest_path or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _strict_instruction(pack: dict) -> str:
    lines = [
        f"Persona: {pack['persona_name']}",
        f"Persona mission: {pack['mission']}",
        f"Voice and behaviour: {pack['voice']}",
        "",
        "Member roles:",
    ]
    for member in pack["member_skills"]:
        lines.append(f"- {member['skill']}: {member['role']}")
    lines.extend(["", "Composition rules:"])
    for index, rule in enumerate(pack["composition_rules"], start=1):
        lines.append(f"{index}. {rule}")
    lines.extend(["", "Handoffs:"])
    for name, trigger in pack["handoffs"].items():
        lines.append(f"- {name}: {trigger}")
    lines.extend(
        [
            "",
            "Runtime contract:",
            "- Route work to the smallest relevant member set.",
            "- Apply composition rules before conflicting member guidance.",
            "- Follow declared handoffs and stop rather than inventing an undeclared one.",
            "- Never continue if a required member is missing, disabled, or digest-mismatched.",
        ]
    )
    return "\n".join(lines).strip()


def _make_strict_persona_export(
    bundles_dir: Path,
    hermes_home: Path,
    *,
    include_activation: bool = True,
    first_body: str = "Skill A strict content.",
) -> tuple[Path, Path]:
    root = bundles_dir / "demo-persona"
    first = _make_skill(root / "skills", "skill-a", first_body)
    second = _make_skill(root / "skills", "skill-b", "Skill B strict content.")
    pack = {
        "pack_version": "1",
        "kind": "persona",
        "name": "demo-persona",
        "description": "Strict persona fixture.",
        "persona_name": "Demo Persona",
        "mission": "Produce grounded demo work.",
        "voice": "Direct and precise.",
        "member_skills": [
            {"skill": "skill-a", "role": "Research facts."},
            {"skill": "skill-b", "role": "Assemble the answer."},
        ],
        "composition_rules": ["Research before assembly."],
        "handoffs": {"research_to_answer": "after facts are verified"},
    }
    pack_path = root / "PACK.json"
    pack_path.write_text(json.dumps(pack, indent=2))
    instruction = _strict_instruction(pack)
    content_digest = "a" * 64
    activation_digest = "b" * 64
    manifest = {
        "schema_version": 1,
        "kind": "persona",
        "strict": True,
        "name": "demo-persona",
        "description": "Strict persona fixture.",
        "pack_digest": hashlib.sha256(pack_path.read_bytes()).hexdigest(),
        "persona_tree_digest": _strict_tree_digest(root),
        "approvals": {
            "content": {
                "purpose": "persona-content",
                "bundle_digest": content_digest,
            },
            "activation": {
                "purpose": "persona-activation",
                "bundle_digest": activation_digest,
            },
        },
        "members": [
            {
                "skill": "skill-a",
                "role": "Research facts.",
                "digest": hashlib.sha256((first / "SKILL.md").read_bytes()).hexdigest(),
            },
            {
                "skill": "skill-b",
                "role": "Assemble the answer.",
                "digest": hashlib.sha256((second / "SKILL.md").read_bytes()).hexdigest(),
            },
        ],
        "instruction": instruction,
        "instruction_digest": hashlib.sha256(instruction.encode()).hexdigest(),
    }
    (root / "persona.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
    )

    approved = {
        content_digest: {
            "purpose": "persona-content",
            "persona": "demo-persona",
            "pack_digest": manifest["pack_digest"],
            "persona_tree_digest": manifest["persona_tree_digest"],
        }
    }
    if include_activation:
        approved[activation_digest] = {
            "purpose": "persona-activation",
            "persona": "demo-persona",
            "pack_digest": manifest["pack_digest"],
            "persona_tree_digest": manifest["persona_tree_digest"],
        }
    registry = hermes_home / "approvals" / "persona-bundles.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"schema_version": 1, "approved": approved}))
    registry.chmod(0o600)
    return root, registry


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








class TestStrictPersonaBundles:
    def test_discovers_canonical_nested_export_and_invokes_all_members(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _make_strict_persona_export(bundles_dir, hermes_home)

        assert "/demo-persona" in scan_bundles()
        result = build_bundle_invocation_message("/demo-persona")
        assert result is not None
        message, loaded, missing = result
        assert loaded == ["skill-a", "skill-b"]
        assert missing == []
        assert "Produce grounded demo work." in message
        assert "Research before assembly." in message
        assert "Skill A strict content." in message
        assert "Skill B strict content." in message

    def test_requires_separate_activation_authority(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _make_strict_persona_export(
            bundles_dir, hermes_home, include_activation=False
        )
        assert "/demo-persona" not in scan_bundles()

    def test_revalidates_members_on_every_invocation(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        root, _ = _make_strict_persona_export(bundles_dir, hermes_home)
        assert "/demo-persona" in scan_bundles()
        skill_md = root / "skills" / "skill-a" / "SKILL.md"
        skill_md.write_bytes(skill_md.read_bytes() + b"\ntampered\n")
        assert build_bundle_invocation_message("/demo-persona") is None

    def test_required_disabled_member_rejects_complete_persona(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _make_strict_persona_export(bundles_dir, hermes_home)
        assert "/demo-persona" in scan_bundles()
        monkeypatch.setattr(
            "agent.skill_utils.get_disabled_skill_names",
            lambda platform=None: {"skill-b"},
        )
        assert build_bundle_invocation_message("/demo-persona") is None

    def test_rejects_unsafe_registry_and_symlinked_member(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        root, registry = _make_strict_persona_export(bundles_dir, hermes_home)
        registry.chmod(0o660)
        assert "/demo-persona" not in scan_bundles()

        registry.chmod(0o600)
        skill_md = root / "skills" / "skill-a" / "SKILL.md"
        outside = tmp_path / "outside.md"
        outside.write_text("replacement")
        skill_md.unlink()
        skill_md.symlink_to(outside)
        assert "/demo-persona" not in scan_bundles()

    def test_non_utf8_manifest_does_not_crash_ordinary_bundle_discovery(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        root, _ = _make_strict_persona_export(bundles_dir, hermes_home)
        _make_bundle_yaml(bundles_dir, "ordinary", ["skill-a"])
        (root / "persona.yaml").write_bytes(b"\xff\xfe invalid manifest")

        found = scan_bundles()
        assert "/demo-persona" not in found
        assert "/ordinary" in found

    def test_non_utf8_pack_does_not_crash_ordinary_bundle_discovery(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        root, _ = _make_strict_persona_export(bundles_dir, hermes_home)
        _make_bundle_yaml(bundles_dir, "ordinary", ["skill-a"])

        pack_path = root / "PACK.json"
        pack_path.write_bytes(b"\xff\xfe\x00 invalid utf8 pack")
        manifest_path = root / "persona.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["pack_digest"] = hashlib.sha256(pack_path.read_bytes()).hexdigest()
        manifest["persona_tree_digest"] = _strict_tree_digest(root)
        manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

        found = scan_bundles()
        assert "/demo-persona" not in found
        assert "/ordinary" in found

    def test_non_utf8_registry_does_not_crash_ordinary_bundle_discovery(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _, registry = _make_strict_persona_export(bundles_dir, hermes_home)
        _make_bundle_yaml(bundles_dir, "ordinary", ["skill-a"])
        registry.write_bytes(b"\xff\xfe invalid registry")

        found = scan_bundles()
        assert "/demo-persona" not in found
        assert "/ordinary" in found

    def test_unreadable_strict_file_does_not_crash_ordinary_bundle_discovery(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _make_strict_persona_export(bundles_dir, hermes_home)
        _make_bundle_yaml(bundles_dir, "ordinary", ["skill-a"])

        original_read_bytes = Path.read_bytes

        def denied_read_bytes(path):
            if path.name == "PACK.json":
                raise PermissionError("denied")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", denied_read_bytes)
        found = scan_bundles()
        assert "/demo-persona" not in found
        assert "/ordinary" in found

    def test_disabled_lookup_failure_rejects_strict_invocation(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _make_strict_persona_export(bundles_dir, hermes_home)
        assert "/demo-persona" in scan_bundles()

        def unavailable(*_args, **_kwargs):
            raise OSError("config unavailable")

        monkeypatch.setattr(
            "agent.skill_utils.get_disabled_skill_names",
            unavailable,
        )
        assert build_bundle_invocation_message("/demo-persona") is None

    def test_invocation_uses_the_exact_member_bytes_that_passed_digest_check(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _make_strict_persona_export(bundles_dir, hermes_home)
        assert "/demo-persona" in scan_bundles()

        original_read_text = Path.read_text

        def swapped_read_text(path, *args, **kwargs):
            if path.name == "SKILL.md" and path.parent.name == "skill-a":
                return "UNVERIFIED SWAPPED CONTENT"
            return original_read_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", swapped_read_text)
        result = build_bundle_invocation_message("/demo-persona")
        assert result is not None
        message, _, _ = result
        assert "Skill A strict content." in message
        assert "UNVERIFIED SWAPPED CONTENT" not in message

    def test_rejects_registry_parent_symlink_escape(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _, registry = _make_strict_persona_export(bundles_dir, hermes_home)
        authority = registry.read_bytes()
        registry.unlink()
        registry.parent.rmdir()
        external = tmp_path / "external-approvals"
        external.mkdir()
        external_registry = external / "persona-bundles.json"
        external_registry.write_bytes(authority)
        external_registry.chmod(0o600)
        registry.parent.symlink_to(external, target_is_directory=True)

        assert "/demo-persona" not in scan_bundles()

    def test_rejects_complete_persona_over_prompt_limit(
        self, bundles_env, monkeypatch, tmp_path
    ):
        bundles_dir, _ = bundles_env
        hermes_home = tmp_path / "home"
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        _make_strict_persona_export(
            bundles_dir,
            hermes_home,
            first_body="x" * (97 * 1024),
        )
        assert "/demo-persona" in scan_bundles()
        assert build_bundle_invocation_message("/demo-persona") is None


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
