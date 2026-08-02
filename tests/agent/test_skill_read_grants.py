from __future__ import annotations

import json
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agent import skill_utils
from tools import skills_tool


@pytest.fixture
def disabled_skills_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    skills_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)

    config_path = home / "config.yaml"
    config_path.write_text(
        "skills:\n  disabled:\n    - alpha\n    - beta\n",
        encoding="utf-8",
    )
    for name in ("alpha", "beta", "enabled"):
        skill_dir = skills_dir / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            (
                f"---\nname: {name}\ndescription: {name} skill\n---\n\n"
                f"{name} body\n"
                "prompt-canary credential-canary user-data-canary\n"
            ),
            encoding="utf-8",
        )
        references = skill_dir / "references"
        references.mkdir()
        (references / "details.md").write_text(
            f"{name} details\n", encoding="utf-8"
        )

    skill_utils._raw_config_cache_clear()
    yield home, config_path
    skill_utils._raw_config_cache_clear()


def _view(name: str) -> dict:
    return json.loads(skills_tool.skill_view(name, preprocess=False))


def _write_alias_test_skill(
    root: Path,
    relative_selector: str,
    canonical: str,
    body: str,
    *,
    legacy_flat: bool,
) -> Path:
    target = root / relative_selector
    skill_md = target.with_suffix(".md") if legacy_flat else target / "SKILL.md"
    skill_md.parent.mkdir(parents=True, exist_ok=True)
    skill_md.write_text(
        (
            f"---\nname: {canonical}\ndescription: alias test\n---\n\n"
            f"{body}\n"
        ),
        encoding="utf-8",
    )
    return skill_md


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")


def _audit_events(home: Path) -> list[dict]:
    path = home / "logs" / "skill-grants.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exact_disabled_skill_grant_preserves_config(disabled_skills_home):
    home, config_path = disabled_skills_home
    original_config = config_path.read_bytes()
    original_disabled = skill_utils.get_disabled_skill_names()

    assert _view("alpha")["success"] is False
    assert _view("enabled")["success"] is True

    with skill_utils.skill_read_grant_scope(
        ["alpha"],
        session_id="session-1",
        profile="build",
        requester="local-cli",
        source="cli",
        ttl_seconds=60,
    ):
        assert _view("alpha")["success"] is True
        alpha_file = json.loads(
            skills_tool.skill_view(
                "alpha", file_path="references/details.md", preprocess=False
            )
        )
        assert alpha_file["success"] is True
        assert alpha_file["content"] == "alpha details\n"
        assert _view("beta")["success"] is False
        listed = json.loads(skills_tool.skills_list())
        listed_names = {skill["name"] for skill in listed["skills"]}
        assert "alpha" not in listed_names

    assert _view("alpha")["success"] is False
    assert config_path.read_bytes() == original_config
    assert skill_utils.get_disabled_skill_names() == original_disabled == {
        "alpha",
        "beta",
    }

    events = _audit_events(home)
    assert [event["event"] for event in events] == ["issued", "closed"]
    issued, closed = events
    expected_fields = {
        "event",
        "grant_id",
        "session_id",
        "task_id",
        "profile",
        "skill_names",
        "requester",
        "source",
        "issued_at",
        "expires_at",
        "recorded_at",
        "terminal_status",
    }
    assert set(issued) == expected_fields
    assert set(closed) == expected_fields
    assert issued["session_id"] == "session-1"
    assert issued["task_id"] is None
    assert issued["profile"] == "build"
    assert issued["skill_names"] == ["alpha"]
    assert issued["requester"] == "local-cli"
    assert issued["source"] == "cli"
    assert issued["issued_at"] < issued["expires_at"]
    assert issued["terminal_status"] == "active"
    assert closed["terminal_status"] == "completed"
    serialized = json.dumps(events).lower()
    for forbidden in (
        "content",
        "prompt",
        "credential",
        "user-data",
        "alpha body",
    ):
        assert forbidden not in serialized


def test_grant_claims_are_immutable(disabled_skills_home):
    with skill_utils.skill_read_grant_scope(
        ["alpha"],
        session_id="session-immutable",
        profile="build",
        requester="local-cli",
        source="cli",
        ttl_seconds=60,
    ) as grant:
        assert grant is not None
        with pytest.raises(FrozenInstanceError):
            grant.profile = "attacker"  # type: ignore[misc]


def test_nested_grant_scope_restores_outer_context(disabled_skills_home):
    with skill_utils.skill_read_grant_scope(
        ["alpha"],
        session_id="session-outer",
        profile="build",
        requester="local-cli",
        source="cli",
        ttl_seconds=60,
    ) as outer:
        assert outer is not None
        assert skill_utils.current_skill_read_grant() is outer
        assert skill_utils.is_skill_read_granted("alpha") is True

        with skill_utils.skill_read_grant_scope(
            ["beta"],
            session_id="session-inner",
            profile="build",
            requester="local-cli",
            source="cli",
            ttl_seconds=60,
        ) as inner:
            assert inner is not None
            assert skill_utils.current_skill_read_grant() is inner
            assert skill_utils.is_skill_read_granted("alpha") is False
            assert skill_utils.is_skill_read_granted("beta") is True

        assert skill_utils.current_skill_read_grant() is outer
        assert skill_utils.is_skill_read_granted("alpha") is True
        assert skill_utils.is_skill_read_granted("beta") is False

    assert skill_utils.current_skill_read_grant() is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"session_id": ""},
        {"source": "untrusted"},
        {"source": "kanban", "task_id": None},
        {"ttl_seconds": 0},
    ],
)
def test_invalid_grant_claims_fail_closed(
    disabled_skills_home, tmp_path, overrides
):
    audit_path = tmp_path / f"invalid-{len(str(overrides))}.jsonl"
    kwargs = {
        "session_id": "session-valid",
        "task_id": None,
        "profile": "build",
        "requester": "local-cli",
        "source": "cli",
        "ttl_seconds": 60,
        "audit_path": audit_path,
    }
    kwargs.update(overrides)

    try:
        with pytest.raises((TypeError, ValueError)):
            skill_utils.issue_skill_read_grant(["alpha"], **kwargs)
    finally:
        active = skill_utils.current_skill_read_grant()
        if active is not None:
            skill_utils.close_skill_read_grant(active, "failed")

    assert skill_utils.is_skill_read_granted("alpha") is False
    assert not audit_path.exists()


def test_enabled_skill_uses_existing_behavior_without_grant_or_audit(
    disabled_skills_home,
):
    home, config_path = disabled_skills_home
    original_config = config_path.read_bytes()
    original_disabled = skill_utils.get_disabled_skill_names()

    grant = skill_utils.issue_skill_read_grant(
        ["enabled"],
        session_id="session-enabled",
        profile="build",
        requester="local-cli",
        source="cli",
        ttl_seconds=60,
    )

    assert grant is None
    assert skill_utils.current_skill_read_grant() is None
    assert _view("enabled")["success"] is True
    assert config_path.read_bytes() == original_config
    assert skill_utils.get_disabled_skill_names() == original_disabled
    assert not (home / "logs" / "skill-grants.jsonl").exists()


def test_expired_grant_is_denied_and_closed(disabled_skills_home, monkeypatch):
    home, _ = disabled_skills_home
    now = 1_000.0
    monkeypatch.setattr(skill_utils.time, "time", lambda: now)

    grant = skill_utils.issue_skill_read_grant(
        ["alpha"],
        session_id="session-expired",
        profile="build",
        requester="local-cli",
        source="cli",
        ttl_seconds=10,
    )
    assert grant is not None
    assert skill_utils.is_skill_read_granted("alpha") is True

    now = 1_011.0
    assert skill_utils.is_skill_read_granted("alpha") is False
    skill_utils.close_skill_read_grant(grant, "completed")

    events = _audit_events(home)
    assert events[-1]["event"] == "closed"
    assert events[-1]["terminal_status"] == "expired"
    assert skill_utils.is_skill_read_granted("alpha") is False


@pytest.mark.parametrize(
    ("error", "terminal_status"),
    [
        (RuntimeError("boom"), "failed"),
        (KeyboardInterrupt(), "cancelled"),
        (TimeoutError("slow"), "timed_out"),
    ],
)
def test_grant_scope_cleans_up_on_abnormal_exit(
    disabled_skills_home, error, terminal_status
):
    home, _ = disabled_skills_home

    with pytest.raises(type(error)):
        with skill_utils.skill_read_grant_scope(
            ["alpha"],
            session_id=f"session-{terminal_status}",
            profile="build",
            requester="local-cli",
            source="cli",
            ttl_seconds=60,
        ):
            assert skill_utils.is_skill_read_granted("alpha") is True
            raise error

    assert skill_utils.is_skill_read_granted("alpha") is False
    assert _audit_events(home)[-1]["terminal_status"] == terminal_status


@pytest.mark.parametrize(
    ("exit_code", "terminal_status"),
    [
        (None, "completed"),
        (0, "completed"),
        (130, "cancelled"),
        (1, "failed"),
    ],
)
def test_grant_scope_classifies_system_exit(
    disabled_skills_home, exit_code, terminal_status
):
    home, _ = disabled_skills_home

    with pytest.raises(SystemExit) as exc_info:
        with skill_utils.skill_read_grant_scope(
            ["alpha"],
            session_id=f"session-exit-{exit_code}",
            profile="build",
            requester="local-cli",
            source="cli",
            ttl_seconds=60,
        ):
            assert skill_utils.is_skill_read_granted("alpha") is True
            raise SystemExit(exit_code)

    assert exc_info.value.code == exit_code
    assert skill_utils.current_skill_read_grant() is None
    assert skill_utils.is_skill_read_granted("alpha") is False
    assert _audit_events(home)[-1]["terminal_status"] == terminal_status


def test_parallel_grants_are_isolated(disabled_skills_home):
    home, config_path = disabled_skills_home
    original_config = config_path.read_bytes()
    original_disabled = skill_utils.get_disabled_skill_names()
    barrier = threading.Barrier(2)
    results: dict[str, tuple[bool, bool, str, str | None]] = {}

    def run(name: str, other: str) -> None:
        with skill_utils.skill_read_grant_scope(
            [name],
            session_id=f"session-{name}",
            task_id=f"task-{name}",
            profile="build",
            requester="build",
            source="kanban",
            ttl_seconds=60,
        ):
            barrier.wait(timeout=5)
            active = skill_utils.current_skill_read_grant()
            assert active is not None
            results[name] = (
                skill_utils.is_skill_read_granted(name),
                skill_utils.is_skill_read_granted(other),
                active.session_id,
                active.task_id,
            )

    alpha = threading.Thread(target=run, args=("alpha", "beta"))
    beta = threading.Thread(target=run, args=("beta", "alpha"))
    alpha.start()
    beta.start()
    alpha.join(timeout=5)
    beta.join(timeout=5)

    assert results == {
        "alpha": (True, False, "session-alpha", "task-alpha"),
        "beta": (True, False, "session-beta", "task-beta"),
    }
    assert skill_utils.is_skill_read_granted("alpha") is False
    assert skill_utils.is_skill_read_granted("beta") is False
    assert config_path.read_bytes() == original_config
    assert skill_utils.get_disabled_skill_names() == original_disabled

    events = _audit_events(home)
    by_grant: dict[str, list[dict]] = {}
    for event in events:
        by_grant.setdefault(event["grant_id"], []).append(event)
    assert len(by_grant) == 2
    attribution = {
        (
            records[0]["session_id"],
            records[0]["task_id"],
            records[0]["skill_names"][0],
        )
        for records in by_grant.values()
    }
    assert attribution == {
        ("session-alpha", "task-alpha", "alpha"),
        ("session-beta", "task-beta", "beta"),
    }
    assert all(
        [record["terminal_status"] for record in records]
        == ["active", "completed"]
        for records in by_grant.values()
    )


def test_tool_thread_context_propagates_only_the_current_grant(
    disabled_skills_home, tmp_path
):
    from tools.thread_context import propagate_context_to_thread

    audit_path = tmp_path / "propagated.jsonl"
    result = []
    with skill_utils.skill_read_grant_scope(
        ["alpha"],
        session_id="session-parent",
        profile="build",
        requester="local-cli",
        source="cli",
        ttl_seconds=60,
        audit_path=audit_path,
    ):
        wrapped = propagate_context_to_thread(
            lambda: result.append(
                (
                    skill_utils.is_skill_read_granted("alpha"),
                    skill_utils.is_skill_read_granted("beta"),
                )
            )
        )
        thread = threading.Thread(target=wrapped)
        thread.start()
        thread.join()

    assert result == [(True, False)]


def test_grant_fails_closed_when_issuance_audit_cannot_be_written(
    disabled_skills_home, monkeypatch, tmp_path
):
    monkeypatch.setattr(skill_utils, "_write_skill_grant_audit", lambda *a, **k: False)

    with pytest.raises(RuntimeError, match="required skill-grant audit"):
        skill_utils.issue_skill_read_grant(
            ["alpha"],
            session_id="session-audit-failure",
            profile="build",
            requester="local-cli",
            source="cli",
            ttl_seconds=60,
            audit_path=tmp_path / "unwritten.jsonl",
        )

    assert skill_utils.is_skill_read_granted("alpha") is False


def test_categorized_skill_grant_stays_bound_to_requested_identity(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    target = skills_dir / "category" / "directory-alias"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: categorized\n---\n\nalpha body\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "skills:\n  disabled: [alpha]\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    skill_utils._raw_config_cache_clear()

    for requested in ("category/directory-alias", "category:directory-alias"):
        targets = skills_tool.resolve_skill_read_grant_targets([requested])
        assert list(targets) == [requested]
        assert "alpha" in targets[requested]
        with skill_utils.skill_read_grant_scope(
            list(targets),
            session_id=f"session-{requested}",
            profile="build",
            requester="local-cli",
            source="cli",
            authorization_aliases=targets,
        ):
            assert skill_utils.is_skill_read_granted(requested) is True
            assert _view(requested)["success"] is True


def test_categorized_colon_identity_matches_disabled_config(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    target = skills_dir / "category" / "directory-alias"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: categorized\n---\n\ncolon body\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "skills:\n  disabled: [category:directory-alias]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    skill_utils._raw_config_cache_clear()

    requested = "category:directory-alias"
    targets = skills_tool.resolve_skill_read_grant_targets([requested])
    assert list(targets) == [requested]
    with skill_utils.skill_read_grant_scope(
        list(targets),
        session_id="session-colon",
        profile="build",
        requester="local-cli",
        source="cli",
        authorization_aliases=targets,
    ):
        assert skill_utils.is_skill_read_granted(requested) is True
        viewed = _view(requested)
        assert viewed["success"] is True
        assert "colon body" in viewed["content"]


def test_colon_disabled_local_is_denied_via_slash_and_unique_bare_before_read(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    target = skills_dir / "category" / "directory-alias"
    target.mkdir(parents=True)
    skill_md = target / "SKILL.md"
    skill_md.write_text(
        "---\nname: alpha\ndescription: categorized\n---\n\n"
        "disabled-content-canary\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        "skills:\n  disabled: [category:directory-alias]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    skill_utils._raw_config_cache_clear()

    slash_targets = skills_tool.resolve_skill_read_grant_targets(
        ["category/directory-alias"]
    )
    bare_targets = skills_tool.resolve_skill_read_grant_targets(["alpha"])
    assert slash_targets["category/directory-alias"] >= {
        "category/directory-alias",
        "category:directory-alias",
        "alpha",
    }
    assert bare_targets["alpha"] >= {
        "category/directory-alias",
        "category:directory-alias",
        "alpha",
    }

    original_read_text = Path.read_text

    def reject_full_skill_read(path, *args, **kwargs):
        if path == skill_md:
            raise AssertionError("disabled skill content was read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_full_skill_read)
    for selector in ("category/directory-alias", "alpha"):
        viewed = _view(selector)
        assert viewed["success"] is False
        assert "disabled" in viewed["error"].lower()
        assert "disabled-content-canary" not in json.dumps(viewed)


def test_slash_grant_authorizes_colon_disabled_target_but_not_sibling(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    for category in ("one", "two"):
        target = skills_dir / category / "alias"
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text(
            (
                "---\nname: shared\ndescription: collision\n---\n\n"
                f"{category} body\n"
            ),
            encoding="utf-8",
        )
    (home / "config.yaml").write_text(
        "skills:\n  disabled: [one:alias, two:alias]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    skill_utils._raw_config_cache_clear()

    requested = "one/alias"
    targets = skills_tool.resolve_skill_read_grant_targets([requested])
    assert "one:alias" in targets[requested]
    with skill_utils.skill_read_grant_scope(
        list(targets),
        session_id="session-slash-colon",
        profile="build",
        requester="local-cli",
        source="cli",
        authorization_aliases=targets,
    ):
        assert skill_utils.is_skill_read_granted(requested) is True
        selected = _view(requested)
        sibling = _view("two/alias")
        assert selected["success"] is True
        assert "one body" in selected["content"]
        assert sibling["success"] is False
        assert "two body" not in json.dumps(sibling)


def test_explicit_categorized_path_ignores_leaf_collision_and_denies_sibling(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    for category in ("one", "two"):
        target = skills_dir / category / "alias"
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text(
            (
                "---\nname: shared\ndescription: collision\n---\n\n"
                f"{category} body\n"
            ),
            encoding="utf-8",
        )
    (home / "config.yaml").write_text(
        "skills:\n  disabled: [shared]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    skill_utils._raw_config_cache_clear()

    requested = "one/alias"
    targets = skills_tool.resolve_skill_read_grant_targets([requested])
    assert list(targets) == [requested]
    assert "shared" in targets[requested]
    assert skills_tool.resolve_skill_read_grant_names(["alias"]) == []
    assert skills_tool.resolve_skill_read_grant_names(["shared"]) == []
    for selector in ("one/alias", "two/alias"):
        denied = _view(selector)
        assert denied["success"] is False
        assert f"{selector.split('/')[0]} body" not in json.dumps(denied)
    with skill_utils.skill_read_grant_scope(
        list(targets),
        session_id="session-explicit",
        profile="build",
        requester="local-cli",
        source="cli",
        authorization_aliases=targets,
    ):
        selected = _view(requested)
        sibling = _view("two/alias")
        assert selected["success"] is True
        assert "one body" in selected["content"]
        assert sibling["success"] is False
        assert "two body" not in json.dumps(sibling)


@pytest.mark.parametrize("legacy_flat", [False, True], ids=["package", "flat"])
def test_fallback_name_disabled_denies_canonical_selector_and_sibling(
    tmp_path, monkeypatch, legacy_flat
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"

    def write_skill(category, canonical, body):
        target = skills_dir / category / "directory-alias"
        skill_md = target.with_suffix(".md") if legacy_flat else target / "SKILL.md"
        skill_md.parent.mkdir(parents=True)
        skill_md.write_text(
            (
                f"---\nname: {canonical}\ndescription: fallback alias\n---\n\n"
                f"{body}\n"
            ),
            encoding="utf-8",
        )

    write_skill("category", "alpha", "alpha body")
    write_skill("sibling", "beta", "beta body")
    (home / "config.yaml").write_text(
        "skills:\n  disabled: [directory-alias]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    skill_utils._raw_config_cache_clear()

    targets = skills_tool.resolve_skill_read_grant_targets(["alpha"])
    assert targets["alpha"] >= {"alpha", "directory-alias"}
    assert skills_tool.resolve_skill_read_grant_names(["directory-alias"]) == []
    for selector, body in (("alpha", "alpha body"), ("beta", "beta body")):
        denied = _view(selector)
        assert denied["success"] is False
        assert body not in json.dumps(denied)

    with skill_utils.skill_read_grant_scope(
        list(targets),
        session_id=f"session-fallback-{'flat' if legacy_flat else 'package'}",
        profile="build",
        requester="local-cli",
        source="cli",
        authorization_aliases=targets,
    ) as grant:
        assert grant is not None
        selected = _view("alpha")
        sibling = _view("beta")
        assert selected["success"] is True
        assert "alpha body" in selected["content"]
        assert sibling["success"] is False
        assert "beta body" not in json.dumps(sibling)


@pytest.mark.parametrize("legacy_flat", [False, True], ids=["package", "flat"])
def test_ambiguous_relative_path_is_not_a_disabled_or_grant_alias(
    tmp_path, monkeypatch, legacy_flat
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    external_dir = tmp_path / "external-skills"

    def write_skill(root, canonical, body):
        target = root / "category" / "foo"
        skill_md = target.with_suffix(".md") if legacy_flat else target / "SKILL.md"
        skill_md.parent.mkdir(parents=True)
        skill_md.write_text(
            (
                f"---\nname: {canonical}\ndescription: relative collision\n---\n\n"
                f"{body}\n"
            ),
            encoding="utf-8",
        )

    write_skill(skills_dir, "alpha", "primary body")
    write_skill(external_dir, "beta", "external body")
    (home / "config.yaml").write_text(
        "skills:\n  disabled: [category/foo]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(
        skill_utils, "get_external_skills_dirs", lambda: [external_dir]
    )
    skill_utils._raw_config_cache_clear()

    targets = skills_tool.resolve_skill_read_grant_targets(["alpha"])
    assert list(targets) == ["alpha"]
    assert "category/foo" not in targets["alpha"]
    assert "category:foo" not in targets["alpha"]
    assert _view("alpha")["success"] is True

    with skill_utils.skill_read_grant_scope(
        list(targets),
        session_id=f"session-relative-{'flat' if legacy_flat else 'package'}",
        profile="build",
        requester="local-cli",
        source="cli",
        authorization_aliases=targets,
    ) as grant:
        assert grant is None
        assert skill_utils.current_skill_read_grant() is None

    assert skills_tool.resolve_skill_read_grant_names(["category/foo"]) == []
    ambiguous = _view("category/foo")
    assert ambiguous["success"] is False
    assert "ambiguous" in ambiguous["error"].lower()
    assert "primary body" not in json.dumps(ambiguous)
    assert "external body" not in json.dumps(ambiguous)


@pytest.mark.parametrize("legacy_flat", [False, True], ids=["package", "flat"])
def test_overlapping_roots_enumerate_every_same_file_selector_before_read(
    tmp_path, monkeypatch, legacy_flat
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    overlap_root = skills_dir / "overlap"
    skill_md = _write_alias_test_skill(
        skills_dir,
        "overlap/category/foo",
        "alpha",
        "overlap body",
        legacy_flat=legacy_flat,
    )
    _write_alias_test_skill(
        skills_dir,
        "sibling/bar",
        "beta",
        "sibling body",
        legacy_flat=legacy_flat,
    )
    config_path = home / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(
        skill_utils, "get_external_skills_dirs", lambda: [overlap_root]
    )

    expected_paths = {"overlap/category/foo", "category/foo"}
    allow_full_read = False
    original_read_text = Path.read_text

    def reject_ungranted_full_read(path, *args, **kwargs):
        if (
            not allow_full_read
            and skills_tool._same_local_skill_file(path, skill_md)
        ):
            raise AssertionError("disabled skill content was read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_ungranted_full_read)
    for disabled_path in sorted(expected_paths):
        config_path.write_text(
            f"skills:\n  disabled: [{disabled_path}, beta]\n",
            encoding="utf-8",
        )
        skill_utils._raw_config_cache_clear()
        targets = skills_tool.resolve_skill_read_grant_targets(["alpha"])
        assert targets["alpha"] >= expected_paths | {"alpha"}

        allow_full_read = False
        denied = _view("alpha")
        assert denied["success"] is False
        assert "disabled" in denied["error"].lower()
        assert "overlap body" not in json.dumps(denied)

        allow_full_read = True
        with skill_utils.skill_read_grant_scope(
            list(targets),
            session_id=(
                f"session-overlap-{'flat' if legacy_flat else 'package'}-"
                f"{disabled_path}"
            ),
            profile="build",
            requester="local-cli",
            source="cli",
            authorization_aliases=targets,
        ) as grant:
            assert grant is not None
            selected = _view("alpha")
            sibling = _view("beta")
            assert selected["success"] is True
            assert "overlap body" in selected["content"]
            assert sibling["success"] is False
            assert "sibling body" not in json.dumps(sibling)


@pytest.mark.parametrize("legacy_flat", [False, True], ids=["package", "flat"])
def test_symlink_aliases_enumerate_real_and_alias_selectors_before_read(
    tmp_path, monkeypatch, legacy_flat
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    skill_md = _write_alias_test_skill(
        skills_dir,
        "real/foo",
        "alpha",
        "symlink body",
        legacy_flat=legacy_flat,
    )
    if legacy_flat:
        alias_dir = skills_dir / "alias"
        alias_dir.mkdir(parents=True)
        _symlink_or_skip(
            alias_dir / "foo.md", skill_md, target_is_directory=False
        )
    else:
        _symlink_or_skip(
            skills_dir / "alias",
            skills_dir / "real",
            target_is_directory=True,
        )
    _write_alias_test_skill(
        skills_dir,
        "sibling/bar",
        "beta",
        "sibling body",
        legacy_flat=legacy_flat,
    )
    config_path = home / "config.yaml"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)

    expected_paths = {"real/foo", "alias/foo"}
    allow_full_read = False
    original_read_text = Path.read_text

    def reject_ungranted_full_read(path, *args, **kwargs):
        if (
            not allow_full_read
            and skills_tool._same_local_skill_file(path, skill_md)
        ):
            raise AssertionError("disabled skill content was read")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_ungranted_full_read)
    for disabled_path in sorted(expected_paths):
        config_path.write_text(
            f"skills:\n  disabled: [{disabled_path}, beta]\n",
            encoding="utf-8",
        )
        skill_utils._raw_config_cache_clear()
        targets = skills_tool.resolve_skill_read_grant_targets(["alpha"])
        assert targets["alpha"] >= expected_paths | {"alpha"}

        allow_full_read = False
        denied = _view("alpha")
        assert denied["success"] is False
        assert "disabled" in denied["error"].lower()
        assert "symlink body" not in json.dumps(denied)

        allow_full_read = True
        with skill_utils.skill_read_grant_scope(
            list(targets),
            session_id=(
                f"session-symlink-{'flat' if legacy_flat else 'package'}-"
                f"{disabled_path}"
            ),
            profile="build",
            requester="local-cli",
            source="cli",
            authorization_aliases=targets,
        ) as grant:
            assert grant is not None
            selected = _view("alpha")
            sibling = _view("beta")
            assert selected["success"] is True
            assert "symlink body" in selected["content"]
            assert sibling["success"] is False
            assert "sibling body" not in json.dumps(sibling)


@pytest.mark.parametrize(
    ("excluded_dir", "legacy_flat"),
    [("node_modules", False), (".git", True)],
    ids=["node-modules-package", "git-flat"],
)
def test_excluded_flat_markdown_cannot_resolve_or_collide(
    tmp_path, monkeypatch, excluded_dir, legacy_flat
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    canonical = f"visible-{'flat' if legacy_flat else 'package'}"
    _write_alias_test_skill(
        skills_dir,
        "category/visible",
        canonical,
        "legitimate body",
        legacy_flat=legacy_flat,
    )

    excluded_root = skills_dir / excluded_dir
    excluded_root.mkdir(parents=True)
    (excluded_root / "matching.md").write_text(
        (
            f"---\nname: {canonical}\ndescription: excluded collision\n---\n\n"
            "excluded collision body\n"
        ),
        encoding="utf-8",
    )
    hidden_name = f"excluded-only-{'git' if excluded_dir == '.git' else 'deps'}"
    (excluded_root / "hidden.md").write_text(
        (
            f"---\nname: {hidden_name}\ndescription: excluded only\n---\n\n"
            "excluded only body\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(skill_utils, "get_external_skills_dirs", lambda: [])
    skill_utils._raw_config_cache_clear()

    targets = skills_tool.resolve_skill_read_grant_targets([canonical])
    assert list(targets) == [canonical]
    assert all(excluded_dir not in alias for alias in targets[canonical])

    visible = _view(canonical)
    assert visible["success"] is True
    assert "legitimate body" in visible["content"]
    assert "excluded collision body" not in json.dumps(visible)

    for hidden_selector in (hidden_name, "hidden"):
        hidden = _view(hidden_selector)
        assert hidden["success"] is False
        assert "excluded only body" not in json.dumps(hidden)


def test_canonicalization_refuses_ambiguous_bare_name(tmp_path, monkeypatch):
    skills_dir = tmp_path / ".hermes" / "skills"
    for category in ("one", "two"):
        target = skills_dir / category / "alpha"
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: collision\n---\n\nbody\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)

    assert skills_tool.resolve_skill_read_grant_names(["alpha"]) == []


def test_canonicalization_distinguishes_plugin_colon_from_local_slash(
    tmp_path, monkeypatch
):
    skills_dir = tmp_path / ".hermes" / "skills"
    local = skills_dir / "bundle" / "alpha"
    local.mkdir(parents=True)
    (local / "SKILL.md").write_text(
        "---\nname: local-alpha\ndescription: local\n---\n\nlocal\n",
        encoding="utf-8",
    )
    plugin_md = tmp_path / "plugin" / "SKILL.md"
    plugin_md.parent.mkdir()
    plugin_md.write_text(
        "---\nname: alpha\ndescription: plugin\n---\n\nplugin\n",
        encoding="utf-8",
    )

    class PluginManager:
        def find_plugin_skill(self, name):
            return plugin_md if name == "bundle:alpha" else None

        def list_plugin_skills(self, namespace):
            return ["alpha"] if namespace == "bundle" else []

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager", lambda: PluginManager()
    )

    assert skills_tool.resolve_skill_read_grant_names(["bundle:alpha"]) == [
        "bundle:alpha"
    ]
    assert skills_tool.resolve_skill_read_grant_names(["bundle/alpha"]) == [
        "bundle/alpha"
    ]


def test_real_plugin_identity_neither_disables_nor_authorizes_local_slash(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    local = skills_dir / "bundle" / "alpha"
    local.mkdir(parents=True)
    (local / "SKILL.md").write_text(
        "---\nname: local-alpha\ndescription: local\n---\n\nlocal body\n",
        encoding="utf-8",
    )
    plugin_md = tmp_path / "plugin" / "SKILL.md"
    plugin_md.parent.mkdir()
    plugin_md.write_text(
        "---\nname: alpha\ndescription: plugin\n---\n\nplugin body\n",
        encoding="utf-8",
    )

    class PluginManager:
        def find_plugin_skill(self, name):
            return plugin_md if name == "bundle:alpha" else None

        def list_plugin_skills(self, namespace):
            return ["alpha"] if namespace == "bundle" else []

    config_path = home / "config.yaml"
    config_path.write_text(
        "skills:\n  disabled: [bundle:alpha]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager", lambda: PluginManager()
    )
    skill_utils._raw_config_cache_clear()

    local_targets = skills_tool.resolve_skill_read_grant_targets(["bundle/alpha"])
    assert "bundle:alpha" not in local_targets["bundle/alpha"]
    assert _view("bundle/alpha")["success"] is True

    config_path.write_text(
        "skills:\n  disabled: [bundle:alpha, local-alpha]\n",
        encoding="utf-8",
    )
    skill_utils._raw_config_cache_clear()
    plugin_targets = skills_tool.resolve_skill_read_grant_targets(["bundle:alpha"])
    with skill_utils.skill_read_grant_scope(
        list(plugin_targets),
        session_id="session-plugin",
        profile="build",
        requester="local-cli",
        source="cli",
        authorization_aliases=plugin_targets,
    ):
        assert _view("bundle:alpha")["success"] is True
        local_view = _view("bundle/alpha")
        assert local_view["success"] is False
        assert "local body" not in json.dumps(local_view)


def test_plugin_owned_frontmatter_name_cannot_alias_local_skill(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    local = skills_dir / "local" / "foo"
    local.mkdir(parents=True)
    (local / "SKILL.md").write_text(
        "---\nname: bundle:alpha\ndescription: local\n---\n\nlocal body\n",
        encoding="utf-8",
    )
    plugin_md = tmp_path / "plugin" / "SKILL.md"
    plugin_md.parent.mkdir()
    plugin_md.write_text(
        "---\nname: alpha\ndescription: plugin\n---\n\nplugin body\n",
        encoding="utf-8",
    )

    class PluginManager:
        def find_plugin_skill(self, name):
            return plugin_md if name == "bundle:alpha" else None

        def list_plugin_skills(self, namespace):
            return ["alpha"] if namespace == "bundle" else []

    (home / "config.yaml").write_text(
        "skills:\n  disabled: [bundle:alpha]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)
    monkeypatch.setattr(
        "hermes_cli.plugins.get_plugin_manager", lambda: PluginManager()
    )
    skill_utils._raw_config_cache_clear()

    local_targets = skills_tool.resolve_skill_read_grant_targets(["local/foo"])
    assert "bundle:alpha" not in local_targets["local/foo"]
    grant = skill_utils.issue_skill_read_grant(
        list(local_targets),
        session_id="session-local-poison",
        profile="build",
        requester="local-cli",
        source="cli",
        authorization_aliases=local_targets,
    )
    assert grant is None
    assert skill_utils.current_skill_read_grant() is None
    local_view = _view("local/foo")
    plugin_view = _view("bundle:alpha")
    assert local_view["success"] is True
    assert "local body" in local_view["content"]
    assert plugin_view["success"] is False
    assert "plugin body" not in json.dumps(plugin_view)


def test_frontmatter_slash_alias_must_resolve_back_to_selected_skill(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    selected = skills_dir / "local" / "foo"
    selected.mkdir(parents=True)
    (selected / "SKILL.md").write_text(
        "---\nname: other/target\ndescription: selected\n---\n\nselected body\n",
        encoding="utf-8",
    )
    actual = skills_dir / "other" / "target"
    actual.mkdir(parents=True)
    (actual / "SKILL.md").write_text(
        "---\nname: actual-target\ndescription: actual\n---\n\nactual body\n",
        encoding="utf-8",
    )
    config_path = home / "config.yaml"
    config_path.write_text(
        "skills:\n  disabled: [other/target]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    skill_utils._raw_config_cache_clear()

    selected_targets = skills_tool.resolve_skill_read_grant_targets(["local/foo"])
    assert "other/target" not in selected_targets["local/foo"]
    grant = skill_utils.issue_skill_read_grant(
        list(selected_targets),
        session_id="session-slash-poison",
        profile="build",
        requester="local-cli",
        source="cli",
        authorization_aliases=selected_targets,
    )
    assert grant is None
    assert _view("local/foo")["success"] is True
    assert _view("other/target")["success"] is False

    config_path.write_text(
        "skills:\n  disabled: [other/target, local/foo]\n",
        encoding="utf-8",
    )
    skill_utils._raw_config_cache_clear()
    actual_targets = skills_tool.resolve_skill_read_grant_targets(["other/target"])
    with skill_utils.skill_read_grant_scope(
        list(actual_targets),
        session_id="session-actual-target",
        profile="build",
        requester="local-cli",
        source="cli",
        authorization_aliases=actual_targets,
    ):
        assert _view("other/target")["success"] is True
        poisoned_view = _view("local/foo")
        assert poisoned_view["success"] is False
        assert "selected body" not in json.dumps(poisoned_view)


@pytest.mark.parametrize(
    "requested",
    ["../alpha", "/absolute/alpha", "missing", "bundle:", "alpha/../beta"],
)
def test_invalid_skill_identity_does_not_resolve_to_a_grant(
    disabled_skills_home, requested
):
    assert skills_tool.resolve_skill_read_grant_names([requested]) == []
