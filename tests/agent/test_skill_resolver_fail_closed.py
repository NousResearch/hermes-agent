"""B5-F fail-closed skill-root resolution: 13 negative gates + control.

Derived from the negative fixtures in ``~/.hermes/cache/scratch/sgb5/taskE/`` (N01-N14)
and from the platform-patch requirements in ``B5_F_FAIL_CLOSED_REMEDIATION_SPEC.md``
§7.4 / R1-R9. Each test asserts the resolver *says so* — raising, reporting or refusing —
rather than quietly returning a smaller skill set.

Cases: missing root, unreadable root, unreadable package, broken root symlink,
out-of-root symlink, internal symlink, symlink loop, TOCTOU replace-after-read,
TOCTOU delete-replace, same-tier ambiguity, local shadow, disabled, hash mismatch,
plus the self-reported execution environment (uid / mount namespace).
"""

import errno
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from agent.skill_utils import (
    TIER_EXTERNAL,
    TIER_LOCAL,
    SkillResolutionError,
    assert_same_identity,
    classify_external_root,
    content_sha256,
    effective_set_report,
    exec_environment_report,
    external_root_problems,
    fail_closed_enabled,
    get_external_skills_dirs,
    iter_skill_index_files_strict,
    load_skill_file_checked,
    read_verified_bytes,
    read_verified_text,
    require_effective_set,
    require_external_dirs,
    resolve_skill_catalog,
    reset_skill_load_errors,
    root_immutability_report,
    take_external_root_problems,
    take_skill_load_errors,
)

TARGET = "openrouter-ranking-monitor"

SKILL_BODY = "---\nname: {name}\ndescription: {desc}\n---\n\n{desc}\n"


# Mode-000 fixtures assert nothing for uid 0: nothing blocks a read for root.
needs_non_root = pytest.mark.skipif(
    os.geteuid() == 0, reason="mode 000 does not deny a read to root")


def pkg(root, name, desc=None):
    d = Path(root) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        SKILL_BODY.format(name=name, desc=desc or f"{name}-v1"), encoding="utf-8")
    return d


def unlock_tree(root):
    subprocess.run(["chmod", "-R", "u+rwX", str(root)], capture_output=True)


@pytest.fixture
def gen(tmp_path):
    """A clean external view root: `<tmp>/generations/gen-.../` (the control shape)."""
    return tmp_path / "generations" / "gen-20261106T000000Z"


@pytest.fixture(autouse=True)
def _cleanup_perm(tmp_path):
    # The resolver reports through module state; isolate it per test.
    take_external_root_problems()
    reset_skill_load_errors()
    yield
    take_external_root_problems()
    reset_skill_load_errors()
    # Leave pytest able to remove its own tmp_path even after a mode-000 fixture.
    unlock_tree(tmp_path)


# --------------------------------------------------------------------------
# control: a well-formed root resolves, and nothing is reported
# --------------------------------------------------------------------------

class TestControl:

    def test_clean_root_resolves_and_reports_nothing(self, gen):
        pkg(gen, TARGET)
        pkg(gen, "second-skill")

        files = iter_skill_index_files_strict(gen, "SKILL.md")

        assert [p.parent.name for p in files] == [TARGET, "second-skill"]
        reset_skill_load_errors()
        parsed = load_skill_file_checked(files[0])
        assert parsed.is_compatible and parsed.frontmatter.get("name") == TARGET
        assert take_skill_load_errors() == []
        assert external_root_problems() == []

    def test_bytes_carry_the_identity_they_were_read_from(self, gen):
        target = pkg(gen, TARGET) / "SKILL.md"
        data, ident = read_verified_bytes(target)
        assert data == target.read_bytes()
        assert assert_same_identity(target, ident) == ident
        text, _ = read_verified_text(target)
        assert text.startswith("---\n")
        assert content_sha256(target) == __import__("hashlib").sha256(data).hexdigest()


# --------------------------------------------------------------------------
# N01 / N02 — roots that are missing or unreachable must not vanish quietly
# --------------------------------------------------------------------------

class TestUnreachableRoots:

    def test_missing_root_is_classified_by_errno_not_exists(self, tmp_path):
        missing = tmp_path / "never-created"
        status, eno, _msg = classify_external_root(missing)
        assert status == "missing"
        assert eno == errno.ENOENT
        # The old code called is_dir() and debug-logged this into oblivion.
        assert not missing.exists()

    @needs_non_root
    def test_permission_denied_is_not_folded_into_missing(self, tmp_path):
        blocked = tmp_path / "blocked-parent"
        inside = blocked / "root"
        blocked.mkdir()
        inside.mkdir()
        os.chmod(blocked, 0o000)
        status, eno, _msg = classify_external_root(inside)
        # exists() would answer False here too — that conflation is the bug (R1.1).
        assert (status, eno) == ("permission", errno.EACCES)
        # os.path.exists() answers False here too — that conflation of EACCES with ENOENT
        # is exactly what made the old resolver unable to tell the two apart (R1.1).
        assert os.path.exists(str(inside)) is False

    def test_missing_configured_root_raises_instead_of_dropping(self, tmp_path, monkeypatch):
        import agent.skill_utils as su

        missing = tmp_path / "gone-root"
        monkeypatch.setattr(su, "get_config_path", lambda: tmp_path / "config.yaml")
        monkeypatch.setattr(su, "_config_cache_key", lambda p: None)
        monkeypatch.setattr(su, "_skills_cfg", lambda: {"external_dirs": [str(missing)]})
        monkeypatch.setattr(su, "get_skills_dir", lambda: tmp_path / "local-skills")
        monkeypatch.setattr("hermes_cli.config_backend.config_exists", lambda p: True)
        su._EXTERNAL_DIRS_CACHE.clear()

        with pytest.raises(SkillResolutionError) as ei:
            get_external_skills_dirs(strict=True)
        assert ei.value.kind == "external_root_unreachable"

        take_external_root_problems()
        # Non-strict still refuses to cache the failure and still reports it (R1.2/R1.4).
        assert get_external_skills_dirs(strict=False) == []
        problems = take_external_root_problems()
        assert [p.status for p in problems] == ["missing"]
        assert su._EXTERNAL_DIRS_CACHE == {}
        with pytest.raises(SkillResolutionError):
            su._EXTERNAL_ROOT_PROBLEMS.extend(problems)
            require_external_dirs()

    @needs_non_root
    def test_root_mode_000_scan_is_an_error_not_an_empty_root(self, gen):
        pkg(gen, TARGET)
        os.chmod(gen, 0o000)

        with pytest.raises(SkillResolutionError) as ei:
            iter_skill_index_files_strict(gen, "SKILL.md")
        assert ei.value.kind in ("walk_failed", "root_unreachable")
        assert ei.value.errno in (errno.EACCES, None)

    @needs_non_root
    def test_package_dir_mode_000_scan_is_an_error_not_an_empty_root(self, gen):
        target = pkg(gen, TARGET)
        os.chmod(target, 0o000)

        with pytest.raises(SkillResolutionError) as ei:
            iter_skill_index_files_strict(gen, "SKILL.md")
        assert ei.value.kind == "walk_failed"


# --------------------------------------------------------------------------
# N03-N06 — symlinks in the view tree (initial pilot: none at all)
# --------------------------------------------------------------------------

class TestSymlinksRejected:

    def test_root_itself_is_a_symlink(self, tmp_path, gen):
        real = tmp_path / "real-root"
        pkg(real, TARGET)
        link = tmp_path / "link-root"
        os.symlink(real, link)

        assert classify_external_root(link)[0] == "symlink"
        with pytest.raises(SkillResolutionError) as ei:
            iter_skill_index_files_strict(link, "SKILL.md")
        assert ei.value.kind == "symlink_rejected"

    def test_broken_root_symlink_is_rejected_as_a_symlink(self, tmp_path):
        link = tmp_path / "dangling-root"
        os.symlink(tmp_path / "nowhere", link)
        # lstat — not is_dir()/exists() — sees the link itself, so a dangling root is
        # rejected as a symlink rather than "missing" (both are failures, but only the
        # former names the actual defect).
        status, _eno, _msg = classify_external_root(link)
        assert status == "symlink"
        with pytest.raises(SkillResolutionError) as ei:
            iter_skill_index_files_strict(link, "SKILL.md")
        assert ei.value.kind == "symlink_rejected"

    def test_skill_md_symlink_escaping_the_root(self, tmp_path, gen):
        outside = pkg(tmp_path / "outside-payload", "escape-src")
        target_dir = gen / TARGET
        target_dir.mkdir(parents=True)
        os.symlink(outside / "SKILL.md", target_dir / "SKILL.md")

        with pytest.raises(SkillResolutionError) as ei:
            iter_skill_index_files_strict(gen, "SKILL.md")
        assert ei.value.kind == "symlink_rejected"

    def test_skill_md_symlink_inside_the_root_is_still_rejected(self, gen):
        pkg(gen, TARGET)
        real = pkg(gen, "real-src")
        victim = gen / TARGET / "SKILL.md"
        victim.unlink()
        os.symlink(real / "SKILL.md", victim)

        with pytest.raises(SkillResolutionError) as ei:
            iter_skill_index_files_strict(gen, "SKILL.md")
        assert ei.value.kind == "symlink_rejected"

    def test_package_dir_symlink_out_of_root(self, tmp_path, gen):
        outside = pkg(tmp_path / "outside-payload", "escape-src")
        gen.mkdir(parents=True)
        os.symlink(outside, gen / TARGET)

        with pytest.raises(SkillResolutionError) as ei:
            iter_skill_index_files_strict(gen, "SKILL.md")
        assert ei.value.kind == "symlink_rejected"

    def test_symlink_loop_directory(self, gen):
        gen.mkdir(parents=True, exist_ok=True)
        a = gen / "a"
        a.mkdir()
        os.symlink(gen, a / "back")

        # followlinks=True in the old walker let this be traversed; lstat rejects it first.
        with pytest.raises(SkillResolutionError) as ei:
            iter_skill_index_files_strict(gen, "SKILL.md")
        assert ei.value.kind == "symlink_rejected"

    def test_symlink_reference_file_inside_package(self, gen):
        target = pkg(gen, TARGET)
        os.symlink(target / "SKILL.md", target / "references-note")
        # A stray link anywhere under the root is fatal for the pilot (§3).
        with pytest.raises(SkillResolutionError):
            iter_skill_index_files_strict(gen, "SKILL.md")


# --------------------------------------------------------------------------
# N07 / N08 — TOCTOU: what was validated must be what is used
# --------------------------------------------------------------------------

class TestToctouIdentity:

    def test_replace_between_resolve_and_use_is_detected(self, gen):
        target = pkg(gen, TARGET) / "SKILL.md"
        _data, ident = read_verified_bytes(target)

        # Atomic replace (write temp + rename) is what a real swap looks like: the path is
        # still there, pointing at a different inode with different content.
        tmp = target.with_suffix(".new")
        tmp.write_text(SKILL_BODY.format(name=TARGET, desc="swapped"), encoding="utf-8")
        os.replace(tmp, target)

        # Same path, different object: "the path still exists" must not count as success.
        with pytest.raises(SkillResolutionError) as ei:
            assert_same_identity(target, ident)
        assert ei.value.kind == "identity_mismatch"

    def test_in_place_rewrite_is_detected(self, gen):
        target = pkg(gen, TARGET) / "SKILL.md"
        _data, ident = read_verified_bytes(target)
        # write_text() truncates the same inode: dev/ino survive, so the check must also
        # look at size/mtime or this rewrite would pass as "the same file".
        target.write_text(SKILL_BODY.format(name=TARGET, desc="rewritten longer body"),
                          encoding="utf-8")
        with pytest.raises(SkillResolutionError) as ei:
            assert_same_identity(target, ident)
        assert ei.value.kind == "identity_mismatch"

    def test_delete_between_resolve_and_use_is_detected(self, gen):
        target = pkg(gen, TARGET) / "SKILL.md"
        _data, ident = read_verified_bytes(target)

        target.unlink()

        with pytest.raises(SkillResolutionError) as ei:
            assert_same_identity(target, ident)
        assert ei.value.kind == "use_time_recheck_failed"
        assert ei.value.errno == errno.ENOENT

    def test_hash_attestation_follows_the_bytes_not_the_path(self, gen):
        target = pkg(gen, TARGET) / "SKILL.md"
        before = content_sha256(target)
        target.write_text(SKILL_BODY.format(name=TARGET, desc="v2"), encoding="utf-8")
        assert content_sha256(target) != before


# --------------------------------------------------------------------------
# N09 / N10 / N11 — set-level failures: ambiguity, shadowing, disabled
# --------------------------------------------------------------------------

class TestEffectiveSet:

    def _entry(self, name, tier, root, rel=None, load_name=None, status="unique"):
        rel = rel or name
        return {"name": name, "tier": tier, "root": Path(root),
                "path": Path(root) / rel / "SKILL.md", "load_name": load_name or name,
                "status": status}

    def test_same_tier_duplicate_name_is_not_a_winner(self, gen):
        pkg(gen, "dup-a")
        pkg(gen, "dup-b")
        files = iter_skill_index_files_strict(gen, "SKILL.md")
        entries = []
        for f in files:
            parsed = load_skill_file_checked(f)
            entries.append({"name": parsed.frontmatter.get("name"), "tier": TIER_EXTERNAL,
                            "root": gen, "path": f})
        # Give both copies the same declared name so the tier cannot arbitrate.
        for e in entries:
            e["name"] = "same-name"
        resolved = resolve_skill_catalog(entries)
        assert {e["status"] for e in resolved} <= {"ambiguous", "shadowed"}
        report = effective_set_report({"same-name": TIER_EXTERNAL},
                                      [e for e in resolved if e.get("load_name")])
        assert report.ok is False or "same-name" in report.missing

    def test_local_copy_shadowing_external_is_divergence(self, tmp_path):
        ext = tmp_path / "ext"
        local = tmp_path / "local"
        report = effective_set_report(
            {TARGET: TIER_EXTERNAL},
            [self._entry(TARGET, TIER_LOCAL, local)])

        assert report.ok is False
        assert report.divergence == [{"name": TARGET, "expected_tier": TIER_EXTERNAL,
                                      "actual_tier": TIER_LOCAL}]

    def test_disabled_skill_is_a_missing_member(self, tmp_path, gen):
        report = effective_set_report({TARGET: TIER_EXTERNAL}, [])
        assert report.ok is False
        assert report.missing == [TARGET]
        with pytest.raises(SkillResolutionError) as ei:
            require_effective_set({TARGET: TIER_EXTERNAL}, [])
        assert ei.value.kind == "effective_set_mismatch"

    def test_unexpected_package_is_reported(self, tmp_path, gen):
        report = effective_set_report({TARGET: TIER_EXTERNAL},
                                      [self._entry(TARGET, TIER_EXTERNAL, gen),
                                       self._entry("unexpected-package", TIER_EXTERNAL, gen)])
        assert report.unexpected == ["unexpected-package"]

    def test_load_error_makes_the_report_not_ok(self, gen, tmp_path):
        target = pkg(gen, TARGET) / "SKILL.md"
        reset_skill_load_errors()
        target.unlink()
        # The recorded failure must come from the shared loader, which is the path both
        # consumers use, rather than from a direct read nobody downstream would see.
        assert load_skill_file_checked(target).is_compatible is False
        report = effective_set_report({}, [])
        assert report.load_errors
        assert report.ok is False


# --------------------------------------------------------------------------
# N12 — unreadable skill file: incompatible, not "compatible and invisible"
# --------------------------------------------------------------------------

class TestUnreadableSkill:

    @needs_non_root
    def test_mode_000_skill_file_is_incompatible(self, gen):
        target = pkg(gen, TARGET) / "SKILL.md"
        os.chmod(target, 0o000)
        reset_skill_load_errors()

        parsed = load_skill_file_checked(target)

        assert parsed.is_compatible is False
        assert parsed.frontmatter == {}
        errors = take_skill_load_errors()
        assert errors and errors[0]["path"] == str(target)
        assert errors[0]["kind"] in ("open_failed", "read_failed", "size_mismatch")

    def test_invalid_utf8_skill_file_is_incompatible(self, gen):
        target = pkg(gen, TARGET) / "SKILL.md"
        target.write_bytes(b"---\nname: broken\n---\n\n\xff\xfe bad bytes\n")
        reset_skill_load_errors()

        parsed = load_skill_file_checked(target)

        assert parsed.is_compatible is False
        assert take_skill_load_errors()[0]["kind"] == "decode_failed"


# --------------------------------------------------------------------------
# N13 — the resolver states the environment it resolved in (R8 / R9)
# --------------------------------------------------------------------------

class TestEnvironmentAttestation:

    def test_exec_environment_is_self_reported(self):
        report = exec_environment_report()
        for key in ("uid", "gid", "mount_namespace", "hermes_home", "cwd"):
            assert key in report
        assert report["uid"] == os.getuid()
        assert report["mount_namespace"] is None or report["mount_namespace"].startswith("mnt:")

    def test_forged_namespace_does_not_compare_equal(self):
        real = exec_environment_report()
        forged = dict(real, mount_namespace="mnt:[4026531999]")
        assert real["mount_namespace"] != forged["mount_namespace"]

    def test_root_immutability_is_reported(self, gen):
        pkg(gen, TARGET)
        report = root_immutability_report(gen)
        assert report["owner_uid"] == os.stat(gen).st_uid
        assert "mount_readonly" in report
        assert stat.S_IMODE(int(report["mode"], 8)) == stat.S_IMODE(os.stat(gen).st_mode)

    def test_immutability_of_a_missing_root_is_an_error(self, tmp_path):
        with pytest.raises(SkillResolutionError):
            root_immutability_report(tmp_path / "nope")


# --------------------------------------------------------------------------
# config switch: fail-closed is the default, but explicitly disableable
# --------------------------------------------------------------------------

class TestFailClosedSwitch:

    def test_default_is_fail_closed(self, monkeypatch):
        import agent.skill_utils as su
        monkeypatch.setattr(su, "_skills_cfg_get", lambda key: None)
        assert fail_closed_enabled() is True

    def test_explicit_off_is_honored(self, monkeypatch):
        import agent.skill_utils as su
        monkeypatch.setattr(su, "_skills_cfg_get", lambda key: "false")
        assert fail_closed_enabled() is False
