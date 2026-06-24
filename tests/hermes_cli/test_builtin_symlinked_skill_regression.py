"""
Regression test for builtin-symlinked skill labeling (t_ddf08a93).

Background
==========
A per-profile skills directory that hosts a symlink (or byte-identical copy)
of a platform-level builtin skill must be labeled Source=builtin, Trust=builtin
in ``hermes skills list`` — NOT Source=local, Trust=local. The audit that
prompted this fix found ``macos-computer-use`` mislabeled ``local`` for every
profile that hosted the file at ``skills/apple/macos-computer-use/`` even
though it was the canonical builtin copy from the platform tree.

These tests pin the labeling behavior against a hermetic filesystem built
out of ``tmp_path`` (no real ``~/.hermes``) and explicitly REQUIRE the
profile copy to be a real symlink — no silent copy-fallback — so the
regression scenario is actually exercised.

Three surfaces are covered:

  1. ``do_list`` (library function) reports Source=builtin, Trust=builtin.
  2. ``do_list(show_provenance=True)`` shows a Provenance column whose
     value is the platform-level install-origin path.
  3. ``hermes skills list --provenance`` end-to-end via ``main()`` shows
     the same labels and column, so the CLI wiring is regression-tested.

All tests skip cleanly on filesystems that cannot create symlinks (e.g.
some Windows CI environments and sandboxes); they never fall back to a
copy silently — that would mask the regression they're meant to catch.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from rich.console import Console

from hermes_cli.skills_hub import do_list


# ─────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────


class _DummyLockFile:
    """Stand-in for tools/skills_hub.py:HubLockFile — no hub installs."""

    def __init__(self, entries=None):
        self._entries = entries or []

    def list_installed(self):
        return list(self._entries)


def _require_symlink_support(target: Path, link: Path) -> None:
    """Create ``link`` as a symlink to ``target`` or skip the test.

    The whole point of these regression tests is to exercise the
    symlink-resolution code path in ``tools/skills_provenance:classify``
    (specifically ``install_path.resolve()`` which dereferences symlinks).
    Silently falling back to a copy would mask the regression we're
    catching — the bug originally presented on systems where the profile
    copy WAS a symlink to the platform tree.
    """
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")


@pytest.fixture()
def symlink_builtin_skill_env(monkeypatch, tmp_path):
    """Build a hermetic symlinked-builtin fixture and patch module paths.

    Layout produced under ``tmp_path``::

        platform/skills/apple/macos-computer-use/SKILL.md   # canonical builtin
        profile/skills/apple/macos-computer-use            # SYMLINK -> above

    Returns a dict with the relevant absolute paths so individual tests
    can assert against them without re-deriving layout.
    """
    import tools.skills_hub as hub
    import tools.skills_provenance as prov
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    platform_root = tmp_path / "platform"
    profile_root = tmp_path / "profile"
    platform_skills = platform_root / "skills"
    profile_skills = profile_root / "skills"
    category = "apple"
    skill_name = "macos-computer-use"

    # Canonical builtin copy in the platform-level skills dir.
    builtin_dir = platform_skills / category / skill_name
    builtin_dir.mkdir(parents=True)
    (builtin_dir / "SKILL.md").write_text(
        "---\n"
        "name: macos-computer-use\n"
        "version: 1.0.0\n"
        "status: production\n"
        "description: Mac desktop computer-use skill.\n"
        "---\n\n"
        "Skill body.\n",
        encoding="utf-8",
    )

    # Symlink from the profile copy to the platform builtin.
    profile_skill_dir = profile_skills / category / skill_name
    profile_skill_dir.parent.mkdir(parents=True)
    _require_symlink_support(builtin_dir, profile_skill_dir)
    assert profile_skill_dir.is_symlink(), (
        f"fixture setup failed — expected a symlink at {profile_skill_dir}, "
        f"got a regular {'directory' if profile_skill_dir.is_dir() else 'path'}"
    )

    # Per-profile .provenance registry starts empty so we exercise the
    # heuristic classify() path (the production scenario that broke).
    provenance_file = profile_skills / ".provenance"

    # Patch every module-level path the listing code reads from.
    monkeypatch.setattr(hub, "SKILLS_DIR", profile_skills)
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", profile_skills)
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", profile_skills)
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", profile_skills / ".bundled_manifest")
    monkeypatch.setattr(prov, "PROFILE_SKILLS_DIR", profile_skills)
    monkeypatch.setattr(prov, "HERMES_HOME", profile_root)
    monkeypatch.setattr(prov, "PROVENANCE_FILE", provenance_file)
    monkeypatch.setattr(prov, "_platform_skills_dir", lambda: platform_skills)

    # Stub the hub lock, the bundled manifest reader, and the registry
    # writers so do_list doesn't touch disk beyond tmp_path.
    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([]))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})
    monkeypatch.setattr(prov, "_read_provenance_file", lambda path=None: {})
    monkeypatch.setattr(prov, "record", lambda *a, **kw: None)
    monkeypatch.setattr(prov, "record_many", lambda *a, **kw: None)

    # Stub _find_all_skills so the test doesn't depend on filesystem walk
    # behavior — we already constructed the layout by hand.
    skill_record = {
        "name": skill_name,
        "category": category,
        "description": "Mac desktop computer-use skill.",
        "install_path": profile_skill_dir,
    }
    monkeypatch.setattr(
        skills_tool,
        "_find_all_skills",
        lambda **_kwargs: [skill_record],
    )

    return {
        "tmp_path": tmp_path,
        "platform_root": platform_root,
        "profile_root": profile_root,
        "platform_skills": platform_skills,
        "profile_skills": profile_skills,
        "builtin_dir": builtin_dir,
        "profile_skill_dir": profile_skill_dir,
        "skill_name": skill_name,
        "skill_record": skill_record,
    }


def _capture_do_list(source_filter: str = "all", show_provenance: bool = False) -> str:
    """Run do_list into an in-memory Rich console and return the rendered output.

    Width is generous (400 cols) so long tmp_path absolute paths from
    nested pytest tempdirs don't get truncated to ``…`` and break
    substring assertions.
    """
    sink = io.StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None, width=400)
    do_list(source_filter=source_filter, console=console, show_provenance=show_provenance)
    return sink.getvalue()


def _row_for(output: str, skill_name: str) -> str:
    """Return the Rich table data row containing ``skill_name``."""
    for line in output.splitlines():
        if line.startswith("│") and skill_name in line:
            return line
    return ""


# ─────────────────────────────────────────────────────────────────────────
# Test 1 — function-level: Source=builtin, Trust=builtin (not local)
# ─────────────────────────────────────────────────────────────────────────


def test_do_list_reports_builtin_for_symlinked_profile_copy(symlink_builtin_skill_env):
    """The bug regression: macos-computer-use must be builtin, not local.

    Reproduces the original audit finding — when ``profile/skills/apple/
    macos-computer-use`` is a symlink to the platform-level builtin,
    ``hermes skills list`` must label it Source=builtin, Trust=builtin.

    The profile_skill_dir is asserted to be a real symlink by the fixture
    (which skips the test rather than silently falling back to a copy),
    so this exercises ``tools/skills_provenance.classify`` Case 1
    (resolved path under platform skills dir) — the symlink-resolution
    code path that was the original failure mode.
    """
    profile_skill_dir = symlink_builtin_skill_env["profile_skill_dir"]
    assert profile_skill_dir.is_symlink(), (
        "fixture did not produce a real symlink — the regression test is "
        "not exercising the symlink-resolution code path"
    )

    output = _capture_do_list()

    row = _row_for(output, "macos-computer-use")
    assert row, (
        f"macos-computer-use row missing from do_list output:\n{output}"
    )
    cells = [c.strip() for c in row.strip("│").split("│")]

    # Column order from hermes_cli/skills_hub.py:do_list:
    #   Name | Category | Source | Trust | ...
    assert cells[2] == "builtin", (
        f"expected Source=builtin for symlinked builtin, got {cells[2]!r}. "
        f"This is the regression — the symlinked profile copy must NOT be "
        f"labeled 'local'. Full row: {row!r}"
    )
    assert "builtin" in cells[3], (
        f"expected Trust=builtin for symlinked builtin, got {cells[3]!r}. "
        f"Full row: {row!r}"
    )
    assert cells[2] != "local", "Source must not be 'local' for a builtin symlink"
    assert "local" not in cells[3] or cells[3] == "builtin", (
        f"Trust must not be 'local' for a builtin symlink, got {cells[3]!r}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Test 2 — function-level: --provenance column shows the platform origin
# ─────────────────────────────────────────────────────────────────────────


def test_do_list_provenance_flag_shows_origin_for_symlinked_builtin(
    symlink_builtin_skill_env,
):
    """`--provenance` must surface the install-origin path for a symlinked builtin.

    The Provenance column reveals WHERE the skill is installed on disk so
    an operator can audit why a skill is labeled builtin/builtin from the
    CLI alone. For a symlinked builtin the install_origin must point at
    the profile-side path (the symlink itself) AND that path must resolve
    to the platform-level builtin — i.e. the user can follow the path and
    find the canonical source. This is what makes the labeling auditable
    from the CLI.
    """
    builtin_dir = symlink_builtin_skill_env["builtin_dir"]
    profile_skill_dir = symlink_builtin_skill_env["profile_skill_dir"]

    output = _capture_do_list(show_provenance=True)

    # The header gains a Provenance column when --provenance is on.
    assert "Provenance" in output, (
        f"--provenance flag did not add a Provenance column:\n{output}"
    )

    row = _row_for(output, "macos-computer-use")
    assert row, (
        f"macos-computer-use row missing from --provenance output:\n{output}"
    )

    # The profile-side install path must appear in the Provenance column
    # — that is the path the user typed / has on disk in their profile
    # skills dir. The column may also surface the resolved platform path,
    # but the profile path is what the user can ls/inspect directly.
    assert str(profile_skill_dir) in output, (
        f"Provenance column did not include the profile install path "
        f"{str(profile_skill_dir)!r}. Full output:\n{output}"
    )

    # The surfaced path must resolve to the platform builtin — this is
    # the deeper invariant the column promises: "follow this path and
    # you'll reach the canonical content". Whether the renderer shows
    # the symlink or its resolved target, resolving it must land on the
    # platform builtin dir.
    surfaced_paths = _extract_provenance_paths(output)
    assert surfaced_paths, (
        f"Could not parse any provenance path out of the rendered table:\n{output}"
    )
    for surfaced in surfaced_paths:
        resolved = Path(surfaced).resolve()
        assert resolved == builtin_dir.resolve(), (
            f"Provenance column surfaced {surfaced!r} which resolves to "
            f"{resolved} but should reach the platform builtin at "
            f"{builtin_dir.resolve()}. Full output:\n{output}"
        )

    # And the labels must remain Source=builtin / Trust=builtin — adding
    # --provenance must never change the labeling.
    cells = [c.strip() for c in row.strip("│").split("│")]
    assert cells[2] == "builtin", (
        f"Source drifted away from 'builtin' when --provenance is on, "
        f"got {cells[2]!r}. Full row: {row!r}"
    )
    assert "builtin" in cells[3], (
        f"Trust drifted away from 'builtin' when --provenance is on, "
        f"got {cells[3]!r}. Full row: {row!r}"
    )


def _extract_provenance_paths(rendered: str) -> list[str]:
    """Pull probable absolute filesystem paths out of the Provenance column.

    The Rich table renders the Provenance column as plain text between
    ``│`` separators. We look for ``/``-prefixed tokens inside data rows
    (rows that start and end with ``│``) and return them as candidates.
    """
    paths = []
    for line in rendered.splitlines():
        if not (line.startswith("│") and line.endswith("│")):
            continue
        # Strip the outer box-drawing chars, then split on remaining ones.
        inner = line.strip("│").strip()
        for cell in inner.split("│"):
            cell = cell.strip()
            if cell.startswith("/"):
                paths.append(cell)
    return paths


# ─────────────────────────────────────────────────────────────────────────
# Test 3 — CLI-level: `hermes skills list --provenance` end-to-end
# ─────────────────────────────────────────────────────────────────────────


def test_cli_skills_list_provenance_flag_for_symlinked_builtin(
    monkeypatch, symlink_builtin_skill_env
):
    """End-to-end CLI regression: ``hermes skills list --provenance``.

    Drives ``hermes_cli.main.main()`` with argv mocked, captures stdout,
    and asserts that the rendered table contains the symlinked builtin
    labeled builtin/builtin AND shows the install-origin path (resolving
    to the platform builtin) in the Provenance column.

    This catches regressions in the CLI wiring itself (subparser
    registration, ``show_provenance`` plumbing through ``cmd_skills``,
    Rich table column ordering) that the function-level tests cannot.
    """
    builtin_dir = symlink_builtin_skill_env["builtin_dir"]
    profile_skill_dir = symlink_builtin_skill_env["profile_skill_dir"]
    profile_root = symlink_builtin_skill_env["profile_root"]

    # Redirect HERMES_HOME so the CLI's get_hermes_home() agrees with
    # the patched module-level paths.
    monkeypatch.setenv("HERMES_HOME", str(profile_root))

    # Capture skills_command to verify the --provenance flag arrived.
    captured: dict = {}

    def fake_skills_command(args):
        captured["provenance"] = getattr(args, "provenance", False)
        captured["source"] = getattr(args, "source", None)
        # Drive the real do_list so we exercise the actual rendering code.
        sink = io.StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None, width=400)
        do_list(
            source_filter=getattr(args, "source", "all") or "all",
            console=console,
            show_provenance=getattr(args, "provenance", False),
        )
        captured["rendered"] = sink.getvalue()

    # The real CLI flow: hermes_cli.main -> cmd_skills -> skills_command.
    # We intercept the final hop and assert on what it received and rendered.
    monkeypatch.setattr("hermes_cli.skills_hub.skills_command", fake_skills_command)
    monkeypatch.setattr(sys, "argv", ["hermes", "skills", "list", "--provenance"])

    # Drive main() with stdout captured so any incidental writes don't
    # pollute the test output.
    buf = io.StringIO()
    with redirect_stdout(buf):
        from hermes_cli.main import main as cli_main

        cli_main()

    assert captured.get("provenance") is True, (
        f"--provenance flag did not reach skills_command. captured={captured!r}"
    )
    rendered = captured.get("rendered", "")
    assert rendered, "skills_command produced no rendered output"

    # Header column must be present.
    assert "Provenance" in rendered, (
        f"Provenance column missing from CLI-rendered output:\n{rendered}"
    )

    # The row must show the symlinked builtin as builtin/builtin.
    row = _row_for(rendered, "macos-computer-use")
    assert row, (
        f"macos-computer-use row missing from CLI --provenance output:\n{rendered}"
    )
    cells = [c.strip() for c in row.strip("│").split("│")]
    assert cells[2] == "builtin", (
        f"CLI: expected Source=builtin for symlinked builtin, got "
        f"{cells[2]!r}. Full row: {row!r}"
    )
    assert "builtin" in cells[3], (
        f"CLI: expected Trust=builtin for symlinked builtin, got "
        f"{cells[3]!r}. Full row: {row!r}"
    )

    # The Provenance column must surface the install-origin path, and
    # following it must land on the platform builtin (whether the
    # renderer shows the symlink or its resolved target, the canonical
    # content lives at builtin_dir).
    assert str(profile_skill_dir) in rendered, (
        f"CLI: Provenance column did not include the profile install path "
        f"{str(profile_skill_dir)!r}. Full output:\n{rendered}"
    )
    surfaced_paths = _extract_provenance_paths(rendered)
    assert surfaced_paths, (
        f"CLI: could not parse any provenance path out of the rendered "
        f"table:\n{rendered}"
    )
    for surfaced in surfaced_paths:
        resolved = Path(surfaced).resolve()
        assert resolved == builtin_dir.resolve(), (
            f"CLI: Provenance column surfaced {surfaced!r} which resolves "
            f"to {resolved} but should reach the platform builtin at "
            f"{builtin_dir.resolve()}. Full output:\n{rendered}"
        )
