"""CLI integration tests for vetting (t_8a86fc9c).

Covers:

* ``hermes skills vet`` subparser is registered, has the right flags,
  and routes to ``do_vet`` in the ``skills_command`` router.
* ``hermes skills list`` has the new ``--vetted`` / ``--unvetted`` flags.
* ``hermes skills install`` has the new ``--require-vetting`` flag.
* ``do_vet`` end-to-end on a fixture skill: dry-run vs. real stamp,
  --all over a fixture tree, missing skill name handling, bad reviewer
  rejection.
* ``do_list`` end-to-end: --vetted filters, --unvetted filters, and
  the unvetted warning banner is printed when neither flag is set
  and at least one enabled local skill is unvetted.

These tests are hermes_cli-layer tests; they use ``tmp_path`` for
hermetic fixtures and patch ``tools.skills_tool._find_all_skills`` to
point at the tmp_path tree so the test never touches the user's real
``~/.hermes/skills/`` install.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest


# ── Subparser registration ───────────────────────────────────────────────


def _build_full_parser():
    """Build the full hermes CLI parser the same way main() does."""
    # Skills is registered in main() after build_top_level_parser, so
    # we replicate that flow here.  Importing main() is heavy (it
    # touches every subsystem) so we just call the parser builders
    # we need.
    from hermes_cli._parser import build_top_level_parser
    from hermes_cli.subcommands.skills import build_skills_parser

    parser, subparsers, _ = build_top_level_parser()
    # The signature in skills.py requires a callable for cmd_skills;
    # pass a no-op since the parser tests only care about the
    # argument shape, not the runtime dispatch.
    build_skills_parser(subparsers, cmd_skills=lambda args: None)
    return parser


def test_skills_subparser_has_vet_command():
    parser = _build_full_parser()
    # The skills subparser action must be one we added; verify by
    # parsing a complete command and reading the action off the
    # namespace.  (We can't pass --help because that exits the
    # process; the parser's own ``skills_action`` dest is the
    # authoritative marker.)
    args = parser.parse_args(
        ["skills", "vet", "agent-handoff", "--by", "wags-reviewer"]
    )
    assert args.skills_action == "vet"


def test_skills_vet_accepts_all_required_flags():
    parser = _build_full_parser()
    args = parser.parse_args(
        ["skills", "vet", "agent-handoff", "--by", "wags-reviewer"]
    )
    assert args.skills_action == "vet"
    assert args.name == "agent-handoff"
    assert args.by == "wags-reviewer"
    assert args.dry_run is False
    assert args.all_skills is False


def test_skills_vet_accepts_dry_run_flag():
    parser = _build_full_parser()
    args = parser.parse_args(
        ["skills", "vet", "agent-handoff", "--by", "wags-reviewer", "--dry-run"]
    )
    assert args.dry_run is True


def test_skills_vet_accepts_all_flag():
    parser = _build_full_parser()
    args = parser.parse_args(["skills", "vet", "--all", "--by", "wags-reviewer"])
    assert args.all_skills is True
    assert args.name is None


def test_skills_list_has_vetted_flag():
    parser = _build_full_parser()
    args = parser.parse_args(["skills", "list", "--vetted"])
    assert args.vetted is True
    assert args.unvetted is False


def test_skills_list_has_unvetted_flag():
    parser = _build_full_parser()
    args = parser.parse_args(["skills", "list", "--unvetted"])
    assert args.unvetted is True
    assert args.vetted is False


def test_skills_install_has_require_vetting_flag():
    parser = _build_full_parser()
    args = parser.parse_args(
        ["skills", "install", "some-skill", "--require-vetting"]
    )
    assert args.require_vetting is True


# ── do_vet() end-to-end ───────────────────────────────────────────────────


def _write_vettable_skill(skill_dir: Path, name: str, *, vetted: bool = False,
                          broken: bool = False) -> Path:
    """Drop a SKILL.md + optional companion file at ``skill_dir/<name>``.

    Returns the SKILL.md path.
    """
    skill_dir = skill_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "EXAMPLES.md").write_text("Examples here.\n" * 30)
    body = dedent(f"""\
        ---
        name: {name}
        version: 1.0.0
        status: vetted
        description: A test skill used by the CLI vetting tests.
        author: Hermes Agent
        ---

        # {name}

        This body is long enough (over 200 characters after dedent) to
        pass the content-quality validator, includes a markdown
        heading, and references [examples](EXAMPLES.md) that exist on
        disk so the link-existence check passes.
        """)
    if vetted:
        body = body.replace(
            "---\n",
            '---\nvetted_at: "2026-06-23T18:30:00Z"\nvetted_by: "wags-reviewer"\n',
            1,
        )
    if broken:
        # Strip the body down to the bare minimum so all four
        # validators fail: frontmatter (name/dir mismatch by
        # removing ``name:``), content_quality (description short
        # and body too short), links_files (link to a non-existent
        # companion file).
        body = dedent("""\
            ---
            version: 1.0.0
            status: vetted
            description: short
            author: Hermes Agent
            ---

            # Body

            too short
            """)
    path = skill_dir / "SKILL.md"
    path.write_text(body)
    return path


class TestDoVet:
    """End-to-end tests for the ``do_vet`` CLI handler."""

    def _stub_find_all(self, skill_dir: Path):
        """Return a list-of-dicts that mimics ``_find_all_skills`` output."""
        def _fake_find_all(*, skip_disabled: bool = False):
            out = []
            for child in sorted(skill_dir.iterdir()):
                if not child.is_dir():
                    continue
                skill_md = child / "SKILL.md"
                if not skill_md.is_file():
                    continue
                out.append({
                    "name": child.name,
                    "category": "",
                    "install_path": str(child),
                })
            return out
        return _fake_find_all

    def test_dry_run_does_not_stamp(self, tmp_path):
        from hermes_cli import skills_hub
        path = _write_vettable_skill(tmp_path, "my-skill")
        with patch("tools.skills_tool._find_all_skills",
                   side_effect=self._stub_find_all(tmp_path)):
            count = skills_hub.do_vet(
                name="my-skill", by="wags-reviewer", dry_run=True
            )
        assert count == 1
        # File was NOT modified.
        assert "vetted_at" not in path.read_text()

    def test_real_stamp_writes_frontmatter(self, tmp_path):
        from hermes_cli import skills_hub
        path = _write_vettable_skill(tmp_path, "my-skill")
        with patch("tools.skills_tool._find_all_skills",
                   side_effect=self._stub_find_all(tmp_path)):
            count = skills_hub.do_vet(
                name="my-skill", by="wags-reviewer",
                when="2026-06-23T18:30:00Z"
            )
        assert count == 1
        # The stamp landed.
        assert "vetted_at" in path.read_text()
        assert "wags-reviewer" in path.read_text()
        assert "2026-06-23T18:30:00Z" in path.read_text()

    def test_broken_skill_not_stamped(self, tmp_path):
        from hermes_cli import skills_hub
        path = _write_vettable_skill(tmp_path, "broken-skill", broken=True)
        with patch("tools.skills_tool._find_all_skills",
                   side_effect=self._stub_find_all(tmp_path)):
            count = skills_hub.do_vet(
                name="broken-skill", by="wags-reviewer"
            )
        assert count == 0
        assert "vetted_at" not in path.read_text()

    def test_unknown_skill_returns_zero(self, tmp_path):
        from hermes_cli import skills_hub
        with patch("tools.skills_tool._find_all_skills",
                   side_effect=self._stub_find_all(tmp_path)):
            count = skills_hub.do_vet(name="not-installed", by="wags-reviewer")
        assert count == 0

    def test_no_by_no_dry_run_returns_zero(self, tmp_path):
        from hermes_cli import skills_hub
        path = _write_vettable_skill(tmp_path, "my-skill")
        with patch("tools.skills_tool._find_all_skills",
                   side_effect=self._stub_find_all(tmp_path)):
            count = skills_hub.do_vet(name="my-skill", by="", dry_run=False)
        # Refuses to stamp without a reviewer identity.
        assert count == 0
        assert "vetted_at" not in path.read_text()

    def test_bad_reviewer_returns_zero(self, tmp_path):
        from hermes_cli import skills_hub
        path = _write_vettable_skill(tmp_path, "my-skill")
        with patch("tools.skills_tool._find_all_skills",
                   side_effect=self._stub_find_all(tmp_path)):
            count = skills_hub.do_vet(
                name="my-skill", by="has spaces", dry_run=True
            )
        assert count == 0

    def test_all_vets_every_skill(self, tmp_path):
        from hermes_cli import skills_hub
        a = _write_vettable_skill(tmp_path, "skill-a")
        b = _write_vettable_skill(tmp_path, "skill-b")
        with patch("tools.skills_tool._find_all_skills",
                   side_effect=self._stub_find_all(tmp_path)):
            count = skills_hub.do_vet(
                name=None, by="wags-reviewer", all_skills=True
            )
        assert count == 2
        assert "vetted_at" in a.read_text()
        assert "vetted_at" in b.read_text()

    def test_all_with_one_broken_stamps_only_the_clean(self, tmp_path):
        from hermes_cli import skills_hub
        good = _write_vettable_skill(tmp_path, "good")
        bad = _write_vettable_skill(tmp_path, "bad", broken=True)
        with patch("tools.skills_tool._find_all_skills",
                   side_effect=self._stub_find_all(tmp_path)):
            count = skills_hub.do_vet(
                name=None, by="wags-reviewer", all_skills=True
            )
        # Only the good one stamps.
        assert count == 1
        assert "vetted_at" in good.read_text()
        assert "vetted_at" not in bad.read_text()


# ── --require-vetting on install ─────────────────────────────────────────


def test_require_vetting_rejects_unstamped_skill():
    """The --require-vetting gate fires before any filesystem write."""
    from hermes_cli import skills_hub

    # Build a fake bundle object so we can drive do_install() up to the
    # vetting check without going through the real installer.
    class _FakeBundle:
        name = "unvetted-skill"
        source = "hub"
        files = {
            "SKILL.md": dedent("""\
                ---
                name: unvetted-skill
                version: 1.0.0
                status: vetted
                description: A perfectly valid description for testing.
                author: Hermes Agent
                ---

                # Body

                Body content long enough to clear the 200-character
                minimum, includes a heading, and is otherwise clean.
                """).encode("utf-8")
        }

    class _FakeMeta:
        name = "unvetted-skill"
        identifier = "test/unvetted-skill"
        source = "hub"
        trust_level = "community"
        description = "A perfectly valid description for testing."
        extra: dict = {}

    fake_meta = _FakeMeta()
    fake_bundle = _FakeBundle()

    # Patch the network-touching helpers so do_install() can be
    # exercised without a real registry round-trip.
    class _NoopConsole:
        def print(self, *a, **k):
            pass

    with patch("tools.skills_hub.GitHubAuth") as Auth, \
         patch("tools.skills_hub.create_source_router"), \
         patch("tools.skills_hub.ensure_hub_dirs"), \
         patch("hermes_cli.skills_hub._resolve_source_meta_and_bundle",
               return_value=(fake_meta, fake_bundle, None)), \
         patch("tools.skills_hub.HubLockFile") as Lock, \
         patch("tools.skills_hub.append_audit_log"), \
         patch("tools.skills_hub.quarantine_bundle") as Quarantine, \
         patch("tools.skills_hub.install_from_quarantine"), \
         patch.object(skills_hub, "_console", _NoopConsole()):
        # Pretend the skill is NOT already installed so we proceed
        # to the vetting check.
        Lock.return_value.get_installed.return_value = None
        skills_hub.do_install(
            "test/unvetted-skill",
            require_vetting=True,
            console=_NoopConsole(),
        )
        # The vetting check should have fired BEFORE quarantine_bundle.
        # If quarantine_bundle was called, the check did not fire.
        Quarantine.assert_not_called()


def test_require_vetting_accepts_vetted_skill():
    """A vetted skill passes the gate; quarantine runs."""
    from hermes_cli import skills_hub

    class _FakeBundle:
        name = "vetted-skill"
        source = "hub"
        files = {
            "SKILL.md": dedent("""\
                ---
                name: vetted-skill
                version: 1.0.0
                status: vetted
                description: A perfectly valid description for testing.
                author: Hermes Agent
                vetted_at: "2026-06-23T18:30:00Z"
                vetted_by: "wags-reviewer"
                ---

                # Body

                Body content long enough to clear the 200-character
                minimum, includes a heading, and is otherwise clean.
                """).encode("utf-8")
        }

    class _FakeMeta:
        name = "vetted-skill"
        identifier = "test/vetted-skill"
        source = "hub"
        trust_level = "community"
        description = "A perfectly valid description for testing."
        extra: dict = {}

    fake_meta = _FakeMeta()
    fake_bundle = _FakeBundle()

    class _NoopConsole:
        def print(self, *a, **k):
            pass

    quarantine_called = {"n": 0}

    def _fake_quarantine(bundle):
        quarantine_called["n"] += 1
        # Return a Path the rest of the flow can use.  We won't let
        # the test progress past quarantine because we don't have a
        # real install dir, but we want to confirm the vetting gate
        # let the request through.
        raise SystemExit(0)  # short-circuit the install

    with patch("tools.skills_hub.GitHubAuth"), \
         patch("tools.skills_hub.create_source_router"), \
         patch("tools.skills_hub.ensure_hub_dirs"), \
         patch("hermes_cli.skills_hub._resolve_source_meta_and_bundle",
               return_value=(fake_meta, fake_bundle, None)), \
         patch("tools.skills_hub.HubLockFile") as Lock, \
         patch("tools.skills_hub.append_audit_log"), \
         patch("tools.skills_hub.quarantine_bundle", side_effect=_fake_quarantine), \
         patch("tools.skills_hub.install_from_quarantine"), \
         patch.object(skills_hub, "_console", _NoopConsole()):
        Lock.return_value.get_installed.return_value = None
        with pytest.raises(SystemExit):
            skills_hub.do_install(
                "test/vetted-skill",
                require_vetting=True,
                console=_NoopConsole(),
            )
        # Vetting gate passed → quarantine was reached.
        assert quarantine_called["n"] == 1


# ── do_list vetting filter + warning banner ──────────────────────────────


def test_list_warning_banner_appears_when_unvetted_enabled_exists(tmp_path):
    """do_list prints a warning when at least one enabled local skill is
    unvetted and neither --vetted nor --unvetted is set.
    """
    from hermes_cli import skills_hub

    # Set up a tiny skill tree.
    skill_a = tmp_path / "skill-a"
    skill_a.mkdir()
    (skill_a / "EXAMPLES.md").write_text("Examples here.\n" * 30)
    (skill_a / "SKILL.md").write_text(dedent("""\
        ---
        name: skill-a
        version: 1.0.0
        status: vetted
        description: A perfectly valid description for testing.
        author: Hermes Agent
        ---

        # Skill A

        This body is long enough to clear the 200-character minimum
        that the content-quality validator enforces, includes a
        markdown heading, and is otherwise clean.
        """))

    printed = []

    class _CapturingConsole:
        def print(self, *args, **kwargs):
            # The Rich Console.print method has a rich-specific protocol
            # (with `end` / `sep` keyword args) that the table printer
            # uses.  Coerce everything to plain str so the assertions
            # are simple string contains.
            for a in args:
                s = str(a)
                # Skip rich markup wrapping so the assertions don't
                # have to know about [green][/] etc.
                s = s.replace("[green]", "").replace("[/]", "")
                s = s.replace("[red]", "").replace("[/]", "")
                s = s.replace("[dim]", "").replace("[/]", "")
                s = s.replace("[bold green]", "").replace("[/]", "")
                s = s.replace("[bold red]", "").replace("[/]", "")
                s = s.replace("[bold yellow]", "").replace("[/]", "")
                printed.append(s)

    with patch("tools.skills_tool._find_all_skills",
               return_value=[{"name": "skill-a", "category": "",
                              "install_path": str(skill_a)}]), \
         patch("agent.skill_utils.get_disabled_skill_names",
               return_value=set()), \
         patch("tools.skills_hub.HubLockFile") as Lock, \
         patch("tools.skills_hub.ensure_hub_dirs"), \
         patch("tools.skills_provenance._read_provenance_file",
               return_value={}), \
         patch("tools.skills_provenance.trust_for", return_value="local"), \
         patch("tools.skills_sync._read_manifest", return_value={}):
        Lock.return_value.list_installed.return_value = []
        skills_hub.do_list(console=_CapturingConsole())

    full = " ".join(printed)
    # Warning banner appeared.
    assert "unvetted" in full.lower()
    assert "skill-a" in full


def test_list_unvetted_filter_excludes_vetted(tmp_path):
    """``--unvetted`` shows only unvetted skills."""
    from hermes_cli import skills_hub
    from rich.console import Console

    # Vetted skill.
    vetted = tmp_path / "vetted-skill"
    vetted.mkdir()
    (vetted / "EXAMPLES.md").write_text("Examples here.\n" * 30)
    (vetted / "SKILL.md").write_text(dedent("""\
        ---
        name: vetted-skill
        version: 1.0.0
        status: vetted
        description: A perfectly valid description for testing.
        author: Hermes Agent
        vetted_at: "2026-06-23T18:30:00Z"
        vetted_by: "wags-reviewer"
        ---

        # Body

        Long enough body to clear 200 characters minimum requirement
        and includes a markdown heading so the content-quality check
        passes when the vetting flow runs end-to-end.
        """))

    # Unvetted skill.
    unvetted = tmp_path / "unvetted-skill"
    unvetted.mkdir()
    (unvetted / "EXAMPLES.md").write_text("Examples here.\n" * 30)
    (unvetted / "SKILL.md").write_text(dedent("""\
        ---
        name: unvetted-skill
        version: 1.0.0
        status: vetted
        description: A perfectly valid description for testing.
        author: Hermes Agent
        ---

        # Body

        Long enough body to clear 200 characters minimum requirement
        and includes a markdown heading so the content-quality check
        passes when the vetting flow runs end-to-end.
        """))

    # Use rich.Console.capture so we can see the rendered table.
    console = Console(record=True, force_terminal=False, width=200)

    with patch("tools.skills_tool._find_all_skills",
               return_value=[
                   {"name": "vetted-skill", "category": "",
                    "install_path": str(vetted)},
                   {"name": "unvetted-skill", "category": "",
                    "install_path": str(unvetted)},
               ]), \
         patch("agent.skill_utils.get_disabled_skill_names",
               return_value=set()), \
         patch("tools.skills_hub.HubLockFile") as Lock, \
         patch("tools.skills_hub.ensure_hub_dirs"), \
         patch("tools.skills_provenance._read_provenance_file",
               return_value={}), \
         patch("tools.skills_provenance.trust_for", return_value="local"), \
         patch("tools.skills_sync._read_manifest", return_value={}):
        Lock.return_value.list_installed.return_value = []
        skills_hub.do_list(unvetted_only=True, console=console)

    full = console.export_text()
    # Look only at the table block (between "(unvetted only)" and the
    # summary line).  The back-filled provenance warning may still
    # mention both names — that's a separate signal.
    title_idx = full.find("(unvetted only)")
    summary_idx = full.find("hub-installed", title_idx)
    table_section = full[title_idx:summary_idx]
    # Use full-word checks to avoid the assertion confusing the test
    # skill name "unvetted-skill" with "vetted-skill" as substrings.
    import re
    assert re.search(r"\bunvetted-skill\b", table_section)
    assert not re.search(r"\bvetted-skill\b", table_section)


def test_list_vetted_filter_excludes_unvetted(tmp_path):
    """``--vetted`` shows only vetted skills."""
    from hermes_cli import skills_hub
    from rich.console import Console

    vetted = tmp_path / "vetted-skill"
    vetted.mkdir()
    (vetted / "EXAMPLES.md").write_text("Examples here.\n" * 30)
    (vetted / "SKILL.md").write_text(dedent("""\
        ---
        name: vetted-skill
        version: 1.0.0
        status: vetted
        description: A perfectly valid description for testing.
        author: Hermes Agent
        vetted_at: "2026-06-23T18:30:00Z"
        vetted_by: "wags-reviewer"
        ---

        # Body

        Long enough body to clear 200 characters minimum requirement
        and includes a markdown heading so the content-quality check
        passes when the vetting flow runs end-to-end.
        """))

    unvetted = tmp_path / "unvetted-skill"
    unvetted.mkdir()
    (unvetted / "EXAMPLES.md").write_text("Examples here.\n" * 30)
    (unvetted / "SKILL.md").write_text(dedent("""\
        ---
        name: unvetted-skill
        version: 1.0.0
        status: vetted
        description: A perfectly valid description for testing.
        author: Hermes Agent
        ---

        # Body

        Long enough body to clear 200 characters minimum requirement
        and includes a markdown heading so the content-quality check
        passes when the vetting flow runs end-to-end.
        """))

    console = Console(record=True, force_terminal=False, width=200)

    with patch("tools.skills_tool._find_all_skills",
               return_value=[
                   {"name": "vetted-skill", "category": "",
                    "install_path": str(vetted)},
                   {"name": "unvetted-skill", "category": "",
                    "install_path": str(unvetted)},
               ]), \
         patch("agent.skill_utils.get_disabled_skill_names",
               return_value=set()), \
         patch("tools.skills_hub.HubLockFile") as Lock, \
         patch("tools.skills_hub.ensure_hub_dirs"), \
         patch("tools.skills_provenance._read_provenance_file",
               return_value={}), \
         patch("tools.skills_provenance.trust_for", return_value="local"), \
         patch("tools.skills_sync._read_manifest", return_value={}):
        Lock.return_value.list_installed.return_value = []
        skills_hub.do_list(vetted_only=True, console=console)

    full = console.export_text()
    title_idx = full.find("(vetted only)")
    summary_idx = full.find("hub-installed", title_idx)
    table_section = full[title_idx:summary_idx]
    import re
    assert re.search(r"\bvetted-skill\b", table_section)
    assert not re.search(r"\bunvetted-skill\b", table_section)
