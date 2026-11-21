"""Contract: the proactive-agent-souls skill validates a SOUL.md proactivity block.

The skill's whole promise is that a proactivity contract is mechanically checkable, so this
file covers the checker: every finding code it can emit is asserted here, the shipped
template must fail as an unfilled scaffold and pass once filled, the frontmatter obeys the
authoring hardline, and the optional-skill source must publish every file the skill ships.
Prose is not asserted — the files and the exit codes are.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO / "optional-skills" / "autonomous-ai-agents" / "proactive-agent-souls"
SCRIPT = SKILL_DIR / "scripts" / "check_proactivity_contract.py"
SKILL_MD = SKILL_DIR / "SKILL.md"
TEMPLATE = SKILL_DIR / "templates" / "proactivity-section.md"
LOG_TEMPLATE = SKILL_DIR / "templates" / "observations-log.md"
TIER_HEADING = re.compile(r"^### Tier (\d+)\b.*$", re.MULTILINE)
MARKETING = re.compile(
    r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
    re.IGNORECASE,
)


def _load_checker():
    name = "check_proactivity_contract"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec and spec.loader, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass field resolution looks the module up in sys.modules.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _codes(findings) -> list[str]:
    return sorted(f.code for f in findings)


def _tier_spans(text: str) -> dict[int, tuple[int, int]]:
    """{tier: (start, end)} for each ``### Tier N`` block, end exclusive."""
    headings = list(TIER_HEADING.finditer(text))
    spans = {}
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        spans.setdefault(int(match.group(1)), (match.start(), end))
    return spans


def _tier_block(text: str, tier: int) -> str:
    start, end = _tier_spans(text)[tier]
    return text[start:end]


def _filled_contract() -> str:
    """The shipped template with every placeholder replaced — a real contract."""
    text = TEMPLATE.read_text(encoding="utf-8-sig")
    for placeholder, value in (
        ("<observations-path>", "~/notes/OBSERVATIONS.md"),
        ("<rules-path>", "~/notes/PROACTIVITY_RULES.md"),
        ("<rule-1>", "two consecutive failed scheduled runs"),
        ("<rule-2>", "a watched price crosses the threshold"),
        ("<what the recipient gets to decide>", "whether to intervene tonight"),
        ("<timezone>", "Europe/Lisbon"),
        ("<N>", "60"),
    ):
        text = text.replace(placeholder, value)
    text = text.replace("<HH:MM>", "23:00", 1).replace("<HH:MM>", "07:30", 1)
    return text.replace(
        "Replace\nevery `<…>` placeholder before saving.",
        "This block is filled in and live.",
    )


GOOD = _filled_contract()


# --------------------------------------------------------------------------- what ships


def test_skill_ships_the_checker_templates_and_docs():
    for path in (SKILL_MD, SCRIPT, TEMPLATE, LOG_TEMPLATE):
        assert path.is_file(), f"{path} missing from the skill"
    log_rows = [
        line
        for line in LOG_TEMPLATE.read_text(encoding="utf-8-sig").splitlines()
        if line.startswith("20")
    ]
    assert len(log_rows) == 3, "the observations template must show worked rows for open, confirmed and actioned"


def test_frontmatter_obeys_the_authoring_hardline():
    content = SKILL_MD.read_text(encoding="utf-8-sig")
    assert content.startswith("---\n"), "SKILL.md must open with frontmatter"
    frontmatter = content.split("---\n", 2)[1]
    fields = dict(re.findall(r"^(\w+):\s*(.+)$", frontmatter, re.MULTILINE))
    assert fields["name"] == SKILL_DIR.name, "frontmatter name must match the directory"
    description = fields["description"].strip().strip('"')
    assert len(description) <= 60, f"description is {len(description)} chars (hardline 60)"
    assert description.endswith("."), "description must end with a period"
    assert not MARKETING.search(description), "marketing word in description"
    assert "platforms: [linux, macos, windows]" in frontmatter, "the checker skill is cross-platform"
    assert fields["author"].split(",")[0].strip() != "Hermes Agent", "credit the human first"


# --------------------------------------------------------------------------- the contract


def test_a_filled_contract_has_no_findings():
    findings = CHECKER.check_text(GOOD)
    assert findings == [], [f.render(Path("SOUL.md")) for f in findings]


def test_shipped_template_is_rejected_as_an_unfilled_scaffold():
    codes = set(_codes(CHECKER.check_text(TEMPLATE.read_text(encoding="utf-8-sig"))))
    assert codes == {
        "observations-log-missing",
        "placeholder",
        "quiet-hours-incomplete",
        "rules-file-missing",
        "suppression-incomplete",
    }, codes


def test_missing_proactivity_section_is_reported():
    codes = _codes(CHECKER.check_text("# Hermes\n\nBe concise.\n"))
    assert "section-missing" in codes, codes


def test_missing_tier_is_reported():
    for tier in (2, 3, 4):
        without = GOOD.replace(_tier_block(GOOD, tier), "")
        assert "tier-missing" in _codes(CHECKER.check_text(without)), tier


def test_tiers_out_of_order_are_reported():
    start, end = _tier_spans(GOOD)[1]
    tier_one, without_one = GOOD[start:end], GOOD[:start] + GOOD[end:]
    third = without_one.index("### Tier 3")
    reordered = without_one[:third] + tier_one + without_one[third:]
    assert _codes(CHECKER.check_text(reordered)) == ["tier-order"]


def test_a_tier_defined_twice_is_reported():
    duplicated = GOOD + "\n### Tier 2 — Record, never send\n\n- a second copy of the same tier\n"
    assert "duplicate-tier" in _codes(CHECKER.check_text(duplicated))


def test_a_tier_with_no_body_is_reported():
    emptied = GOOD.replace(
        _tier_block(GOOD, 3), "### Tier 3 — Bounded self-heal (act, verify, stay quiet)\n\n"
    )
    assert "tier-empty" in _codes(CHECKER.check_text(emptied))


def test_trigger_register_is_required():
    no_triggers = re.sub(r"triggers?", "condition", GOOD, flags=re.IGNORECASE)
    assert "triggers-missing" in _codes(CHECKER.check_text(no_triggers))


def test_the_silence_marker_is_required():
    codes = _codes(CHECKER.check_text(GOOD.replace("`[SILENT]`", "nothing at all")))
    assert "silence-marker-missing" in codes, codes


def test_observations_log_path_is_required():
    codes = _codes(CHECKER.check_text(GOOD.replace("~/notes/OBSERVATIONS.md", "the usual place")))
    assert "observations-log-missing" in codes, codes


def test_operator_rules_file_path_is_required():
    codes = _codes(CHECKER.check_text(GOOD.replace("~/notes/PROACTIVITY_RULES.md", "the rules I wrote")))
    assert "rules-file-missing" in codes, codes


def test_message_quality_bar_is_advisory():
    findings = CHECKER.check_text(GOOD.replace("### Message quality (Tier 4 only)", "### Notes"))
    quality = [f for f in findings if f.code == "message-quality-missing"]
    assert quality and quality[0].severity == CHECKER.WARNING


def test_placeholder_left_in_the_file_is_reported():
    codes = _codes(
        CHECKER.check_text(GOOD + "\n- TODO: reconcile the log\n- <add the missing rule>\n")
    )
    assert codes == ["placeholder", "placeholder"], codes


def test_empty_identity_file_is_its_own_finding():
    assert _codes(CHECKER.check_text("   \n\n")) == ["empty"]


def test_quiet_hours_and_suppression_are_advisory():
    without_windows = "\n".join(
        line
        for line in GOOD.splitlines()
        if not line.strip().startswith(("**Quiet hours:**", "**Suppression:**"))
        and "suppress" not in line.lower()
    )
    findings = CHECKER.check_text(without_windows)
    assert {f.code for f in findings} == {"quiet-hours-missing", "suppression-missing"}
    assert all(f.severity == CHECKER.WARNING for f in findings), "advisory, not fatal"


def test_oversized_identity_file_warns():
    oversized = [f for f in CHECKER.check_text(GOOD, max_chars=100) if f.code == "soul-too-long"]
    assert oversized and oversized[0].severity == CHECKER.WARNING


def test_fenced_samples_do_not_satisfy_the_contract():
    """A fenced copy of the block is documentation; the contract outside it must stand."""
    fenced = "```markdown\n" + GOOD.strip() + "\n```\n"
    assert "section-missing" in _codes(CHECKER.check_text(fenced))


# --------------------------------------------------------------------------- the CLI


def test_cli_reports_ok_and_findings_with_distinct_exit_codes(tmp_path, capsys):
    good = tmp_path / "SOUL.md"
    good.write_text("# Hermes\n\n" + GOOD, encoding="utf-8")
    assert CHECKER.main([str(good)]) == 0
    assert "OK — proactivity contract intact" in capsys.readouterr().out

    bad = tmp_path / "broken" / "SOUL.md"
    bad.parent.mkdir()
    bad.write_text("# Hermes\n\nNo contract here.\n", encoding="utf-8")
    assert CHECKER.main([str(bad), str(good)]) == 1
    out = capsys.readouterr().out
    assert "section-missing" in out and f"{bad}:0:" in out


def test_cli_json_is_machine_readable(tmp_path, capsys):
    target = tmp_path / "SOUL.md"
    target.write_text(GOOD, encoding="utf-8")
    assert CHECKER.main([str(target), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["errors"] == 0 and payload["warnings"] == 0
    assert payload["files"][0]["path"] == str(target)
    assert payload["files"][0]["chars"] == len(GOOD)


def test_cli_strict_promotes_warnings(tmp_path, capsys):
    target = tmp_path / "SOUL.md"
    target.write_text(
        GOOD.replace("**Quiet hours:** 23:00–07:30 Europe/Lisbon", "Quiet: never"), encoding="utf-8"
    )
    assert CHECKER.main([str(target)]) == 0
    capsys.readouterr()
    assert CHECKER.main([str(target), "--strict"]) == 1


def test_cli_resolves_default_and_directory_targets(tmp_path, monkeypatch, capsys):
    home = tmp_path / "profile-home"
    home.mkdir()
    (home / "SOUL.md").write_text(GOOD, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert CHECKER.resolve_targets([]) == [home / "SOUL.md"], "default target is $HERMES_HOME/SOUL.md"
    assert CHECKER.main([str(home)]) == 0
    assert "OK — proactivity contract intact" in capsys.readouterr().out


def test_cli_missing_file_is_a_finding_not_a_crash(tmp_path, capsys):
    missing = tmp_path / "nope" / "SOUL.md"
    assert CHECKER.main([str(missing)]) == 1
    assert "file-missing" in capsys.readouterr().out


def test_checker_stays_stdlib_and_cross_platform():
    text = SCRIPT.read_text(encoding="utf-8-sig")
    # Fragments, so this file never contains the primitive it forbids.
    forbidden = ("fcntl", "termios", "os." + "setsid", "osascript", "/tmp" + "/", "subprocess")
    for token in forbidden:
        assert token not in text, f"{token} would break the windows platform gate"


# ------------------------------------------------------------------- hub discoverability


def test_optional_skill_source_publishes_every_file():
    from tools.skills_hub_official import OptionalSkillSource

    source = OptionalSkillSource()
    source._optional_dir = REPO / "optional-skills"
    identifier = "official/autonomous-ai-agents/proactive-agent-souls"
    assert identifier in {m.identifier for m in source._scan_all()}, "absent from the optional catalog"
    bundle = source.fetch(identifier)
    assert bundle is not None, "the skill does not fetch"
    for expected in (
        "SKILL.md",
        "scripts/check_proactivity_contract.py",
        "templates/proactivity-section.md",
        "templates/observations-log.md",
    ):
        assert expected in bundle.files, sorted(bundle.files)
