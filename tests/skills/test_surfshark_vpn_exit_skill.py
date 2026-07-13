"""Frontmatter + structure tests for skills/devops/surfshark-vpn-exit.

Stdlib + pytest only, no network. Asserts the HARDLINE skill-authoring
constraints rather than freezing the prose content.
"""

import re
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "devops" / "surfshark-vpn-exit"
SKILL_MD = SKILL_DIR / "SKILL.md"


def _read() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "SKILL.md must open with a YAML frontmatter block"
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        fm_match = re.match(r"^([A-Za-z0-9_]+):\s*(.*)$", line)
        if fm_match:
            fm[fm_match.group(1)] = fm_match.group(2).strip()
    return fm


def test_skill_md_exists():
    assert SKILL_MD.is_file()


def test_name_matches_directory():
    fm = _frontmatter(_read())
    assert fm.get("name") == SKILL_DIR.name == "surfshark-vpn-exit"


def test_description_constraints():
    fm = _frontmatter(_read())
    desc = fm.get("description", "").strip().strip('"').strip("'")
    assert desc, "description is required"
    assert len(desc) <= 60, f"description must be <= 60 chars, got {len(desc)}"
    assert desc.endswith("."), "description must end with a period"
    assert desc.count(".") == 1, "description must be a single sentence"
    # Must not repeat the skill name tokens.
    lowered = desc.lower()
    for token in ("surfshark", "exit"):
        assert token not in lowered, f"description should not repeat skill-name token {token!r}"
    # No marketing words.
    for word in ("powerful", "comprehensive", "seamless", "advanced"):
        assert word not in lowered, f"description must not contain marketing word {word!r}"


def test_required_frontmatter_fields():
    fm = _frontmatter(_read())
    for field in ("name", "description", "version", "author", "license"):
        assert fm.get(field), f"missing required frontmatter field: {field}"


def test_linux_platform_declared():
    # POSIX-only (systemd, wireproxy, bash) — must gate platforms to linux.
    text = _read()
    m = re.search(r"^platforms:\s*\[(.*?)\]", text, re.MULTILINE)
    assert m, "platforms must be declared"
    platforms = {p.strip() for p in m.group(1).split(",") if p.strip()}
    assert "linux" in platforms
    assert "macos" not in platforms and "windows" not in platforms


def test_modern_section_order():
    text = _read()
    required = [
        "# Surfshark VPN Exit Skill",
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ]
    positions = []
    for heading in required:
        idx = text.find(heading)
        assert idx != -1, f"missing required section: {heading}"
        positions.append(idx)
    assert positions == sorted(positions), "sections must appear in the modern order"


def test_helper_script_present_and_referenced():
    script = SKILL_DIR / "scripts" / "vpn-exit.sh"
    assert script.is_file(), "expected scripts/vpn-exit.sh"
    assert "scripts/vpn-exit.sh" in _read(), "SKILL.md must reference the helper script"


def test_no_personal_info_leaked():
    """Generic contract: the skill must ship placeholders, never a real host.

    We assert the *shape* of a leak (developer home paths, concrete public
    IPs) rather than embedding any author-specific value, so the test itself
    stays personal-info free.
    """
    blob = _read() + (SKILL_DIR / "scripts" / "vpn-exit.sh").read_text(encoding="utf-8")
    for tpl in (SKILL_DIR / "templates").glob("*"):
        blob += tpl.read_text(encoding="utf-8")
    # No developer home directories.
    assert "/Users/" not in blob, "macOS home path leaked"
    assert not re.search(r"/home/[A-Za-z0-9_.-]+/", blob), "linux home path leaked"
    # Only loopback / RFC1918 / link-local / the documented Surfshark DNS
    # resolvers may appear as literal IPv4s. Any other concrete address is a
    # personal-host leak (e.g. a controller or VPS public IP).
    allowed_dns = {"162.252.172.57", "149.154.159.92"}
    for ip in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", blob):
        first, second, *_ = (int(o) for o in ip.split("."))
        is_loopback = first == 127
        is_unspecified = ip == "0.0.0.0"
        is_rfc1918 = first == 10 or (first == 192 and second == 168) or (
            first == 172 and 16 <= second <= 31
        )
        is_link_local = first == 169 and second == 254
        if is_loopback or is_unspecified or is_rfc1918 or is_link_local:
            continue
        assert ip in allowed_dns, f"unexpected literal public IP leaked: {ip}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
