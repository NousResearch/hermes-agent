"""Regression tests for issue #92993 — `curator pin` false success + invisible pin.

Two stacked defects, each with its own failing assertion:

1. **False success (write-path).** `_cmd_pin` gates on `is_agent_created()`
   and then prints "pinned" unconditionally, but `set_pinned()` routes through
   `_mutate(require_curation_eligible=True)`, which silently no-ops when
   `is_curation_eligible()` is False. A skill can pass the first gate and fail
   the second; the user sees "pinned" while nothing was written. The CLI must
   detect the failed write and exit nonzero with a clear error.

2. **Invisible pin (visibility).** `curator status` reads pins from
   `curated_report()`, which only iterates `list_agent_created_skill_names()` —
   and that list requires the `created_by: agent` management marker. A skill
   that IS curation-eligible but carries no marker (pre-provenance record,
   foreground-created) never appears as a row, so its pin is silently absent
   from the status output even when successfully written.
"""

from __future__ import annotations

import importlib
import io
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(skills_dir: Path, name: str):
    d = skills_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: x\n---\n", encoding="utf-8",
    )
    return d


class _Args:
    def __init__(self, skill: str):
        self.skill = skill


@pytest.fixture
def pin_env(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with freshly reloaded modules (same pattern as
    tests/agent/test_curator.py::curator_env but scoped for the CLI layer)."""
    home = tmp_path / ".hermes"
    skills = home / "skills"
    skills.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    importlib.reload(hermes_constants)
    from tools import skill_usage
    importlib.reload(skill_usage)
    # curator module reads config through its own loader; keep defaults.
    from agent import curator
    importlib.reload(curator)
    monkeypatch.setattr(curator, "_load_config", lambda: {})
    monkeypatch.setattr(skill_usage, "_prune_builtins_enabled", lambda: False)
    from hermes_cli import curator as curator_cli
    importlib.reload(curator_cli)

    yield {
        "home": home,
        "skills": skills,
        "usage": skill_usage,
        "cli": curator_cli,
    }


# ---------------------------------------------------------------------------
# Test 1 — pin must fail loudly when the write does not land
# ---------------------------------------------------------------------------


def test_pin_fails_loudly_when_write_does_not_land(pin_env, capsys):
    """A skill that passes `is_agent_created()` but fails
    `is_curation_eligible()` must produce a NONZERO exit and an explanatory
    error — not a success message over a silent no-write.

    Real trigger: PROTECTED_BUILTIN_SKILLS blocks by NAME. A user's own skill
    literally named ``plan`` is not in the bundled manifest, so
    ``is_agent_created()`` says True — but ``is_protected_builtin()`` makes it
    ineligible, and ``set_pinned()`` silently no-ops through
    ``_mutate(require_curation_eligible=True)``."""
    env = pin_env
    name = "plan"  # collides with the protected built-in name

    _make_skill(env["skills"], name)

    # Sanity: the two gates genuinely disagree for this skill.
    assert env["usage"].is_agent_created(name) is True
    assert env["usage"].is_curation_eligible(name) is False

    cli = env["cli"]
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli._cmd_pin(_Args(name))
    out = buf.getvalue()

    # The bug: rc == 0 with "pinned" printed although nothing was written.
    assert rc != 0, (
        f"_cmd_pin reported success (rc=0, output={out!r}) for a skill whose "
        "pin write silently did not land — this is the false-success defect."
    )
    # Record on disk must NOT claim a pin.
    rec = env["usage"].get_record(name)
    assert not rec.get("pinned"), "usage record claims pinned=true despite failed eligibility"
    # Error output must explain the refusal, not just fail quietly.
    assert "pin" in out.lower(), "refusal must mention the pin outcome"


# ---------------------------------------------------------------------------
# Test 2 — successful pin on eligible-but-unmanaged skill must be VISIBLE
# ---------------------------------------------------------------------------


def test_pinned_eligible_unmanaged_skill_visible_in_status(pin_env):
    """A curation-ELIGIBLE skill without the created_by marker (foreground-
    created / pre-provenance) can be pinned successfully — and once pinned,
    it MUST surface in `curator status` output. Before the fix it vanishes:
    curated_report() skips it because list_agent_created_skill_names()
    requires the management marker."""
    env = pin_env
    usage = env["usage"]
    cli = env["cli"]

    _make_skill(env["skills"], "legacy-skill")
    # No mark_agent_created() call — deliberately unmanaged-but-eligible.

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli._cmd_pin(_Args("legacy-skill"))
    out = buf.getvalue()

    # The pin MUST succeed for an eligible skill — unconditional, so a
    # regression that flips pin to failure fails here instead of silently
    # skipping the visibility assertions below.
    assert rc == 0, f"eligible-skill pin must succeed (got rc={rc}): {out.strip()}"
    rec = usage.get_record("legacy-skill")
    assert rec.get("pinned") is True, "eligible skill pin wrote nothing despite rc=0"

    # ...and then status MUST show it. This is the visibility half.
    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        status_rc = cli._cmd_status(Namespace())
    assert status_rc == 0
    status = buf2.getvalue()
    assert "legacy-skill" in status and "pinned" in status.lower(), (
        "pin landed on disk but the skill is absent from `curator status` "
        "— the invisible-pin defect."
    )


# ---------------------------------------------------------------------------
# Test 3 — regression guard: normal managed pin still works end to end
# ---------------------------------------------------------------------------


def test_pin_managed_skill_end_to_end(pin_env):
    """The happy path must stay green: a properly adopted (managed) skill
    pins, reports success, and shows in status. Guards against fixing the
    two defects by breaking legitimate pins."""
    env = pin_env
    usage = env["usage"]
    cli = env["cli"]

    _make_skill(env["skills"], "managed-skill")
    usage.mark_agent_created("managed-skill")

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli._cmd_pin(_Args("managed-skill"))
    assert rc == 0, "managed skill pin must succeed"
    assert "pinned" in buf.getvalue().lower()

    assert usage.get_record("managed-skill").get("pinned") is True

    buf2 = io.StringIO()
    with redirect_stdout(buf2):
        assert cli._cmd_status(Namespace()) == 0
    status = buf2.getvalue()
    assert "managed-skill" in status
    assert "pinned" in status.lower(), (
        "managed+pinned skill missing from the pinned list in status output"
    )
