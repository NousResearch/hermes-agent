"""User-facing message contracts for `hermes skills/plugins/mcp/cron/doctor` failure paths.

Every failure line must say what happened and name an existing command to type next; none may
lead with a raw exception, a status literal, or a hardcoded ``~/.hermes`` path (UX message audit,
findings cli-08/20/21/23/26/34, tools-runtime-13/14/20/27).
"""

from __future__ import annotations

import re
from types import SimpleNamespace

# ── skills hub ────────────────────────────────────────────────────────────────────────────────

# ── plugins ───────────────────────────────────────────────────────────────────────────────────

def test_plugin_clone_failure_leads_with_next_steps_and_escapes_git_output():
    from hermes_cli.plugins_cmd import _clone_failure_message
    msg = _clone_failure_message("https://github.com/acme/nope", "fatal: repository '[x]' not found")
    assert "\\[x]" in msg  # Rich markup escaped so the raw git text renders verbatim

# ── MCP ───────────────────────────────────────────────────────────────────────────────────────

# ── cron ──────────────────────────────────────────────────────────────────────────────────────

# ── doctor / advisories / paths ───────────────────────────────────────────────────────────────

def test_advisory_remediation_uses_active_hermes_home(monkeypatch, tmp_path):
    from hermes_cli import security_advisories as sa
    monkeypatch.setattr(sa, "display_hermes_home", lambda: "~/.hermes/profiles/work")
    hit = SimpleNamespace(advisory=sa.ADVISORIES[0], package="mistralai", installed_version="2.4.6")
    text = "\n".join(sa.full_remediation_text(hit))
    assert "~/.hermes/profiles/work/.env" in text
    assert "{hermes_home}" not in text
    assert not re.search(r"~/\.hermes/\.env", text)
