"""Tests for Hermes Code Mode slash commands and home screen requirements.

Covers the 18 test cases specified in the Code Mode refactoring task.
"""

from unittest.mock import MagicMock, patch
from rich.console import Console
import hermes_cli.banner as banner


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _console(width: int = 120) -> Console:
    return Console(record=True, force_terminal=False, color_system=None, width=width)


# ===========================================================================
# 1. Home screen render requirements (1-10)
# ===========================================================================

def test_home_renders_without_crash():
    """1. The home real do CLI renderiza sem crash."""
    c = _console()
    banner.build_hermes_code_console(c, model="test-model")
    assert len(c.export_text()) > 0


def test_home_contains_hermes_code_mode():
    """2. A home contém 'Hermes Code Mode' ou 'Hermes Code Console'."""
    c = _console()
    banner.build_hermes_code_console(c, model="test-model")
    out = c.export_text()
    assert "Hermes Code Mode" in out or "Hermes Code Console" in out


def test_home_contains_cockpit_url():
    """3. A home contém http://localhost:3001/code."""
    c = _console()
    banner.build_hermes_code_console(c, model="test-model")
    assert "localhost:3001" in c.export_text()


def test_home_shows_model_when_provided():
    """4. A home mostra provider/model quando disponível."""
    c = _console()
    banner.build_hermes_code_console(c, model="my-unique-model-xyz")
    assert "my-unique-model-xyz" in c.export_text()


def test_home_shows_workspace_label():
    """5a. A home mostra workspace quando disponível."""
    c = _console()
    banner.build_hermes_code_console(c, model="test-model")
    assert "Workspace" in c.export_text()


def test_home_shows_branch_label():
    """5b. A home mostra branch quando disponível."""
    c = _console()
    banner.build_hermes_code_console(c, model="test-model")
    assert "Branch" in c.export_text()


def test_home_works_with_backend_offline():
    """6. A home funciona com backend offline (sem travar)."""
    import requests
    c = _console()
    with patch("requests.get", side_effect=ConnectionRefusedError("backend offline")):
        banner.build_hermes_code_console(c, model="test-model")
    out = c.export_text()
    assert "Hermes Code Mode" in out or "Hermes Code Console" in out


def test_home_works_outside_git_repo():
    """7. A home funciona fora de repositório Git."""
    import subprocess
    c = _console()
    with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
        banner.build_hermes_code_console(c, model="test-model")
    out = c.export_text()
    assert len(out) > 0


def test_home_no_giant_available_tools_block():
    """8. A home não mostra por padrão o bloco gigante 'Available Tools'."""
    c = _console()
    banner.build_hermes_code_console(c, model="test-model")
    out = c.export_text()
    assert "Available Tools" not in out


def test_home_no_giant_available_skills_block():
    """9. A home não mostra por padrão o bloco gigante 'Available Skills'."""
    c = _console()
    banner.build_hermes_code_console(c, model="test-model")
    out = c.export_text()
    assert "Available Skills" not in out


def test_home_palette_no_dominant_yellow():
    """10. A nova paleta não usa amarelo/laranja saturado como cor dominante no title."""
    # The title and border color keys are code_title/code_border — teal, not gold.
    from hermes_cli.skin_engine import get_active_skin
    skin = get_active_skin()
    title_color = skin.get_color("code_title", "#00E5FF")
    # Should not be the old gold/yellow values
    assert title_color.upper() not in ("#FFD700", "#FFBF00", "#FFA500")


def test_code_mode_skin_semantic_colors_exist():
    """Code Mode has semantic cyan/teal color keys with safe fallbacks."""
    from hermes_cli.skin_engine import get_active_skin
    skin = get_active_skin()
    expected = {
        "code_title",
        "code_title_dim",
        "code_accent",
        "code_accent_dim",
        "code_border",
        "code_text",
        "code_muted",
        "code_success",
        "code_warning",
        "code_error",
    }
    assert expected.issubset(set(skin.colors))
    dominant = {
        skin.get_color("code_title").upper(),
        skin.get_color("code_accent").upper(),
        skin.get_color("code_border").upper(),
    }
    assert dominant.isdisjoint({"#FFD700", "#FFBF00", "#FFA500", "#CD7F32"})


# ===========================================================================
# 2. Slash command output tests (11-18)
# ===========================================================================

def _make_cli_with_console():
    """Build a minimal object with a recording console + all Code Mode handler methods."""
    c = Console(record=True, force_terminal=False, color_system=None, width=120)
    import cli as cli_module

    handler_names = [
        "_handle_code_command",
        "_handle_web_command",
        "_handle_workspace_command",
        "_handle_session_command",
        "_handle_approvals_command",
        "_handle_skills_code_command",
    ]

    attrs = {"console": c}
    for name in handler_names:
        attrs[name] = getattr(cli_module.HermesCLI, name)

    MinimalCLI = type("MinimalCLI", (), attrs)
    return MinimalCLI(), c


def test_slash_code_shows_cockpit_info():
    """11. /code mostra ajuda do Code Mode."""
    obj, c = _make_cli_with_console()
    obj._handle_code_command("")
    out = c.export_text()
    assert "Hermes Code Mode" in out
    assert "localhost:3001/code" in out
    assert "Useful: /workspace /session /skills-code /approvals /web" in out


def test_slash_web_shows_correct_urls():
    """12. /web mostra URLs corretas."""
    obj, c = _make_cli_with_console()
    obj._handle_web_command("")
    out = c.export_text()
    assert "localhost:3001" in out
    assert "localhost:9119" in out
    assert "localhost:3001/code" in out
    assert "Logs: tail -f /tmp/hermes-backend.log /tmp/hermes-frontend.log" in out


def test_slash_workspace_does_not_crash():
    """13. /workspace não quebra."""
    obj, c = _make_cli_with_console()
    obj._handle_workspace_command("")
    out = c.export_text()
    assert "Workspace" in out


def test_slash_session_does_not_crash():
    """14. /session não quebra (backend offline ok)."""
    obj, c = _make_cli_with_console()
    with patch("requests.get", side_effect=ConnectionRefusedError("offline")):
        obj._handle_session_command("")
    out = c.export_text()
    assert len(out) > 0


def test_slash_session_handles_backend_sessions_envelope():
    """/session handles the real backend envelope shape."""
    obj, c = _make_cli_with_console()
    response = MagicMock(status_code=200)
    response.json.return_value = {"sessions": [], "total": 0}
    with patch("requests.get", return_value=response):
        obj._handle_session_command("")
    out = c.export_text()
    assert "No active code sessions" in out


def test_slash_session_handles_backend_auth_required_as_empty_fallback():
    """/session degrades gracefully when the backend requires auth."""
    obj, c = _make_cli_with_console()
    response = MagicMock(status_code=401)
    with patch("requests.get", return_value=response):
        obj._handle_session_command("")
    out = c.export_text()
    assert "No active code sessions" in out


def test_slash_approvals_does_not_crash():
    """15. /approvals não quebra (backend offline ok)."""
    obj, c = _make_cli_with_console()
    with patch("requests.get", side_effect=ConnectionRefusedError("offline")):
        obj._handle_approvals_command("")
    out = c.export_text()
    assert len(out) > 0


def test_slash_approvals_handles_backend_approvals_envelope():
    """/approvals handles the real backend envelope shape."""
    obj, c = _make_cli_with_console()
    response = MagicMock(status_code=200)
    response.json.return_value = {"approvals": [], "total": 0}
    with patch("requests.get", return_value=response):
        obj._handle_approvals_command("")
    out = c.export_text()
    assert "No pending approvals" in out


def test_slash_approvals_handles_backend_auth_required_as_empty_fallback():
    """/approvals degrades gracefully when the backend requires auth."""
    obj, c = _make_cli_with_console()
    response = MagicMock(status_code=401)
    with patch("requests.get", return_value=response):
        obj._handle_approvals_command("")
    out = c.export_text()
    assert "No pending approvals" in out


def test_slash_skills_code_lists_seven_skills():
    """16. /skills-code lista as 7 coding skills."""
    obj, c = _make_cli_with_console()
    obj._handle_skills_code_command("")
    out = c.export_text()
    expected = [
        "fix_build",
        "review_diff",
        "stabilize_hanging_task",
        "fix_runtime_error",
        "implement_feature",
        "refactor_react_page",
        "benchmark_provider",
    ]
    for skill in expected:
        assert skill in out, f"Missing skill: {skill}"


def test_slash_help_command_registered():
    """17. /help continua registrado no sistema de comandos."""
    from hermes_cli.commands import resolve_command
    cmd = resolve_command("help")
    assert cmd is not None
    assert cmd.name == "help"


def test_slash_tools_and_skills_still_registered():
    """18. /tools e /skills continuam disponíveis no sistema de comandos."""
    from hermes_cli.commands import resolve_command
    tools_cmd = resolve_command("tools")
    skills_cmd = resolve_command("skills")
    assert tools_cmd is not None and tools_cmd.name == "tools"
    assert skills_cmd is not None and skills_cmd.name == "skills"
