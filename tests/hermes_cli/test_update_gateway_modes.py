from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REMOVED_EXTERNAL_MODE = "external" + "-updater"


def test_official_desktop_updaters_need_no_update_mode_environment():
    sources = [
        ROOT / "apps/bootstrap-installer/src-tauri/src/update.rs",
        ROOT / "scripts/desktop-update/posix.sh",
        ROOT / "scripts/desktop-update/windows.ps1",
    ]

    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert "HERMES_UPDATE_MODE" not in text, source
        assert REMOVED_EXTERNAL_MODE not in text, source


def test_messaging_handoff_never_places_token_in_argv_or_environment():
    source = (ROOT / "gateway/slash_commands.py").read_text(encoding="utf-8")

    assert "HERMES_UPDATE_HANDOFF_TOKEN" not in source
    assert "HERMES_UPDATE_MODE" not in source
    assert "stdin=subprocess.PIPE" in source


def test_update_admission_has_no_external_updater_mode():
    source = (ROOT / "hermes_cli/main.py").read_text(encoding="utf-8")

    assert REMOVED_EXTERNAL_MODE not in source
    assert "HERMES_UPDATE_MODE" not in source
