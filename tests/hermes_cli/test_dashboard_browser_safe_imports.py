"""Static dashboard tests for browser-safe @nous-research/ui imports."""
from pathlib import Path


WEB_SRC = Path(__file__).resolve().parents[2] / "web" / "src"

# Baseline offenders that still predate the browser-safety migration in
# fix(dashboard): avoid node-only ui imports in browser.
#
# This test intentionally allows the preexisting files to keep importing the
# root barrel so that the guardrail only fails on newly introduced offenders.
LEGACY_ROOT_BARREL_OFFENDERS = {
    "components/AutoField.tsx",
    "components/ChatSidebar.tsx",
    "components/ModelInfoCard.tsx",
    "components/ModelPickerDialog.tsx",
    "components/OAuthProvidersCard.tsx",
    "components/PlatformsCard.tsx",
    "components/SlashPopover.tsx",
    "components/ToolCall.tsx",
    "components/ui/confirm-dialog.tsx",
    "pages/AnalyticsPage.tsx",
    "pages/ConfigPage.tsx",
    "pages/EnvPage.tsx",
    "pages/LogsPage.tsx",
    "pages/ProfilesPage.tsx",
    "pages/SessionsPage.tsx",
    "pages/SkillsPage.tsx",
    "plugins/PluginPage.tsx",
}

MIGRATED_FILES = {
    "App.tsx",
    "components/LanguageSwitcher.tsx",
    "components/OAuthLoginModal.tsx",
    "components/SidebarFooter.tsx",
    "components/ThemeSwitcher.tsx",
    "pages/ChatPage.tsx",
    "pages/CronPage.tsx",
}


def test_dashboard_does_not_introduce_new_nous_ui_root_barrel_offenders():
    offenders = set()
    for ext in ("*.tsx", "*.ts"):
        for path in WEB_SRC.rglob(ext):
            content = path.read_text(encoding="utf-8")
            if 'from "@nous-research/ui"' in content or "from '@nous-research/ui'" in content:
                offenders.add(str(path.relative_to(WEB_SRC)))

    unexpected = offenders - LEGACY_ROOT_BARREL_OFFENDERS
    assert unexpected == set(), f"Unexpected root-barrel offenders: {sorted(unexpected)}"

    removed = MIGRATED_FILES & offenders
    assert removed == set(), f"Migrated files still import the root barrel: {sorted(removed)}"
