from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_app_shell_uses_bottom_navigation_and_context_sheet():
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")
    shell_models = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/ShellModels.kt").read_text(encoding="utf-8")
    action_sheet = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/ContextActionSheet.kt").read_text(encoding="utf-8")

    assert 'Scaffold(' in app_shell
    assert 'NavigationBar(' in app_shell
    assert 'FloatingActionButton' in app_shell
    assert 'ContextActionSheet(' in app_shell
    assert 'TabRow' not in app_shell
    assert 'Portal(' in shell_models
    assert 'iconRes = R.drawable.ic_nav_hermes' in shell_models
    assert 'ModalBottomSheet' in action_sheet
    assert 'actions.forEachIndexed' in action_sheet


def test_shell_branding_and_settings_page_hide_context_actions():
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")

    assert 'section.subtitle' in app_shell
    assert 'currentSection == AppSection.Settings' in app_shell
    assert 'setActions(emptyList())' in app_shell
    assert 'Runtime setup and onboarding' in app_shell
