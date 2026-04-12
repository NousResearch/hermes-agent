from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_android_manifest_uses_hermes_theme_and_icon():
    manifest = (REPO_ROOT / "android/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")

    assert 'android:icon="@drawable/ic_hermes_logo"' in manifest
    assert 'android:roundIcon="@drawable/ic_hermes_logo"' in manifest
    assert 'android:theme="@style/Theme.HermesAgent"' in manifest


def test_android_brand_resources_exist_and_define_hermes_palette():
    colors = (REPO_ROOT / "android/app/src/main/res/values/colors.xml").read_text(encoding="utf-8")
    themes = (REPO_ROOT / "android/app/src/main/res/values/themes.xml").read_text(encoding="utf-8")
    icon = (REPO_ROOT / "android/app/src/main/res/drawable/ic_hermes_logo.xml").read_text(encoding="utf-8")
    strings = (REPO_ROOT / "android/app/src/main/res/values/strings.xml").read_text(encoding="utf-8")

    assert 'name="hermes_primary"' in colors
    assert 'name="hermes_background"' in colors
    assert 'name="hermes_surface_dark"' in colors
    assert '#090B10' in colors
    assert 'Theme.HermesAgent' in themes
    assert '@color/hermes_background' in themes
    assert '@color/hermes_surface_dark' in themes
    assert 'viewportWidth="108"' in icon
    assert '#5B2E8C' in icon
    assert '<string name="app_name">Hermes</string>' in strings


def test_app_shell_has_compact_brand_bar_bottom_nav_and_custom_icons():
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")
    shell_models = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/ShellModels.kt").read_text(encoding="utf-8")
    theme_file = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/theme/HermesTheme.kt").read_text(encoding="utf-8")
    main_activity = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/MainActivity.kt").read_text(encoding="utf-8")
    drawable_files = sorted(path.name for path in (REPO_ROOT / "android/app/src/main/res/drawable").glob("ic_*.xml"))

    assert 'HermesTopBar(' in app_shell
    assert 'NavigationBar(' in app_shell
    assert 'R.drawable.ic_action_cog' in app_shell
    assert 'R.drawable.ic_nav_hermes' in shell_models
    assert 'R.drawable.ic_nav_accounts' in shell_models
    assert 'R.drawable.ic_nav_portal' in shell_models
    assert 'R.drawable.ic_nav_device' in shell_models
    assert 'R.drawable.ic_nav_settings' in shell_models
    assert 'darkColorScheme(' in theme_file
    assert 'Color(0xFF090B10)' in theme_file
    assert 'enableEdgeToEdge' in main_activity
    for name in [
        'ic_nav_hermes.xml',
        'ic_nav_accounts.xml',
        'ic_nav_portal.xml',
        'ic_nav_device.xml',
        'ic_nav_settings.xml',
        'ic_action_cog.xml',
        'ic_action_history.xml',
        'ic_action_refresh.xml',
        'ic_action_external.xml',
        'ic_action_mic.xml',
        'ic_action_speaker.xml',
    ]:
        assert name in drawable_files
