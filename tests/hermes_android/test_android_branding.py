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
    assert 'Theme.HermesAgent' in themes
    assert '@color/hermes_background' in themes
    assert 'viewportWidth="108"' in icon
    assert '#5B2E8C' in icon
    assert '<string name="app_name">Hermes</string>' in strings


def test_app_shell_has_alpha_brand_bar_and_hermes_logo():
    app_shell = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt").read_text(encoding="utf-8")
    theme_file = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/theme/HermesTheme.kt").read_text(encoding="utf-8")

    assert 'HermesTheme {' in app_shell
    assert 'HermesBrandBar()' in app_shell
    assert 'painterResource(id = R.drawable.ic_hermes_logo)' in app_shell
    assert 'Android alpha · local runtime + portal access' in app_shell
    assert 'text = "ALPHA"' in app_shell
    assert 'Hermes("Hermes")' in app_shell
    assert 'lightColorScheme(' in theme_file
    assert 'Color(0xFF5B2E8C)' in theme_file
