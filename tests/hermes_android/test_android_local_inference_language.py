from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_app_settings_store_persists_on_device_backend_and_language():
    app_settings = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/data/AppSettingsStore.kt").read_text(encoding="utf-8")

    assert 'onDeviceBackend' in app_settings
    assert 'languageTag' in app_settings
    assert 'KEY_ON_DEVICE_BACKEND' in app_settings
    assert 'KEY_LANGUAGE_TAG' in app_settings


def test_settings_screen_exposes_on_device_backend_switches_and_language_flags():
    settings_screen = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/ui/settings/SettingsScreen.kt").read_text(encoding="utf-8")

    assert 'On-device inference' in settings_screen
    assert 'llama.cpp (GGUF)' in settings_screen
    assert 'LiteRT-LM' in settings_screen
    assert 'App language' in settings_screen
    assert '🇬🇧' in settings_screen
    assert '🇨🇳' in settings_screen
    assert '🇪🇸' in settings_screen
    assert '🇩🇪' in settings_screen
    assert '🇵🇹' in settings_screen
    assert '🇫🇷' in settings_screen


def test_android_build_and_backend_sources_wire_litertlm_and_local_backend_orchestration():
    gradle = (REPO_ROOT / 'android/app/build.gradle.kts').read_text(encoding='utf-8')
    runtime_manager = (REPO_ROOT / 'android/app/src/main/java/com/nousresearch/hermesagent/backend/HermesRuntimeManager.kt').read_text(encoding='utf-8')
    backend_manager = (REPO_ROOT / 'android/app/src/main/java/com/nousresearch/hermesagent/backend/OnDeviceBackendManager.kt').read_text(encoding='utf-8')

    assert 'com.google.ai.edge.litertlm:litertlm-android' in gradle
    assert 'org.nanohttpd:nanohttpd' in gradle
    assert 'OnDeviceBackendManager' in runtime_manager
    assert 'LlamaCppServerController' in backend_manager
    assert 'LiteRtLmOpenAiProxy' in backend_manager
    assert 'BackendKind.LLAMA_CPP' in backend_manager
    assert 'BackendKind.LITERT_LM' in backend_manager


def test_android_linux_assets_include_llama_cpp_package_for_embedded_runtime():
    linux_assets = (REPO_ROOT / 'hermes_android/linux_assets.py').read_text(encoding='utf-8')

    assert 'llama-cpp' in linux_assets


def test_language_infrastructure_wires_app_shell_and_core_screens_to_shared_translations():
    app_shell = (REPO_ROOT / 'android/app/src/main/java/com/nousresearch/hermesagent/ui/shell/AppShell.kt').read_text(encoding='utf-8')
    chat_screen = (REPO_ROOT / 'android/app/src/main/java/com/nousresearch/hermesagent/ui/chat/ChatScreen.kt').read_text(encoding='utf-8')
    device_screen = (REPO_ROOT / 'android/app/src/main/java/com/nousresearch/hermesagent/ui/device/DeviceScreen.kt').read_text(encoding='utf-8')
    auth_screen = (REPO_ROOT / 'android/app/src/main/java/com/nousresearch/hermesagent/ui/auth/AuthScreen.kt').read_text(encoding='utf-8')
    portal_screen = (REPO_ROOT / 'android/app/src/main/java/com/nousresearch/hermesagent/ui/portal/NousPortalScreen.kt').read_text(encoding='utf-8')
    i18n_strings = (REPO_ROOT / 'android/app/src/main/java/com/nousresearch/hermesagent/ui/i18n/HermesStrings.kt').read_text(encoding='utf-8')

    assert 'LocalHermesStrings' in app_shell
    assert 'LocalHermesStrings.current' in chat_screen
    assert 'LocalHermesStrings.current' in device_screen
    assert 'LocalHermesStrings.current' in auth_screen
    assert 'LocalHermesStrings.current' in portal_screen
    assert 'AppLanguage.ENGLISH' in i18n_strings
    assert 'AppLanguage.CHINESE' in i18n_strings
    assert 'AppLanguage.SPANISH' in i18n_strings
    assert 'AppLanguage.GERMAN' in i18n_strings
    assert 'AppLanguage.PORTUGUESE' in i18n_strings
    assert 'AppLanguage.FRENCH' in i18n_strings
