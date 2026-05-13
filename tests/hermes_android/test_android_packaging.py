from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chaquopy_build_preinstalls_android_stubs():
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")

    assert 'prepareHermesAndroidWheel' in gradle
    assert "normalize_chaquopy_assets.py" in gradle
    assert 'it.name.endsWith("PythonRequirementsAssets")' in gradle
    assert 'it.name.startsWith("merge") && it.name.endsWith("Assets")' in gradle
    assert 'options("--no-deps")' in gradle
    assert 'install("../../android/pip-stubs/anthropic-stub")' in gradle
    assert 'install("../../android/pip-stubs/fal-client-stub")' in gradle
    assert 'install("build/hermes-wheel/${hermesWheelName()}")' in gradle
    assert 'install("-r", "../../requirements-android-chaquopy.txt")' in gradle


def test_android_release_workflow_uses_hash_based_python_bytecode():
    workflow = (REPO_ROOT / ".github/workflows/android-release.yml").read_text(encoding="utf-8")

    assert 'SOURCE_DATE_EPOCH: "315532800"' in workflow


def test_chaquopy_asset_normalizer_removes_local_install_urls_and_canonicalizes_pyc():
    script = (REPO_ROOT / "scripts/normalize_chaquopy_assets.py").read_text(encoding="utf-8")

    assert 'name.endswith(".dist-info/direct_url.json")' in script
    assert "marshal.dumps(code, 2)" in script
    assert "zipfile.ZIP_STORED" in script


def test_android_wheel_includes_iteration_limits_module():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "iteration_limits" in pyproject["tool"]["setuptools"]["py-modules"]


def test_fdroid_updatecheck_data_uses_literal_version_code_for_future_tags():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version_name = pyproject["project"]["version"]
    major, minor, patch = (int(part) for part in version_name.split("."))
    expected_code = major * 1_000_000 + minor * 10_000 + patch * 100 + 90
    version_file = dict(
        line.split("=", 1)
        for line in (REPO_ROOT / "fdroid/com.nousresearch.hermesagent.version")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    template = (REPO_ROOT / "fdroid/com.nousresearch.hermesagent.yml.template").read_text(encoding="utf-8")

    assert version_file == {
        "versionName": version_name,
        "versionCode": str(expected_code),
    }
    assert "UpdateCheckMode: Tags" in template
    assert "UpdateCheckData: fdroid/com.nousresearch.hermesagent.version|versionCode=(\\d+)|.|versionName=(.*)" in template


def test_android_anthropic_stub_matches_project_requirement_pin():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    stub_project = tomllib.loads(
        (REPO_ROOT / "android/pip-stubs/anthropic-stub/pyproject.toml").read_text(encoding="utf-8")
    )

    base_anthropic = next(
        dep for dep in pyproject["project"]["optional-dependencies"]["anthropic"]
        if dep.startswith("anthropic==")
    )
    assert base_anthropic == f"anthropic=={stub_project['project']['version']}"


def test_android_runtime_requirements_pin_pre_jiter_openai_sdk():
    requirements = (REPO_ROOT / "requirements-android-chaquopy.txt").read_text(encoding="utf-8")

    assert "croniter==6.0.0" in requirements
    assert "python-dateutil==2.9.0.post0" in requirements
    assert "pytz==2025.2" in requirements
    assert "six==1.17.0" in requirements
    assert "openai==1.39.0" in requirements
    assert "httpx==0.27.2" in requirements
    assert "pydantic==1.10.24" in requirements
    assert "\nfirecrawl-py" not in requirements
    assert "\npydantic_core" not in requirements


def test_runtime_service_enters_foreground_before_runtime_startup():
    service = (REPO_ROOT / "android/app/src/main/java/com/nousresearch/hermesagent/backend/HermesRuntimeService.kt").read_text(encoding="utf-8")
    start_body = service.split("private fun startOrRefreshForeground()", 1)[1].split("private fun buildNotification", 1)[0]

    assert start_body.index("promoteToForeground(runtime = null)") < start_body.index("HermesRuntimeManager.ensureStarted(")
    assert "override fun onCreate()" in service
    assert "promoteToForeground(runtime = null)" in service.split("override fun onCreate()", 1)[1].split("override fun onStartCommand", 1)[0]
    assert "ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC" in service
    assert 'val notification = buildNotification(runtime)' in service


def test_android_anthropic_stub_warns_at_runtime():
    stub_init = (REPO_ROOT / "android/pip-stubs/anthropic-stub/anthropic/__init__.py").read_text(encoding="utf-8")

    assert "not available in the Hermes Android MVP build" in stub_init
    assert "OpenAI-compatible provider" in stub_init


def test_android_fal_client_stub_marks_image_generation_deferred():
    stub_init = (REPO_ROOT / "android/pip-stubs/fal-client-stub/fal_client/__init__.py").read_text(encoding="utf-8")
    toolset_file = (REPO_ROOT / "toolsets.py").read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "android/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")

    assert "__hermes_android_stub__ = True" in stub_init
    assert "Image generation is deferred" in stub_init
    android_toolset_block = toolset_file.split('"hermes-android-app":', 1)[1].split('},', 1)[0]
    assert '"image_generate"' not in android_toolset_block
    assert '"terminal"' in android_toolset_block
    assert '"process"' in android_toolset_block
    assert '"android_device_status"' in android_toolset_block
    assert '"android_shared_folder_list"' in android_toolset_block
    assert '"android_shared_folder_read"' in android_toolset_block
    assert '"android_shared_folder_write"' in android_toolset_block
    assert '"android_ui_snapshot"' in android_toolset_block
    assert '"android_ui_action"' in android_toolset_block
    assert '"android_system_action"' in android_toolset_block
    assert '"read_file"' in android_toolset_block
    assert '"write_file"' in android_toolset_block
    assert 'android.permission.POST_NOTIFICATIONS' in manifest
    assert 'android.permission.ACCESS_WIFI_STATE' in manifest
    assert 'android.permission.BLUETOOTH_CONNECT' in manifest
    assert 'android.permission.NFC' in manifest
    assert 'android.permission.SYSTEM_ALERT_WINDOW' in manifest
    assert 'android.permission.FOREGROUND_SERVICE' in manifest
    assert 'HermesRuntimeService' in manifest
    assert 'HermesNotificationListenerService' in manifest
    assert 'android.permission.BIND_NOTIFICATION_LISTENER_SERVICE' in manifest


def test_android_declares_shizuku_privileged_access_support():
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")
    manifest = (REPO_ROOT / "android/app/src/main/AndroidManifest.xml").read_text(encoding="utf-8")
    bridge = (
        REPO_ROOT
        / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesPrivilegedAccessBridge.kt"
    ).read_text(encoding="utf-8")
    system_bridge = (
        REPO_ROOT
        / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesSystemControlBridge.kt"
    ).read_text(encoding="utf-8")

    assert 'implementation("dev.rikka.shizuku:api:13.1.5")' in gradle
    assert 'implementation("dev.rikka.shizuku:provider:13.1.5")' in gradle
    assert 'moe.shizuku.manager.permission.API' in manifest
    assert 'rikka.shizuku.ShizukuProvider' in manifest
    assert 'android:authorities="${applicationId}.shizuku"' in manifest
    assert "Shizuku.pingBinder()" in bridge
    assert "Shizuku.checkSelfPermission()" in bridge
    assert "Shizuku.requestPermission" in bridge
    assert "open_wireless_debugging_settings" in bridge
    assert "open_shizuku_app" in bridge
    assert "privileged_access" in system_bridge


def test_android_debug_version_code_tracks_project_semver():
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")
    version_file = (REPO_ROOT / "fdroid/com.nousresearch.hermesagent.version").read_text(encoding="utf-8")
    project_metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = project_metadata["project"]["version"]
    major, minor, patch = (int(part) for part in project_version.split("."))
    expected_version_code = (major * 1_000_000) + (minor * 10_000) + (patch * 100) + 90

    assert "fun semverVersionCode(versionText: String): Int?" in gradle
    assert "return semverVersionCode(hermesVersionName()) ?: 1" in gradle
    assert "semverVersionCode(releaseTag)?.let { return it }" in gradle
    assert f"versionName={project_version}" in version_file
    assert f"versionCode={expected_version_code}" in version_file


def test_android_wheel_build_clears_stale_python_build_output():
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")

    assert 'val generatedPythonBuildLibDir = repoRoot.resolve("build/lib")' in gradle
    assert "generatedPythonBuildLibDir.deleteRecursively()" in gradle
    assert "Refusing to remove Python build output outside repository" in gradle


def test_android_visual_harness_supports_wide_screenshots_and_clicks():
    harness = (REPO_ROOT / "scripts/android_visual_harness.py").read_text(encoding="utf-8")

    assert 'exec-out", "screencap", "-p"' in harness
    assert "out.write_bytes(proc.stdout)" in harness
    assert "proc.stdout.replace" not in harness
    assert '"input", "tap"' in harness
    assert '"swipe"' in harness
    assert '"text"' in harness
    assert '"wm", "size"' in harness
    assert '"wm", "density"' in harness
    assert "DEFAULT_READY_TEXT" in harness
    assert "wait_for_ui_text" in harness
    assert "No activities found" in harness
    assert "com.nousresearch.hermesagent" in harness
