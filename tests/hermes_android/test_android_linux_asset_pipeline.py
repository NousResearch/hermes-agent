import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path

from hermes_android.linux_assets import serializable_manifest
from scripts.prepare_android_linux_assets import mirror_data_tar


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_prepare_android_linux_assets_script_exists_and_is_wired_into_gradle():
    script = (REPO_ROOT / "scripts/prepare_android_linux_assets.py").read_text(encoding="utf-8")
    native_script = (REPO_ROOT / "scripts/prepare_android_native_libs.py").read_text(encoding="utf-8")
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")

    assert "def prepare_assets" in script
    assert "resolve_dependency_closure" in script
    assert "prepareHermesAndroidLinuxAssets" in gradle
    assert "prepareHermesAndroidNativeLibs" in gradle
    assert "generated/hermes-linux-assets" in gradle
    assert "generated/hermes-native-libs" in gradle
    assert "assets.srcDir" in gradle
    assert "jniLibs.srcDir" in gradle
    assert "useLegacyPackaging = true" in gradle
    assert "NEEDED_RENAMES" in native_script
    assert "libreadline.so.8" in native_script


def test_prepare_android_linux_assets_script_imports_from_android_workdir():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/prepare_android_linux_assets.py"), "--help"],
        cwd=REPO_ROOT / "android",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Prepare Android Linux CLI assets" in result.stdout


def test_linux_asset_manifest_normalizes_windows_link_targets():
    manifest = serializable_manifest(
        "arm64-v8a",
        packages=[],
        links=[
            {"path": "lib\\libreadline.so.8", "target": "lib\\libreadline.so.8.3"},
            {"path": "/bin\\sh", "target": "bin\\busybox"},
        ],
    )

    assert manifest["links"] == [
        {"path": "bin/sh", "target": "bin/busybox"},
        {"path": "lib/libreadline.so.8", "target": "lib/libreadline.so.8.3"},
    ]


def test_prepare_android_linux_assets_mirrors_absolute_termux_symlinks(tmp_path):
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        directory = tarfile.TarInfo("./data/data/com.termux/files/usr/bin")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)

        payload = b"#!/data/data/com.termux/files/usr/bin/bash\necho ok\n"
        file_info = tarfile.TarInfo("./data/data/com.termux/files/usr/bin/bzdiff")
        file_info.mode = 0o755
        file_info.size = len(payload)
        tar.addfile(file_info, BytesIO(payload))

        link_info = tarfile.TarInfo("./data/data/com.termux/files/usr/bin/bzcmp")
        link_info.type = tarfile.SYMTYPE
        link_info.linkname = "/data/data/com.termux/files/usr/bin/bzdiff"
        tar.addfile(link_info)

    archive.seek(0)
    prefix = tmp_path / "prefix"
    with tarfile.open(fileobj=archive, mode="r:") as tar:
        links = mirror_data_tar(tar, prefix)

    assert (prefix / "bin" / "bzdiff").read_text(encoding="utf-8") == "#!/usr/bin/env bash\necho ok\n"
    assert not (prefix / "bin" / "bzcmp").exists()
    assert links == [{"path": "bin/bzcmp", "target": "bin/bzdiff"}]


def test_android_linux_subsystem_recreates_windows_manifest_links():
    bridge = (
        REPO_ROOT
        / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesLinuxSubsystemBridge.kt"
    ).read_text(encoding="utf-8")

    assert "normalizeAssetRelativePath(item.optString(\"path\"))" in bridge
    assert "normalizeAssetRelativePath(item.optString(\"target\"))" in bridge
    assert ".replace('\\\\', '/')" in bridge


def test_android_linux_subsystem_retries_after_app_update():
    bridge = (
        REPO_ROOT
        / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesLinuxSubsystemBridge.kt"
    ).read_text(encoding="utf-8")

    assert 'state.optLong("app_version_code", -1L) != currentAppVersionCode' in bridge
    assert 'put("app_version_code", currentAppVersionCode)' in bridge
    assert 'state.optString("asset_manifest_sha256") != currentAssetFingerprint' in bridge
    assert 'put("asset_manifest_sha256", currentAssetFingerprint)' in bridge
    assert 'state.optString("native_library_dir") != currentNativeLibraryDir' in bridge
    assert 'state.optString("execution_mode") == SYSTEM_SHELL_MODE' not in bridge
    assert "Embedded Linux assets unavailable" in bridge
    assert '"HERMES_ANDROID_SHELL" to SYSTEM_SHELL_PATH' in bridge
    assert '"HERMES_ANDROID_NATIVE_SHELL" to state.optString("shell_path")' in bridge
    assert "private fun appVersionCode(context: Context): Long" in bridge


def test_android_linux_subsystem_records_embedded_fallback_reason():
    bridge = (
        REPO_ROOT
        / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesLinuxSubsystemBridge.kt"
    ).read_text(encoding="utf-8")
    llama = (
        REPO_ROOT
        / "android/app/src/main/java/com/nousresearch/hermesagent/backend/LlamaCppServerController.kt"
    ).read_text(encoding="utf-8")

    assert "private data class ShellLaunchProbe" in bridge
    assert 'put("fallback_reason", fallbackReason.take(1200))' in bridge
    assert "llama.cpp is not available in native Android shell mode: $fallbackReason" in llama


def test_android_gguf_launchers_use_native_library_directory():
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")
    bridge = (
        REPO_ROOT
        / "android/app/src/main/java/com/nousresearch/hermesagent/device/HermesLinuxSubsystemBridge.kt"
    ).read_text(encoding="utf-8")
    llama = (
        REPO_ROOT
        / "android/app/src/main/java/com/nousresearch/hermesagent/backend/LlamaCppServerController.kt"
    ).read_text(encoding="utf-8")
    native_script = (REPO_ROOT / "scripts/prepare_android_native_libs.py").read_text(encoding="utf-8")

    assert "scripts/prepare_android_native_libs.py" in gradle
    assert "libhermes_android_bash.so" in native_script
    assert "libhermes_android_llama_server.so" in native_script
    assert 'nativeExecutablePath(context, "libhermes_android_bash.so")' in bridge
    assert 'put("native_library_dir", context.applicationInfo.nativeLibraryDir.orEmpty())' in bridge
    assert 'optString("native_llama_server_path").ifBlank { "llama-server" }' in llama
    assert ".readTimeout(750, TimeUnit.MILLISECONDS)" in llama
    assert "repeat(360)" in llama
