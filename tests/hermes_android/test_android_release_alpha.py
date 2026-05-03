from pathlib import Path
import importlib.util
import os

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_android_release_manifest_module():
    script = REPO_ROOT / "scripts/android_release_manifest.py"
    spec = importlib.util.spec_from_file_location("android_release_manifest", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_android_build_gradle_supports_semver_alpha_release_tags():
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")

    assert 'fun androidVersionName()' in gradle
    assert 'v?(\\d+)\\.(\\d+)\\.(\\d+)(?:-([A-Za-z]+)(?:[.-]?(\\d+))?)?' in gradle
    assert '"alpha" -> 1' in gradle
    assert '"beta" -> 2' in gradle
    assert 'versionName = androidVersionName()' in gradle


def test_android_release_workflow_restores_signing_material_and_builds_release_artifacts():
    workflow = (REPO_ROOT / ".github/workflows/android-release.yml").read_text(encoding="utf-8")

    assert 'actions/checkout@v5' in workflow
    assert 'actions/setup-java@v5' in workflow
    assert 'actions/setup-python@v6' in workflow
    assert 'android-actions/setup-android@v4' in workflow
    assert 'ANDROID_KEYSTORE_BASE64' in workflow
    assert 'ANDROID_KEYSTORE_PASSWORD' in workflow
    assert 'ANDROID_KEY_ALIAS' in workflow
    assert 'ANDROID_KEY_PASSWORD' in workflow
    assert './gradlew :app:assembleRelease :app:bundleRelease' in workflow
    assert 'scripts/android_release_manifest.py --tag' in workflow
    assert 'gh release upload "${{ github.event.release.tag_name }}"' in workflow
    assert 'GH_TOKEN: ${{ github.token }}' in workflow


def test_android_push_workflow_uses_node24_ready_action_versions():
    workflow = (REPO_ROOT / ".github/workflows/android.yml").read_text(encoding="utf-8")

    assert 'actions/checkout@v5' in workflow
    assert 'actions/setup-java@v5' in workflow
    assert 'actions/setup-python@v6' in workflow
    assert 'android-actions/setup-android@v4' in workflow
    assert 'actions/upload-artifact@v7' in workflow


def test_android_release_manifest_prefers_universal_apk_over_newer_split(tmp_path):
    manifest = _load_android_release_manifest_module()
    apk_dir = tmp_path / "release"
    apk_dir.mkdir()

    universal = apk_dir / "app-universal-release.apk"
    split = apk_dir / "app-x86_64-release.apk"
    universal.write_bytes(b"universal")
    split.write_bytes(b"x86 only")

    split_mtime = universal.stat().st_mtime + 30
    os.utime(split, (split_mtime, split_mtime))

    assert manifest.select_release_apk(apk_dir) == universal


def test_android_release_manifest_rejects_ambiguous_split_only_apks(tmp_path):
    manifest = _load_android_release_manifest_module()
    apk_dir = tmp_path / "release"
    apk_dir.mkdir()

    (apk_dir / "app-arm64-v8a-release.apk").write_bytes(b"arm64")
    (apk_dir / "app-x86_64-release.apk").write_bytes(b"x86")

    with pytest.raises(ValueError, match="no universal APK"):
        manifest.select_release_apk(apk_dir)
