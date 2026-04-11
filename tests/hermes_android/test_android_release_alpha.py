from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_android_build_gradle_supports_semver_alpha_release_tags():
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")

    assert 'fun androidVersionName()' in gradle
    assert 'v?(\\d+)\\.(\\d+)\\.(\\d+)(?:-([A-Za-z]+)(?:[.-]?(\\d+))?)?' in gradle
    assert '"alpha" -> 1' in gradle
    assert '"beta" -> 2' in gradle
    assert 'versionName = androidVersionName()' in gradle


def test_android_release_workflow_restores_signing_material_and_builds_release_artifacts():
    workflow = (REPO_ROOT / ".github/workflows/android-release.yml").read_text(encoding="utf-8")

    assert 'ANDROID_KEYSTORE_BASE64' in workflow
    assert 'ANDROID_KEYSTORE_PASSWORD' in workflow
    assert 'ANDROID_KEY_ALIAS' in workflow
    assert 'ANDROID_KEY_PASSWORD' in workflow
    assert './gradlew :app:assembleRelease :app:bundleRelease' in workflow
    assert 'scripts/android_release_manifest.py --tag' in workflow
