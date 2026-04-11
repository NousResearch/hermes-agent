from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_chaquopy_build_preinstalls_android_stubs():
    gradle = (REPO_ROOT / "android/app/build.gradle.kts").read_text(encoding="utf-8")

    assert 'prepareHermesAndroidWheel' in gradle
    assert 'options("--no-deps")' in gradle
    assert 'install("../../android/pip-stubs/anthropic-stub")' in gradle
    assert 'install("../../android/pip-stubs/fal-client-stub")' in gradle
    assert 'install("build/hermes-wheel/${hermesWheelName()}")' in gradle
    assert 'install("-r", "../../requirements-android-chaquopy.txt")' in gradle


def test_android_wheel_includes_iteration_limits_module():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "iteration_limits" in pyproject["tool"]["setuptools"]["py-modules"]


def test_android_anthropic_stub_matches_project_requirement_floor():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    stub_project = tomllib.loads(
        (REPO_ROOT / "android/pip-stubs/anthropic-stub/pyproject.toml").read_text(encoding="utf-8")
    )

    base_anthropic = next(
        dep for dep in pyproject["project"]["dependencies"]
        if dep.startswith("anthropic>=")
    )
    assert base_anthropic.startswith(f"anthropic>={stub_project['project']['version']}")


def test_android_runtime_requirements_pin_pre_jiter_openai_sdk():
    requirements = (REPO_ROOT / "requirements-android-chaquopy.txt").read_text(encoding="utf-8")

    assert "openai==1.39.0" in requirements
    assert "httpx==0.27.2" in requirements
    assert "pydantic==1.10.24" in requirements
    assert "\nfirecrawl-py" not in requirements
    assert "\npydantic_core" not in requirements


def test_android_anthropic_stub_warns_at_runtime():
    stub_init = (REPO_ROOT / "android/pip-stubs/anthropic-stub/anthropic/__init__.py").read_text(encoding="utf-8")

    assert "not available in the Hermes Android MVP build" in stub_init
    assert "OpenAI-compatible provider" in stub_init


def test_android_fal_client_stub_marks_image_generation_deferred():
    stub_init = (REPO_ROOT / "android/pip-stubs/fal-client-stub/fal_client/__init__.py").read_text(encoding="utf-8")
    toolset_file = (REPO_ROOT / "toolsets.py").read_text(encoding="utf-8")

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
    assert '"read_file"' in android_toolset_block
    assert '"write_file"' in android_toolset_block
