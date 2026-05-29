import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "packaging" / "dobby-package"
FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "preflight"


def run_preflight(package_root):
    return subprocess.run(
        ["bash", str(package_root / "scripts" / "preflight.sh"), str(package_root)],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )


def assert_passed(result):
    assert result.returncode == 0, result.stdout + result.stderr


def assert_failed(result, expected):
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert expected in output


def copy_package(tmp_path):
    package_copy = tmp_path / "repo" / "packaging" / "dobby-package"
    package_copy.parent.mkdir(parents=True)
    shutil.copytree(PACKAGE_ROOT, package_copy, ignore=shutil.ignore_patterns(".env"))
    return package_copy


def write_runtime_env(package_root, tmp_path, overrides=None, omit=()):
    template = (FIXTURE_ROOT / "runtime.env.template").read_text(encoding="utf-8")
    env_text = template.format(
        hermes_home=tmp_path / "isolated-hermes-home",
        safe_root=tmp_path / "safe-workspace",
    )
    values = dict(line.split("=", 1) for line in env_text.splitlines() if line)
    values.update(overrides or {})

    env_lines = [f"{key}={value}" for key, value in values.items() if key not in omit]
    (package_root / "config" / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")


def replace_in_file(path, old, new):
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


def test_template_package_preflight_passes():
    result = subprocess.run(
        ["bash", "packaging/dobby-package/scripts/preflight.sh", "packaging/dobby-package"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )

    assert_passed(result)


def test_runtime_package_preflight_passes_with_isolated_staging_env(tmp_path):
    package_root = copy_package(tmp_path)
    write_runtime_env(package_root, tmp_path)

    result = run_preflight(package_root)

    assert_passed(result)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"OPENAI_API_KEY": "<MODEL_API_KEY>"}, "OPENAI_API_KEY still has a placeholder"),
        ({"WEBHOOK_SECRET": "short-stage-key"}, "WEBHOOK_SECRET must be at least 32 characters"),
        ({"GATEWAY_ALLOW_ALL_USERS": "true"}, "GATEWAY_ALLOW_ALL_USERS must be false"),
        ({"HERMES_REDACT_SECRETS": "false"}, "HERMES_REDACT_SECRETS must be true"),
        ({"DISCORD_ALLOWED_USERS": "*"}, "DISCORD_ALLOWED_USERS contains a broad allowlist value"),
        ({"DISCORD_ALLOWED_CHANNELS": "public"}, "DISCORD_ALLOWED_CHANNELS contains a broad allowlist value"),
    ],
)
def test_runtime_preflight_rejects_unsafe_env_values(tmp_path, overrides, expected):
    package_root = copy_package(tmp_path)
    write_runtime_env(package_root, tmp_path, overrides=overrides)

    result = run_preflight(package_root)

    assert_failed(result, expected)


@pytest.mark.parametrize(
    ("missing_key", "expected"),
    [
        ("HERMES_MODEL", "missing or empty env key: HERMES_MODEL"),
        ("DISCORD_BOT_TOKEN", "missing or empty env key: DISCORD_BOT_TOKEN"),
    ],
)
def test_runtime_preflight_rejects_missing_model_or_discord_keys(tmp_path, missing_key, expected):
    package_root = copy_package(tmp_path)
    write_runtime_env(package_root, tmp_path, omit={missing_key})

    result = run_preflight(package_root)

    assert_failed(result, expected)


def test_runtime_preflight_rejects_unsafe_hermes_home_paths(tmp_path):
    unsafe_path_specs = [
        "~/.hermes",
        "/Users/leo/.hermes",
        "package repo root",
        "/",
    ]

    for index, unsafe_path_spec in enumerate(unsafe_path_specs):
        case_dir = tmp_path / f"case-{index}"
        package_root = copy_package(case_dir)
        unsafe_path = (
            str(package_root.parents[1])
            if unsafe_path_spec == "package repo root"
            else unsafe_path_spec
        )
        write_runtime_env(package_root, tmp_path, overrides={"HERMES_HOME": unsafe_path})

        result = run_preflight(package_root)

        assert_failed(result, "HERMES_HOME")


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ('    - "<DISCORD_USER_ID>"', '    - "*"', "config allowlists contain broad user or channel values"),
        ("  require_signature: true", "  require_signature: false", "webhook policy must require HMAC signature"),
        ('    - "/webhooks/dobby/events"', '    - "*"', "webhook route allowlist must contain explicit non-wildcard routes"),
    ],
)
def test_runtime_preflight_rejects_unsafe_config_policy(tmp_path, old, new, expected):
    package_root = copy_package(tmp_path)
    write_runtime_env(package_root, tmp_path)
    replace_in_file(package_root / "config" / "config.example.yaml", old, new)

    result = run_preflight(package_root)

    assert_failed(result, expected)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("    default_enabled: false", "    default_enabled: true", "tool policy must not enable capabilities by default"),
        ('default_action: "deny"', 'default_action: "allow"', "tool policy must deny by default"),
    ],
)
def test_runtime_preflight_rejects_default_on_or_broad_tool_policy(tmp_path, old, new, expected):
    package_root = copy_package(tmp_path)
    write_runtime_env(package_root, tmp_path)
    replace_in_file(package_root / "config" / "tool-policy.example.yaml", old, new)

    result = run_preflight(package_root)

    assert_failed(result, expected)
