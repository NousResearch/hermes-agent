"""Relationship tests for the interactive CLI's effective config snapshot."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.fixture
def config_modules(tmp_path, monkeypatch):
    original_env = dict(os.environ)
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)

    import cli
    from hermes_cli import managed_scope
    from hermes_cli import config as canonical

    monkeypatch.setattr(cli, "_hermes_home", home)
    canonical._LOAD_CONFIG_CACHE.clear()
    canonical._RAW_CONFIG_CACHE.clear()
    canonical._LAST_EXPANDED_CONFIG_BY_PATH.clear()
    managed_scope.invalidate_managed_cache()
    try:
        yield cli, canonical, managed_scope, home, managed
    finally:
        # load_cli_config intentionally bridges effective config into os.environ
        # with direct assignments. Restore the complete process environment,
        # including keys that monkeypatch did not create itself.
        os.environ.clear()
        os.environ.update(original_env)


def _at(config, dotted_path):
    value = config
    for part in dotted_path.split("."):
        value = value[part]
    return value


@pytest.mark.parametrize(
    "dotted_path",
    [
        "agent.max_turns",
        "display.streaming",
        "delegation.max_iterations",
        "auxiliary.vision.timeout",
    ],
)
def test_cli_defaults_are_owned_by_canonical_config(config_modules, dotted_path):
    cli, canonical, _managed_scope, _home, _managed = config_modules

    cli_config = cli.load_cli_config()
    effective = canonical.load_config()

    assert _at(cli_config, dotted_path) == _at(effective, dotted_path)
    assert _at(cli_config, dotted_path) == _at(canonical.DEFAULT_CONFIG, dotted_path)


def test_cli_uses_canonical_recursive_merge_without_materializing_defaults(
    config_modules,
):
    cli, canonical, _managed_scope, home, _managed = config_modules
    raw = {
        "browser": {"camofox": {"rewrite_loopback_urls": True}},
        "auxiliary": {"vision": {"model": "test/vision"}},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    cli_config = cli.load_cli_config()
    effective = canonical.load_config()

    assert cli_config["browser"]["camofox"] == effective["browser"]["camofox"]
    assert cli_config["auxiliary"]["vision"] == effective["auxiliary"]["vision"]
    assert cli_config["browser"]["camofox"]["rewrite_loopback_urls"] is True
    assert "loopback_host_alias" in cli_config["browser"]["camofox"]
    assert canonical.read_raw_config() == raw


def test_partial_terminal_config_only_overrides_authored_env_keys(
    config_modules, monkeypatch
):
    cli, _canonical, _managed_scope, home, _managed = config_modules
    (home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "docker"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TERMINAL_TIMEOUT", "987")

    cli.load_cli_config()

    assert os.environ["TERMINAL_ENV"] == "docker"
    assert os.environ["TERMINAL_TIMEOUT"] == "987"


def test_legacy_env_type_is_not_shadowed_by_default_backend(
    config_modules, monkeypatch
):
    cli, _canonical, _managed_scope, home, _managed = config_modules
    (home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"env_type": "docker"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TERMINAL_ENV", "from-shell")

    loaded = cli.load_cli_config()

    assert loaded["terminal"]["env_type"] == "docker"
    assert os.environ["TERMINAL_ENV"] == "docker"


@pytest.mark.parametrize(
    ("user_terminal", "managed_terminal", "expected"),
    [
        ({}, {}, "local"),
        ({"env_type": "user-legacy"}, {}, "user-legacy"),
        (
            {"backend": "user-canonical", "env_type": "user-legacy"},
            {},
            "user-canonical",
        ),
        (
            {"backend": "user-canonical"},
            {"env_type": "managed-legacy"},
            "managed-legacy",
        ),
        (
            {"env_type": "user-legacy"},
            {"backend": "managed-canonical"},
            "managed-canonical",
        ),
        (
            {"backend": "user-canonical", "env_type": "user-legacy"},
            {"backend": "managed-canonical", "env_type": "managed-legacy"},
            "managed-canonical",
        ),
    ],
)
def test_terminal_backend_alias_precedence(
    config_modules,
    monkeypatch,
    user_terminal,
    managed_terminal,
    expected,
):
    cli, _canonical, managed_scope, home, managed = config_modules
    if user_terminal:
        (home / "config.yaml").write_text(
            yaml.safe_dump({"terminal": user_terminal}), encoding="utf-8"
        )
    if managed_terminal:
        (managed / "config.yaml").write_text(
            yaml.safe_dump({"terminal": managed_terminal}), encoding="utf-8"
        )
        managed_scope.invalidate_managed_cache()
    monkeypatch.setenv("TERMINAL_ENV", "from-shell")

    loaded = cli.load_cli_config()

    assert loaded["terminal"]["backend"] == expected
    assert loaded["terminal"]["env_type"] == expected
    expected_env = "from-shell" if not user_terminal and not managed_terminal else expected
    assert os.environ["TERMINAL_ENV"] == expected_env


def test_managed_backend_overrides_legacy_env_type_and_existing_env(
    config_modules, monkeypatch
):
    cli, _canonical, managed_scope, home, managed = config_modules
    (home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"env_type": "docker"}}),
        encoding="utf-8",
    )
    (managed / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"backend": "modal"}}),
        encoding="utf-8",
    )
    managed_scope.invalidate_managed_cache()
    monkeypatch.setenv("TERMINAL_ENV", "from-shell")

    loaded = cli.load_cli_config()

    assert loaded["terminal"]["env_type"] == "modal"
    assert os.environ["TERMINAL_ENV"] == "modal"


def test_managed_terminal_leaves_override_existing_env_per_leaf(
    config_modules, monkeypatch
):
    cli, _canonical, managed_scope, _home, managed = config_modules
    (managed / "config.yaml").write_text(
        yaml.safe_dump({
            "terminal": {
                "modal_mode": "managed",
                "temp_dir": "/managed/tmp",
            }
        }),
        encoding="utf-8",
    )
    managed_scope.invalidate_managed_cache()
    monkeypatch.setenv("TERMINAL_MODAL_MODE", "direct")
    monkeypatch.setenv("TERMINAL_TEMP_DIR", "/shell/tmp")
    monkeypatch.setenv("TERMINAL_TIMEOUT", "987")

    cli.load_cli_config()

    assert os.environ["TERMINAL_MODAL_MODE"] == "managed"
    assert os.environ["TERMINAL_TEMP_DIR"] == "/managed/tmp"
    assert os.environ["TERMINAL_TIMEOUT"] == "987"


def test_cli_uses_canonical_terminal_mapping_for_modal_mode_and_temp_dir(
    config_modules,
):
    cli, _canonical, _managed_scope, home, _managed = config_modules
    (home / "config.yaml").write_text(
        yaml.safe_dump({"terminal": {"modal_mode": "direct", "temp_dir": "/user/tmp"}}),
        encoding="utf-8",
    )

    cli.load_cli_config()

    assert os.environ["TERMINAL_MODAL_MODE"] == "direct"
    assert os.environ["TERMINAL_TEMP_DIR"] == "/user/tmp"


def test_effective_and_raw_snapshot_retry_across_atomic_replacement(
    config_modules, monkeypatch
):
    _cli, canonical, _managed_scope, home, _managed = config_modules
    config_path = home / "config.yaml"
    replacement = home / "config.yaml.next"
    config_path.write_text("display:\n  skin: first\n", encoding="utf-8")
    replacement.write_text("display:\n  skin: other\n", encoding="utf-8")
    original_read_raw = canonical.read_user_config_raw
    reads = 0

    def replace_before_first_raw_read(path):
        nonlocal reads
        reads += 1
        if reads == 1:
            os.replace(replacement, config_path)
        return original_read_raw(path)

    monkeypatch.setattr(canonical, "read_user_config_raw", replace_before_first_raw_read)

    effective, raw = canonical.load_config_snapshot_with_raw(config_path)

    assert reads == 2
    assert effective["display"]["skin"] == "other"
    assert raw["display"]["skin"] == "other"


def test_effective_and_raw_snapshot_preserves_lkg_on_parse_failure(config_modules):
    _cli, canonical, _managed_scope, home, _managed = config_modules
    config_path = home / "config.yaml"
    config_path.write_text("display:\n  skin: known-good\n", encoding="utf-8")
    first_effective, first_raw = canonical.load_config_snapshot_with_raw(config_path)
    assert first_effective["display"]["skin"] == "known-good"
    assert first_raw["display"]["skin"] == "known-good"

    config_path.write_text("display:\n  skin: [unclosed\n", encoding="utf-8")
    effective, raw = canonical.load_config_snapshot_with_raw(config_path)

    assert effective["display"]["skin"] == "known-good"
    assert raw == {}


def test_cli_calls_canonical_terminal_bridge(config_modules, monkeypatch):
    cli, canonical, _managed_scope, _home, _managed = config_modules
    calls = []
    original = canonical.apply_terminal_config_to_env

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(canonical, "apply_terminal_config_to_env", spy)

    cli.load_cli_config()

    assert len(calls) == 1
    assert calls[0]["config"]["terminal"]
    assert calls[0]["authoritative_keys"] is not None


def test_explicit_relative_snapshot_paths_have_stable_distinct_cache_identity(
    config_modules, monkeypatch, tmp_path
):
    _cli, canonical, _managed_scope, _home, _managed = config_modules
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    # Equal-size files plus an identical timestamp reproduce a relative cache-key
    # collision when the loader keys only on the spelling "config.yaml".
    first_config = first / "config.yaml"
    second_config = second / "config.yaml"
    first_config.write_text("display:\n  skin: first\n", encoding="utf-8")
    second_config.write_text("display:\n  skin: other\n", encoding="utf-8")
    timestamp_ns = 1_700_000_000_000_000_000
    os.utime(first_config, ns=(timestamp_ns, timestamp_ns))
    os.utime(second_config, ns=(timestamp_ns, timestamp_ns))

    monkeypatch.chdir(first)
    first_loaded = canonical.load_config_snapshot(Path("config.yaml"))
    monkeypatch.chdir(second)
    second_loaded = canonical.load_config_snapshot(Path("config.yaml"))

    assert first_loaded["display"]["skin"] == "first"
    assert second_loaded["display"]["skin"] == "other"
    assert {str(first_config.resolve()), str(second_config.resolve())} <= set(
        canonical._LOAD_CONFIG_CACHE
    )


def test_cli_model_dict_has_complete_compatibility_shape_without_overrides(
    config_modules,
):
    cli, _canonical, _managed_scope, home, _managed = config_modules
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {
                "default": "user/model",
                "provider": "user-provider",
            }
        }),
        encoding="utf-8",
    )

    model = cli.load_cli_config()["model"]

    assert {"default", "base_url", "provider"} <= model.keys()
    assert model["default"] == "user/model"
    assert model["base_url"] == ""
    assert model["provider"] == "user-provider"


def test_cli_matches_canonical_env_expansion_and_managed_precedence(
    config_modules, monkeypatch
):
    cli, canonical, managed_scope, home, managed = config_modules
    monkeypatch.setenv("USER_VISION_MODEL", "test/user-vision")
    monkeypatch.setenv("MANAGED_VISION_MODEL", "test/managed-vision")
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "auxiliary": {
                "vision": {
                    "model": "${USER_VISION_MODEL}",
                    "provider": "user-provider",
                }
            }
        }),
        encoding="utf-8",
    )
    (managed / "config.yaml").write_text(
        yaml.safe_dump({
            "auxiliary": {"vision": {"model": "${env:MANAGED_VISION_MODEL}"}}
        }),
        encoding="utf-8",
    )
    managed_scope.invalidate_managed_cache()

    cli_config = cli.load_cli_config()
    effective = canonical.load_config()

    assert cli_config["auxiliary"]["vision"] == effective["auxiliary"]["vision"]
    assert cli_config["auxiliary"]["vision"]["model"] == "test/managed-vision"
    assert cli_config["auxiliary"]["vision"]["provider"] == "user-provider"
    assert (
        canonical.read_raw_config()["auxiliary"]["vision"]["model"]
        == "${USER_VISION_MODEL}"
    )


def test_ignore_user_config_keeps_project_fallback_on_canonical_defaults(
    config_modules, monkeypatch, tmp_path
):
    cli, canonical, _managed_scope, home, _managed = config_modules
    (home / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"system_prompt": "must-not-leak"}}),
        encoding="utf-8",
    )
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(cli, "__file__", str(project / "cli.py"))
    (project / "cli-config.yaml").write_text(
        yaml.safe_dump({
            "display": {"compact": True},
            "browser": {"camofox": {"rewrite_loopback_urls": True}},
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_IGNORE_USER_CONFIG", "1")

    cli_config = cli.load_cli_config()

    assert cli_config["agent"].get("system_prompt") != "must-not-leak"
    assert cli_config["display"]["compact"] is True
    assert cli_config["browser"]["camofox"]["rewrite_loopback_urls"] is True
    assert (
        cli_config["browser"]["camofox"]["loopback_host_alias"]
        == canonical.DEFAULT_CONFIG["browser"]["camofox"]["loopback_host_alias"]
    )


def test_import_time_cli_config_isolated_and_preserves_parent_environment(tmp_path):
    home = tmp_path / "subprocess-home"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "model": {"default": "import/model"},
            "terminal": {"backend": "docker"},
        }),
        encoding="utf-8",
    )
    child_env = copy.deepcopy(dict(os.environ))
    child_env.update({
        "HERMES_HOME": str(home),
        "HERMES_MANAGED_DIR": str(tmp_path / "managed"),
        "TERMINAL_TIMEOUT": "654",
    })
    parent_env = dict(os.environ)
    script = """
import json
import os
import cli
print("CONFIG_RESULT=" + json.dumps({
    "model": cli.CLI_CONFIG["model"],
    "terminal_env": os.environ.get("TERMINAL_ENV"),
    "terminal_timeout": os.environ.get("TERMINAL_TIMEOUT"),
}, sort_keys=True))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
    )
    result_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith("CONFIG_RESULT=")
    )
    result = json.loads(result_line.removeprefix("CONFIG_RESULT="))

    assert {"default", "base_url", "provider"} <= result["model"].keys()
    assert result["model"]["default"] == "import/model"
    assert result["terminal_env"] == "docker"
    assert result["terminal_timeout"] == "654"
    assert dict(os.environ) == parent_env
