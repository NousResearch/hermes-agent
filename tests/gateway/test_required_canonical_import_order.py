"""Fresh-process contracts for privileged gateway import ordering.

The production canonical unit imports ``gateway.run`` with an exact CLI flag.
That import must not consume mutable user configuration, credentials, or
executable plugin/provider registries before the sealed config validator runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_PREFIX = "__CANONICAL_IMPORT_RESULT__="


def _write_drifted_runtime_inputs(hermes_home: Path) -> None:
    hermes_home.mkdir(parents=True)
    (hermes_home / ".env").write_text(
        "CANARY_IMPORT_SENTINEL=loaded-from-dotenv\n"
        "HERMES_MAX_ITERATIONS=61\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        textwrap.dedent(
            """
            CANARY_CONFIG_SENTINEL: loaded-from-config
            agent:
              max_turns: 777
            auxiliary:
              approval:
                provider: untrusted-provider
                model: untrusted-model
            secrets:
              bitwarden:
                enabled: false
            """
        ).lstrip(),
        encoding="utf-8",
    )


def _run_gateway_import(hermes_home: Path, *, required: bool) -> dict[str, object]:
    argv = ["hermes-gateway"]
    if required:
        argv.extend(
            [
                "--config",
                "/var/lib/hermes-gateway/.hermes/config.yaml",
                "--require-canonical-writer",
            ]
        )

    program = textwrap.dedent(
        f"""
        import json
        import os
        import sys

        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        sys.argv = json.loads(os.environ.pop("_HERMES_IMPORT_PROBE_ARGV"))

        import gateway.run as gateway_run
        import providers
        from hermes_cli import config as config_module
        from hermes_cli import env_loader

        provider_modules = sorted(
            name for name in sys.modules
            if name.startswith("plugins.model_providers.")
            or name.startswith("_hermes_user_provider_")
        )
        result = {{
            "required_quarantine": gateway_run._REQUIRED_CANONICAL_IMPORT_QUARANTINE,
            "dotenv_sentinel": os.environ.get("CANARY_IMPORT_SENTINEL"),
            "config_sentinel": os.environ.get("CANARY_CONFIG_SENTINEL"),
            "max_iterations": os.environ.get("HERMES_MAX_ITERATIONS"),
            "auxiliary_provider": os.environ.get("AUXILIARY_APPROVAL_PROVIDER"),
            "auxiliary_model": os.environ.get("AUXILIARY_APPROVAL_MODEL"),
            "external_secret_sources_applied": (
                str(os.environ["HERMES_HOME"])
                in {{str(path) for path in env_loader._APPLIED_HOMES}}
            ),
            "general_plugins_imported": "hermes_cli.plugins" in sys.modules,
            "providers_discovered": providers._discovered,
            "provider_registry": sorted(providers._REGISTRY),
            "provider_aliases": sorted(providers._ALIASES),
            "provider_modules": provider_modules,
            "profile_env_vars_resolved": config_module._profile_env_vars_injected,
            "platform_env_vars_resolved": (
                config_module._platform_plugin_env_vars_injected
            ),
            "quiet": os.environ.get("HERMES_QUIET"),
            "exec_ask": os.environ.get("HERMES_EXEC_ASK"),
            "gateway_marker": os.environ.get("_HERMES_GATEWAY"),
        }}
        print({RESULT_PREFIX!r} + json.dumps(result, sort_keys=True))
        """
    )

    env = dict(os.environ)
    for key in (
        "AUXILIARY_APPROVAL_MODEL",
        "AUXILIARY_APPROVAL_PROVIDER",
        "CANARY_CONFIG_SENTINEL",
        "CANARY_IMPORT_SENTINEL",
        "HERMES_CONFIG",
        "HERMES_EXEC_ASK",
        "HERMES_MANAGED_DIR",
        "HERMES_MAX_ITERATIONS",
        "HERMES_QUIET",
        "MESSAGING_CWD",
        "TERMINAL_CWD",
        "TERMINAL_ENV",
        "_HERMES_GATEWAY",
    ):
        env.pop(key, None)
    env.update(
        {
            "HERMES_HOME": str(hermes_home),
            "HOME": str(hermes_home.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
            "_HERMES_IMPORT_PROBE_ARGV": json.dumps(argv),
        }
    )

    completed = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"gateway import probe failed (rc={completed.returncode})\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    result_line = next(
        (
            line
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    assert result_line is not None, completed.stdout
    return json.loads(result_line.removeprefix(RESULT_PREFIX))


def test_required_canonical_import_quarantines_unvalidated_runtime_inputs(
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "mutable-hermes-home"
    _write_drifted_runtime_inputs(hermes_home)

    result = _run_gateway_import(hermes_home, required=True)

    assert result == {
        "auxiliary_model": None,
        "auxiliary_provider": None,
        "config_sentinel": None,
        "dotenv_sentinel": None,
        "exec_ask": "1",
        "external_secret_sources_applied": False,
        "gateway_marker": "1",
        "general_plugins_imported": False,
        "max_iterations": None,
        "platform_env_vars_resolved": False,
        "profile_env_vars_resolved": False,
        "provider_aliases": [],
        "provider_modules": [],
        "provider_registry": [],
        "providers_discovered": False,
        "quiet": "1",
        "required_quarantine": True,
    }


def test_normal_gateway_import_preserves_dotenv_and_config_bridge(
    tmp_path: Path,
) -> None:
    hermes_home = tmp_path / "normal-hermes-home"
    _write_drifted_runtime_inputs(hermes_home)

    result = _run_gateway_import(hermes_home, required=False)

    assert result["required_quarantine"] is False
    assert result["dotenv_sentinel"] == "loaded-from-dotenv"
    assert result["config_sentinel"] == "loaded-from-config"
    assert result["max_iterations"] == "777"
    assert result["auxiliary_provider"] == "untrusted-provider"
    assert result["auxiliary_model"] == "untrusted-model"
    assert result["external_secret_sources_applied"] is True
    assert result["gateway_marker"] == "1"
    assert result["quiet"] == "1"
    assert result["exec_ask"] == "1"


def test_config_optional_env_registry_resolves_providers_only_on_first_read() -> None:
    program = textwrap.dedent(
        f"""
        import json
        import sys

        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        import hermes_cli.config as config_module
        import providers

        before = {{
            "discovered": providers._discovered,
            "registry": sorted(providers._REGISTRY),
        }}
        gmi = config_module.OPTIONAL_ENV_VARS["GMI_API_KEY"]
        after = {{
            "discovered": providers._discovered,
            "category": gmi["category"],
            "password": gmi["password"],
        }}
        print({RESULT_PREFIX!r} + json.dumps({{"before": before, "after": after}}))
        """
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONHOME", "PYTHONPATH"}
    }
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result_line = next(
        line
        for line in completed.stdout.splitlines()
        if line.startswith(RESULT_PREFIX)
    )
    result = json.loads(result_line.removeprefix(RESULT_PREFIX))
    assert result == {
        "before": {"discovered": False, "registry": []},
        "after": {"discovered": True, "category": "provider", "password": True},
    }
