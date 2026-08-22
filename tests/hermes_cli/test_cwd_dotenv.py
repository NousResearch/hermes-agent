"""Security and entrypoint coverage for trusted CWD dotenv loading."""

from pathlib import Path
import os
import subprocess
import sys

import pytest

from hermes_cli import env_loader
from hermes_cli.env_loader import load_hermes_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINTS = [
    "cli.py",
    "run_agent.py",
    "hermes_cli/main.py",
    "acp_adapter/entry.py",
    "tui_gateway/server.py",
]


def _configure_trusted_cwds(home: Path, *paths: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    entries = "\n".join(f'    - "{path}"' for path in paths)
    (home / "config.yaml").write_text(
        f"dotenv:\n  trusted_cwds:\n{entries}\n",
        encoding="utf-8",
    )


def test_cwd_dotenv_is_disabled_without_explicit_trust(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    env_file = project / ".env"
    env_file.write_text("PROJECT_ONLY_KEY=blocked\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.delenv("PROJECT_ONLY_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, cwd_env=True)

    assert loaded == []
    assert os.getenv("PROJECT_ONLY_KEY") is None


def test_trusted_cwd_loads_allowed_values_with_user_precedence(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _configure_trusted_cwds(home, project)
    user_env = home / ".env"
    user_env.write_text("SHARED_KEY=user\n", encoding="utf-8")
    cwd_env = project / ".env"
    cwd_env.write_text("SHARED_KEY=project\nPROJECT_API_KEY=local\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.delenv("SHARED_KEY", raising=False)
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, cwd_env=True)

    assert loaded == [user_env, cwd_env]
    assert os.getenv("SHARED_KEY") == "user"
    assert os.getenv("PROJECT_API_KEY") == "local"


def test_user_env_ownership_cannot_be_bypassed_with_case_variant(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _configure_trusted_cwds(home, project)
    (home / ".env").write_text("OPENAI_API_KEY=user\n", encoding="utf-8")
    (project / ".env").write_text("openai_api_key=attacker\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("openai_api_key", raising=False)

    load_hermes_dotenv(
        hermes_home=home,
        cwd_env=True,
        cwd_path=project,
    )

    assert os.getenv("OPENAI_API_KEY") == "user"
    assert os.getenv("openai_api_key") is None


def test_trusted_cwd_overrides_stale_shell_without_user_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _configure_trusted_cwds(home, project)
    cwd_env = project / ".env"
    cwd_env.write_text("PROJECT_API_KEY=project\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setenv("PROJECT_API_KEY", "stale-shell")

    loaded = load_hermes_dotenv(hermes_home=home, cwd_env=True)

    assert loaded == [cwd_env]
    assert os.getenv("PROJECT_API_KEY") == "project"


def test_relative_trust_entries_are_rejected(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    home.mkdir()
    (home / "config.yaml").write_text(
        "dotenv:\n  trusted_cwds:\n    - ../project\n",
        encoding="utf-8",
    )
    (project / ".env").write_text("PROJECT_API_KEY=blocked\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    assert load_hermes_dotenv(hermes_home=home, cwd_env=True) == []
    assert os.getenv("PROJECT_API_KEY") is None


@pytest.mark.parametrize(
    "trust_entry",
    ["$PWD", "${PWD}", "~/project", "$UNDEFINED_TRUST_ROOT/project"],
)
def test_mutable_or_expanding_trust_entries_are_ignored(
    tmp_path, monkeypatch, trust_entry
):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "config.yaml").write_text(
        "dotenv:\n  trusted_cwds:\n" f'    - "{trust_entry}"\n',
        encoding="utf-8",
    )
    (project / ".env").write_text("PROJECT_API_KEY=unsafe\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setenv("PWD", str(project))
    monkeypatch.setenv("UNDEFINED_TRUST_ROOT", str(tmp_path))
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, cwd_env=True)

    assert loaded == []
    assert os.getenv("PROJECT_API_KEY") is None


def test_cwd_values_do_not_interpolate_protected_ambient_secrets(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _configure_trusted_cwds(home, project)
    (project / ".env").write_text(
        "ATTACKER_API_KEY=${SUDO_PASSWORD}\n", encoding="utf-8"
    )
    monkeypatch.setenv("SUDO_PASSWORD", "do-not-copy")
    monkeypatch.delenv("ATTACKER_API_KEY", raising=False)

    load_hermes_dotenv(hermes_home=home, cwd_env=True, cwd_path=project)

    assert os.getenv("ATTACKER_API_KEY") == "${SUDO_PASSWORD}"
    assert os.getenv("ATTACKER_API_KEY") != "do-not-copy"


@pytest.mark.parametrize(
    "name",
    [
        "_HERMES_GATEWAY",
        "SECURITY_GUIDANCE_DISABLE",
        "AUXILIARY_APPROVAL_PROVIDER",
        "AUXILIARY_APPROVAL_MODEL",
        "AUXILIARY_APPROVAL_BASE_URL",
        "AUXILIARY_APPROVAL_API_KEY",
        "BROWSER_CDP_URL",
        "ALL_PROXY",
        "WSS_PROXY",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "BASHOPTS",
        "SHELLOPTS",
        "NODE_PATH",
        "PERL5LIB",
        "RUBYLIB",
        "_JAVA_OPTIONS",
        "JDK_JAVA_OPTIONS",
    ],
)
def test_additional_security_and_process_controls_are_blocked(
    tmp_path, monkeypatch, name
):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _configure_trusted_cwds(home, project)
    (project / ".env").write_text(f"{name}=attacker\n", encoding="utf-8")
    monkeypatch.delenv(name, raising=False)

    load_hermes_dotenv(hermes_home=home, cwd_env=True, cwd_path=project)

    assert os.getenv(name) is None


def test_symlinked_cwd_env_is_rejected(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _configure_trusted_cwds(home, project)
    outside = tmp_path / "outside.env"
    outside.write_text("PROJECT_API_KEY=outside\n", encoding="utf-8")
    try:
        (project / ".env").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable on this platform")
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    loaded = load_hermes_dotenv(hermes_home=home, cwd_env=True, cwd_path=project)

    assert loaded == []
    assert os.getenv("PROJECT_API_KEY") is None


def test_four_source_precedence_is_explicit(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _configure_trusted_cwds(home, project)
    (home / ".env").write_text(
        "SHARED=user\nUSER_OWNED=user\n", encoding="utf-8"
    )
    (project / ".env").write_text(
        "SHARED=cwd\nCWD_BEATS_SHELL=cwd\nCWD_ONLY=cwd\n", encoding="utf-8"
    )
    source_tree_env = tmp_path / "source-tree.env"
    source_tree_env.write_text(
        "SHARED=source\n"
        "CWD_BEATS_SHELL=source\n"
        "SHELL_STAYS=source\n"
        "SOURCE_ONLY=source\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SHARED", "shell")
    monkeypatch.setenv("CWD_BEATS_SHELL", "shell")
    monkeypatch.setenv("SHELL_STAYS", "shell")
    monkeypatch.delenv("USER_OWNED", raising=False)
    monkeypatch.delenv("CWD_ONLY", raising=False)
    monkeypatch.delenv("SOURCE_ONLY", raising=False)

    loaded = load_hermes_dotenv(
        hermes_home=home,
        cwd_env=True,
        cwd_path=project,
        project_env=source_tree_env,
    )

    assert loaded == [home / ".env", project / ".env", source_tree_env]
    assert os.getenv("SHARED") == "user"
    assert os.getenv("USER_OWNED") == "user"
    assert os.getenv("CWD_BEATS_SHELL") == "cwd"
    assert os.getenv("CWD_ONLY") == "cwd"
    assert os.getenv("SHELL_STAYS") == "shell"
    assert os.getenv("SOURCE_ONLY") == "source"


def test_protected_process_and_hermes_variables_are_never_loaded(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _configure_trusted_cwds(home, project)
    cwd_env = project / ".env"
    cwd_env.write_text(
        "PROJECT_API_KEY=allowed\n"
        "HERMES_YOLO_MODE=1\n"
        "HERMES_REDACT_SECRETS=false\n"
        "TERMINAL_CWD=/tmp/escape\n"
        "PYTHONPATH=/tmp/inject\n"
        "PATH=/tmp/bin\n"
        "SUDO_PASSWORD=secret-value-must-not-be-printed\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(project)
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setenv("HERMES_REDACT_SECRETS", "true")
    original_path = os.environ.get("PATH")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(env_loader, "_WARNED_CWD_ENV_KEYS", set())

    loaded = load_hermes_dotenv(hermes_home=home, cwd_env=True)

    assert loaded == [cwd_env]
    assert os.getenv("PROJECT_API_KEY") == "allowed"
    assert os.getenv("HERMES_YOLO_MODE") is None
    assert os.getenv("HERMES_REDACT_SECRETS") == "true"
    assert os.getenv("TERMINAL_CWD") is None
    assert os.getenv("PYTHONPATH") is None
    assert os.getenv("PATH") == original_path
    assert os.getenv("SUDO_PASSWORD") is None
    warning = capsys.readouterr().err
    assert "HERMES_YOLO_MODE" in warning
    assert "HERMES_REDACT_SECRETS" in warning
    assert "secret-value-must-not-be-printed" not in warning


def test_explicit_cwd_path_uses_same_exact_trust_policy(tmp_path, monkeypatch):
    home = tmp_path / "home"
    trusted = tmp_path / "trusted"
    elsewhere = tmp_path / "elsewhere"
    trusted.mkdir()
    elsewhere.mkdir()
    _configure_trusted_cwds(home, trusted)
    env_file = trusted / ".env"
    env_file.write_text("PROJECT_API_KEY=trusted\n", encoding="utf-8")
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    loaded = load_hermes_dotenv(
        hermes_home=home,
        cwd_env=True,
        cwd_path=trusted,
    )

    assert loaded == [env_file]
    assert os.getenv("PROJECT_API_KEY") == "trusted"


def test_gateway_terminal_cwd_is_the_canonical_project_directory(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    project = tmp_path / "configured-project"
    launch_dir = tmp_path / "service-launch"
    home.mkdir()
    project.mkdir()
    launch_dir.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n"
        f'  cwd: "{project}"\n'
        "dotenv:\n"
        "  trusted_cwds:\n"
        f'    - "{project}"\n',
        encoding="utf-8",
    )
    env_file = project / ".env"
    env_file.write_text("PROJECT_API_KEY=configured-cwd\n", encoding="utf-8")
    monkeypatch.chdir(launch_dir)
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    loaded = load_hermes_dotenv(
        hermes_home=home,
        cwd_env=True,
        use_terminal_cwd=True,
    )

    assert loaded == [env_file]
    assert os.getenv("PROJECT_API_KEY") == "configured-cwd"


def test_gateway_placeholder_reads_messaging_cwd_from_user_env(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "messaging-project"
    home.mkdir()
    project.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: local\n"
        "  cwd: .\n"
        "dotenv:\n"
        "  trusted_cwds:\n"
        f'    - "{project}"\n',
        encoding="utf-8",
    )
    (home / ".env").write_text(
        f"MESSAGING_CWD={project}\n", encoding="utf-8"
    )
    project_env = project / ".env"
    project_env.write_text("PROJECT_API_KEY=messaging-cwd\n", encoding="utf-8")
    monkeypatch.delenv("MESSAGING_CWD", raising=False)
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    loaded = load_hermes_dotenv(
        hermes_home=home,
        cwd_env=True,
        use_terminal_cwd=True,
    )

    assert loaded == [home / ".env", project_env]
    assert os.getenv("PROJECT_API_KEY") == "messaging-cwd"


def test_gateway_owned_marker_and_terminal_cwd_survive_user_env_reload(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(
        "_HERMES_GATEWAY=0\nTERMINAL_CWD=/stale/from/dotenv\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.setenv("TERMINAL_CWD", "/canonical/gateway/cwd")

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("_HERMES_GATEWAY") == "1"
    assert os.getenv("TERMINAL_CWD") == "/canonical/gateway/cwd"


def test_gateway_owned_terminal_cwd_absence_survives_initial_user_env_load(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text(
        "TERMINAL_CWD=/stale/from/dotenv\n", encoding="utf-8"
    )
    monkeypatch.setenv("_HERMES_GATEWAY", "1")
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("_HERMES_GATEWAY") == "1"
    assert os.getenv("TERMINAL_CWD") is None


def test_user_env_cannot_create_internal_gateway_context(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".env").write_text("_HERMES_GATEWAY=1\n", encoding="utf-8")
    monkeypatch.delenv("_HERMES_GATEWAY", raising=False)

    load_hermes_dotenv(hermes_home=home)

    assert os.getenv("_HERMES_GATEWAY") is None


def test_gateway_cwd_resolution_applies_managed_terminal_overlay(
    tmp_path, monkeypatch
):
    from hermes_cli import managed_scope

    home = tmp_path / "home"
    user_project = tmp_path / "user-project"
    managed_project = tmp_path / "managed-project"
    home.mkdir()
    user_project.mkdir()
    managed_project.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: local\n"
        f'  cwd: "{user_project}"\n'
        "dotenv:\n"
        "  trusted_cwds:\n"
        f'    - "{managed_project}"\n',
        encoding="utf-8",
    )
    managed_env = managed_project / ".env"
    managed_env.write_text("PROJECT_API_KEY=managed-cwd\n", encoding="utf-8")

    def apply_overlay(config):
        merged = dict(config)
        merged["terminal"] = {
            "backend": "local",
            "cwd": str(managed_project),
        }
        return merged

    monkeypatch.setattr(managed_scope, "apply_managed_overlay", apply_overlay)
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    loaded = load_hermes_dotenv(
        hermes_home=home,
        cwd_env=True,
        use_terminal_cwd=True,
    )

    assert loaded == [managed_env]
    assert os.getenv("PROJECT_API_KEY") == "managed-cwd"


def test_local_cli_uses_launch_cwd_not_explicit_terminal_cwd(tmp_path, monkeypatch):
    home = tmp_path / "home"
    launch_project = tmp_path / "launch-project"
    configured_project = tmp_path / "configured-project"
    home.mkdir()
    launch_project.mkdir()
    configured_project.mkdir()
    (home / "config.yaml").write_text(
        "terminal:\n"
        "  backend: local\n"
        f'  cwd: "{configured_project}"\n'
        "dotenv:\n"
        "  trusted_cwds:\n"
        f'    - "{launch_project}"\n'
        f'    - "{configured_project}"\n',
        encoding="utf-8",
    )
    (launch_project / ".env").write_text(
        "PROJECT_API_KEY=launch-cwd\n", encoding="utf-8"
    )
    (configured_project / ".env").write_text(
        "PROJECT_API_KEY=configured-cwd\n", encoding="utf-8"
    )
    monkeypatch.chdir(launch_project)
    monkeypatch.delenv("PROJECT_API_KEY", raising=False)

    load_hermes_dotenv(hermes_home=home, cwd_env=True)

    assert os.getenv("PROJECT_API_KEY") == "launch-cwd"


def test_import_time_approval_and_redaction_snapshots_cannot_be_overridden(
    tmp_path,
):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project.mkdir()
    _configure_trusted_cwds(home, project)
    (project / ".env").write_text(
        "HERMES_YOLO_MODE=1\nHERMES_REDACT_SECRETS=false\n",
        encoding="utf-8",
    )
    code = f"""
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(hermes_home={str(home)!r}, cwd_env=True, cwd_path={str(project)!r})
from tools.approval import _YOLO_MODE_FROZEN
from agent.redact import _REDACT_ENABLED
assert _YOLO_MODE_FROZEN is False
assert _REDACT_ENABLED is True
"""
    child_env = os.environ.copy()
    child_env.pop("HERMES_YOLO_MODE", None)
    child_env["HERMES_REDACT_SECRETS"] = "true"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_all_supported_ingress_and_gateway_reload_enable_safe_cwd_loading():
    for relative_path in ENTRYPOINTS:
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "cwd_env=True" in source, relative_path

    gateway_source = (REPO_ROOT / "gateway" / "run.py").read_text(encoding="utf-8")
    assert gateway_source.count("cwd_env=True") >= 2
    assert gateway_source.count("use_terminal_cwd=True") >= 2
    reload_start = gateway_source.index(
        "def _reload_runtime_env_preserving_config_authority"
    )
    reload_end = gateway_source.index(
        "def _bridge_max_turns_from_config", reload_start
    )
    assert "cwd_env=True" in gateway_source[reload_start:reload_end]
    assert "use_terminal_cwd=True" in gateway_source[reload_start:reload_end]

    for relative_path in ("cli.py", "run_agent.py"):
        source = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert 'os.environ.get("_HERMES_GATEWAY") == "1"' in source
        assert "use_terminal_cwd=_dotenv_gateway_context" in source
