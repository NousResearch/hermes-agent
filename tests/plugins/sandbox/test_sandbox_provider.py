import argparse
import re
import subprocess


def test_discover_and_load_bundled_openshell_provider():
    from plugins.sandbox import discover_plugin_cli_commands, discover_sandbox_providers, load_sandbox_provider

    providers = {name: desc for name, desc, _available in discover_sandbox_providers()}
    assert "openshell" in providers

    provider = load_sandbox_provider("openshell")
    assert provider is not None
    assert provider.name == "openshell"

    commands = {entry["name"]: entry for entry in discover_plugin_cli_commands()}
    assert "openshell" in commands
    assert callable(commands["openshell"]["setup_fn"])


def test_openshell_cli_registers_expected_subcommands():
    from plugins.sandbox.openshell.cli import register_cli

    parser = argparse.ArgumentParser()
    register_cli(parser)

    args = parser.parse_args(["enable", "--default", "--mode", "remote", "--no-smoke"])
    assert args.openshell_command == "enable"
    assert args.default is True
    assert args.mode == "remote"
    assert args.no_smoke is True

    args = parser.parse_args(["exec", "--scope", "abc", "--gateway-name", "gw", "--gateway-port", "18081", "--", "pwd"])
    assert args.openshell_command == "exec"
    assert args.scope == "abc"
    assert args.gateway_name == "gw"
    assert args.gateway_port == 18081
    assert args.exec_command == ["--", "pwd"]


def test_build_sandbox_name_is_kubernetes_safe():
    from plugins.sandbox.openshell.environment import build_sandbox_name

    name = build_sandbox_name("20260425_130221_bab5f1.a95cd9c1")

    assert name.startswith("hermes-20260425-130221-bab5f1-a95cd9c1-")
    assert len(name) <= 63
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name)
    assert "_" not in name
    assert "." not in name


def test_build_sandbox_name_trims_long_scope_to_kubernetes_limit():
    from plugins.sandbox.openshell.environment import build_sandbox_name

    name = build_sandbox_name("...A" * 80)

    assert len(name) <= 63
    assert re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name)
    assert name.startswith("hermes-a-a")


def test_prepare_openshell_source_imports_stable_image_for_local_gateway(tmp_path, monkeypatch):
    from plugins.sandbox.openshell import environment as openshell_env

    source = tmp_path / "source"
    source.mkdir()
    (source / "Dockerfile").write_text("FROM alpine:3.20\nCOPY start.sh /start.sh\n", encoding="utf-8")
    (source / "start.sh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(openshell_env.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(openshell_env, "_resolve_gateway_container", lambda docker, gateway: "openshell-cluster-test")
    monkeypatch.setattr(openshell_env, "_gateway_container_has_image", lambda docker, container, tag: False)
    monkeypatch.setattr(openshell_env, "_import_image_into_gateway", lambda docker, container, tag, timeout: calls.append(("import", tag)) or True)

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        if argv[:3] == ["/usr/bin/docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(openshell_env.subprocess, "run", fake_run)

    tag = openshell_env.prepare_openshell_source(
        {"source": str(source), "gateway": "test", "timeout_seconds": 120}
    )

    assert tag.startswith("openshell/sandbox-from:hermes-")
    assert calls[0] == ("/usr/bin/docker", "image", "inspect", tag)
    assert calls[1] == (
        "/usr/bin/docker",
        "build",
        "-t",
        tag,
        "-f",
        str(source / "Dockerfile"),
        str(source),
    )
    assert calls[2] == ("import", tag)


def test_prepare_openshell_source_reuses_gateway_imported_image(tmp_path, monkeypatch):
    from plugins.sandbox.openshell import environment as openshell_env

    source = tmp_path / "source"
    source.mkdir()
    (source / "Dockerfile").write_text("FROM alpine:3.20\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(openshell_env.shutil, "which", lambda name: "/usr/bin/docker" if name == "docker" else None)
    monkeypatch.setattr(openshell_env, "_resolve_gateway_container", lambda docker, gateway: "openshell-cluster-test")
    monkeypatch.setattr(openshell_env, "_gateway_container_has_image", lambda docker, container, tag: True)
    monkeypatch.setattr(openshell_env, "_import_image_into_gateway", lambda *args, **kwargs: calls.append("import") or True)

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(openshell_env.subprocess, "run", fake_run)

    tag = openshell_env.prepare_openshell_source(
        {"source": str(source), "gateway": "test", "timeout_seconds": 120}
    )

    assert tag.startswith("openshell/sandbox-from:hermes-")
    assert calls == [("/usr/bin/docker", "image", "inspect", tag)]


def test_prepare_openshell_source_falls_back_for_remote_gateway(tmp_path, monkeypatch):
    from plugins.sandbox.openshell import environment as openshell_env

    source = tmp_path / "source"
    source.mkdir()
    (source / "Dockerfile").write_text("FROM alpine:3.20\n", encoding="utf-8")
    monkeypatch.setattr(openshell_env.shutil, "which", lambda name: "/usr/bin/docker")

    assert (
        openshell_env.prepare_openshell_source(
            {"source": str(source), "gateway_endpoint": "https://gateway.example", "timeout_seconds": 120}
        )
        == str(source)
    )


def test_run_openshell_cli_builds_gateway_argv(monkeypatch):
    from plugins.sandbox.openshell.environment import normalize_openshell_config, run_openshell_cli

    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = normalize_openshell_config(
        {
            "command": "openshell",
            "gateway": "local-gw",
            "gateway_endpoint": "http://127.0.0.1:8080",
        }
    )
    run_openshell_cli(cfg, ["sandbox", "get", "demo"], cwd="/tmp", timeout=9)

    assert seen["argv"] == [
        "openshell",
        "--gateway-endpoint",
        "http://127.0.0.1:8080",
        "sandbox",
        "get",
        "demo",
    ]
    assert seen["kwargs"]["cwd"] == "/tmp"
    assert seen["kwargs"]["timeout"] == 9


def test_openshell_create_argv_uses_default_keep(monkeypatch):
    from plugins.sandbox.openshell import cli as openshell_cli

    seen = []

    def fake_run(cfg, args, **kwargs):
        seen.append(args)
        if args[:2] == ["sandbox", "get"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(openshell_cli, "run_openshell_cli", fake_run)
    monkeypatch.setattr(openshell_cli, "ensure_openshell_gateway", lambda cfg, **kwargs: None)

    cfg = {
        "source": "/tmp/source",
        "policy": "",
        "gpu": False,
        "auto_providers": False,
        "providers": ["anthropic"],
        "timeout_seconds": 120,
    }
    result = openshell_cli._ensure_sandbox(cfg, "hermes-test")

    assert result.returncode == 0
    assert seen[1] == [
        "sandbox",
        "create",
        "--name",
        "hermes-test",
        "--from",
        "/tmp/source",
        "--no-auto-providers",
        "--provider",
        "anthropic",
        "--",
        "true",
    ]


def test_openshell_enable_detects_existing_provider_config():
    from plugins.sandbox.openshell import cli as openshell_cli

    updates = {
        "sandbox.enabled": False,
        "sandbox.provider": "openshell",
        "sandbox.mode": "mirror",
        "sandbox.openshell.command": "openshell",
        "sandbox.openshell.from": "/tmp/source",
        "sandbox.openshell.policy": "",
        "sandbox.openshell.providers": [],
        "sandbox.openshell.auto_providers": True,
        "sandbox.openshell.gpu": False,
        "sandbox.openshell.gateway": "hermes-openshell",
        "sandbox.openshell.gateway_endpoint": "",
        "sandbox.openshell.gateway_port": 18080,
        "sandbox.openshell.gateway_host": "",
        "sandbox.openshell.remote_workspace_dir": "/sandbox",
        "sandbox.openshell.remote_agent_workspace_dir": "/agent",
        "sandbox.openshell.sync_hermes_files": False,
        "sandbox.openshell.extra_syncs": [],
    }
    current = {
        "sandbox": {
            "enabled": False,
            "provider": "openshell",
            "mode": "mirror",
            "openshell": {
                "command": "openshell",
                "from": "/tmp/source",
                "policy": "",
                "providers": [],
                "auto_providers": True,
                "gpu": False,
                "gateway": "hermes-openshell",
                "gateway_endpoint": "",
                "gateway_port": 18080,
                "gateway_host": "",
                "remote_workspace_dir": "/sandbox",
                "remote_agent_workspace_dir": "/agent",
                "sync_hermes_files": False,
                "extra_syncs": [],
            },
        }
    }

    assert openshell_cli._is_already_configured(current, updates)
    updates["sandbox.enabled"] = True
    assert openshell_cli._is_already_configured(current, updates)
    updates["sandbox.openshell.gateway_port"] = 18081
    assert not openshell_cli._is_already_configured(current, updates)


def test_ensure_gateway_uses_configured_port(monkeypatch):
    from plugins.sandbox.openshell.environment import ensure_openshell_gateway, normalize_openshell_config

    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    cfg = normalize_openshell_config(
        {
            "command": "openshell",
            "gateway": "hermes-test",
            "gateway_port": 18081,
            "gateway_host": "host.docker.internal",
        }
    )
    result = ensure_openshell_gateway(cfg)

    assert result.returncode == 0
    assert seen["argv"] == [
        "openshell",
        "gateway",
        "start",
        "--name",
        "hermes-test",
        "--port",
        "18081",
        "--gateway-host",
        "host.docker.internal",
    ]


def test_openshell_cli_config_honors_env_and_args(monkeypatch):
    from plugins.sandbox.openshell import cli as openshell_cli

    monkeypatch.setattr(
        openshell_cli,
        "_load_raw_config",
        lambda: {
            "sandbox": {
                "mode": "mirror",
                "openshell": {
                    "gateway": "saved-gw",
                    "gateway_port": 18080,
                    "from": "saved-source",
                },
            }
        },
    )
    monkeypatch.setenv("HERMES_OPENSHELL_GATEWAY", "env-gw")
    monkeypatch.setenv("HERMES_OPENSHELL_GATEWAY_PORT", "18082")
    args = argparse.Namespace(
        source="cli-source",
        gateway_name="cli-gw",
        gateway_port=18083,
        gateway_endpoint="",
        gateway_host="",
        timeout_seconds=None,
    )

    cfg = openshell_cli._load_openshell_config(args)

    assert cfg["source"] == "cli-source"
    assert cfg["gateway"] == "cli-gw"
    assert cfg["gateway_port"] == 18083


def test_normalize_openshell_config_parses_extra_syncs(tmp_path):
    from plugins.sandbox.openshell.environment import normalize_openshell_config

    extra = tmp_path / "worktree-a"
    extra.mkdir()

    cfg = normalize_openshell_config(
        {
            "extra_syncs": [
                {"host": str(extra), "remote": "/worktrees/a", "mode": "mirror"},
                f"{extra}:/worktrees/b",
            ]
        }
    )

    assert cfg["extra_syncs"][0]["host"] == str(extra)
    assert cfg["extra_syncs"][0]["remote"] == "/worktrees/a"
    assert cfg["extra_syncs"][0]["mode"] == "mirror"
    assert cfg["extra_syncs"][1]["remote"] == "/worktrees/b"


def test_openshell_exec_uses_ssh_config(monkeypatch):
    from plugins.sandbox.openshell import cli as openshell_cli

    seen = {"openshell": [], "subprocess": []}

    def fake_load():
        return {"timeout_seconds": 120}

    def fake_run_for_sandbox(args, fn):
        result = fn(fake_load(), "hermes-test")
        seen["result"] = result

    def fake_run(cfg, args, **kwargs):
        seen["openshell"].append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="Host hermes-test\n  HostName 127.0.0.1\n",
            stderr="",
        )

    def fake_subprocess_run(argv, **kwargs):
        seen["subprocess"].append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(openshell_cli, "_run_for_sandbox", fake_run_for_sandbox)
    monkeypatch.setattr(openshell_cli, "run_openshell_cli", fake_run)
    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    args = argparse.Namespace(exec_command=["--", "pwd"])
    openshell_cli._cmd_exec(args)

    assert seen["openshell"][0] == ["sandbox", "ssh-config", "hermes-test"]
    assert seen["subprocess"][0][:4] == ["ssh", "-F", seen["subprocess"][0][2], "-T"]
    assert seen["subprocess"][0][-2:] == ["hermes-test", "pwd"]
