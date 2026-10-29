"""`hermes sandbox run`: the container lock cannot be loosened, and the export runs nothing the code carries."""

import argparse
import os
import subprocess

import pytest

from hermes_cli.sandbox_cmd import (
    NO_RUNTIME_EXIT, export_path, export_ref, sandbox_run_argv, sandbox_settings, sandbox_steps,
)
from hermes_cli.subcommands.sandbox import build_sandbox_parser

# Everything the terminal's docker backend would otherwise pass along.
_HOSTILE_CONFIG = {"terminal": {
    "docker_volumes": ["/:/host"], "docker_mount_cwd_to_workspace": True,
    "docker_forward_env": ["GITHUB_TOKEN"], "docker_env": {"LEAK": "1"},
    "docker_extra_args": ["--privileged", "--network", "host"],
    "env_passthrough": ["GITHUB_TOKEN"], "credential_files": ["auth.json"],
    "container_cpu": 2, "container_memory": 2048}}


def _parse(argv):
    parser = argparse.ArgumentParser()
    build_sandbox_parser(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


@pytest.mark.parametrize("setup_flags", [[], ["--setup-network", "none"], ["--setup-network", "open"]],
                         ids=["setup-default", "setup-none", "setup-open"])
@pytest.mark.parametrize("limits", [True, False])
@pytest.mark.parametrize("image", [None, "attacker/image:latest"])
@pytest.mark.parametrize("command", [
    ["python", "-m", "pytest"],
    ["sh", "-c", "cat ~/.ssh/id_rsa"],
    ["-v", "/:/host", "-e", "GITHUB_TOKEN", "--network", "host", "--privileged"],
])
def test_run_argv_is_locked_whatever_the_options(monkeypatch, setup_flags, limits, image, command):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_fake")
    args = _parse(["sandbox", "run", "--path", ".", "--setup", "pip install -e .", *setup_flags,
                   *(["--image", image] if image else []), "--", *command])
    chosen, cpus, memory_mb = sandbox_settings(_HOSTILE_CONFIG, args.image)
    steps = sandbox_steps(args.setup, args.setup_network, args.run_command[1:])
    assert [cmd for cmd, _ in steps] == [["sh", "-c", "pip install -e ."], command]

    for index, (step_command, network) in enumerate(steps):
        argv = sandbox_run_argv("docker", chosen, "/scratch/x", step_command, network=network,
                                limits=limits, cpus=cpus, memory_mb=memory_mb, name="hermes-sandbox-t")
        split = argv.index("--entrypoint")
        flags, tail = argv[:split], argv[split:]
        assert tail == ["--entrypoint", "env", chosen, "HOME=/sandbox/home", *step_command]
        pairs = list(zip(flags, flags[1:]))
        for pair in [("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges"),
                     ("--user", "65534:65534"), ("-v", "/scratch/x:/sandbox")]:
            assert pair in pairs
        assert {"--rm", "--read-only"} <= set(flags)
        assert flags.count("-v") == 1 and flags.count("--security-opt") == 1
        assert not set(flags) & {"-e", "--env", "--env-file", "--privileged", "--cap-add", "--mount",
                                 "--volume", "--volumes-from", "--device", "--pid", "--ipc", "--userns"}
        assert not any("GITHUB_TOKEN" in a or "LEAK" in a or "/host" in a for a in flags)
        # The run step never has a network; setup only when "open" was asked for explicitly.
        setup_open = index == 0 and setup_flags == ["--setup-network", "open"]
        assert [b for a, b in pairs if a == "--network"] == ([] if setup_open else ["none"])

    # No runtime: a distinct exit code, and nothing is run on the host instead.
    import tools.environments.docker as docker_env

    spawned = []
    monkeypatch.setattr(docker_env, "find_docker", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: spawned.append(a))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: spawned.append(a))
    assert args.func(args) == NO_RUNTIME_EXIT != 0 and spawned == []


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_export_runs_nothing_the_code_carries(tmp_path):
    marks, secret, repo = tmp_path / "marks", tmp_path / "secret.txt", tmp_path / "repo"
    marks.mkdir()
    secret.write_text("FAKE-SECRET")
    repo.mkdir()
    payload = f"open({str(marks / 'imported')!r}, 'w').close()\n"
    (repo / "conftest.py").write_text(payload)
    (repo / ".gitattributes").write_text("* filter=evil\n")
    (repo / "leak").symlink_to(secret)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "commit", "-qm", "x")
    # What a hostile checkout could carry locally: a filter driver git archive would run, an
    # fsmonitor, and hooks a checkout or fetch would run.
    _git(repo, "config", "filter.evil.smudge", f"sh -c 'touch {marks}/smudge; cat'")
    _git(repo, "config", "core.fsmonitor", f"sh -c 'touch {marks}/fsmonitor'")
    for hook in ("post-checkout", "reference-transaction", "pre-auto-gc"):
        script = repo / ".git" / "hooks" / hook
        script.write_text(f"#!/bin/sh\ntouch {marks}/{hook}\n")
        script.chmod(0o755)

    export_ref(repo, "HEAD", tmp_path / "from-ref")
    export_path(repo, tmp_path / "from-path")

    assert sorted(os.listdir(marks)) == []
    for out in (tmp_path / "from-ref", tmp_path / "from-path"):
        assert (out / "conftest.py").read_text() == payload
        assert (out / "leak").is_symlink() and os.readlink(out / "leak") == str(secret)
    assert not (tmp_path / "from-path" / ".git").exists()
