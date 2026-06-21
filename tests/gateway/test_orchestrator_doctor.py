import json
import sys

from gateway.orchestrator.command import CommandResult, FakeCommandRunner, SubprocessCommandRunner
from gateway.orchestrator.doctor import check_codex_sandbox, run_doctor
from gateway.orchestrator.registry import AgentKind, AgentSpec


def test_binary_detection_reports_codex_available_and_missing_binary():
    specs = [
        AgentSpec("codex", AgentKind.BINARY, ("codex", "--version"), sandbox=True),
        AgentSpec("missingbin", AgentKind.BINARY, ("missingbin", "--version")),
    ]
    runner = FakeCommandRunner({
        ("/bin/codex", "--version"): CommandResult(0, "codex-cli 0.141.0\n", ""),
        ("bwrap", "--unshare-user", "--uid", "0", "--gid", "0", "true"): CommandResult(0, "", ""),
        ("unshare", "-Ur", "true"): CommandResult(0, "", ""),
        ("unshare", "-n", "true"): CommandResult(0, "", ""),
    })

    report = run_doctor(
        specs=specs,
        which_fn=lambda name: f"/bin/{name}" if name in {"codex", "bwrap", "unshare"} else None,
        runner=runner,
    )
    by_name = {agent.name: agent for agent in report.agents}

    assert by_name["codex"].status == "available"
    assert by_name["codex"].path == "/bin/codex"
    assert by_name["codex"].version == "codex-cli 0.141.0"
    assert by_name["missingbin"].status == "missing"
    assert by_name["missingbin"].version is None


def test_shell_function_detection_uses_bash_type_without_executing_function():
    specs = [
        AgentSpec("ccg", AgentKind.SHELL_FUNCTION),
        AgentSpec("ccd", AgentKind.SHELL_FUNCTION),
    ]
    runner = FakeCommandRunner({
        ("bash", "-ic", "type -t ccg"): CommandResult(0, "function\n", ""),
        ("bash", "-ic", "type -t ccd"): CommandResult(1, "", ""),
    })
    which_calls: list[str] = []

    report = run_doctor(specs=specs, which_fn=lambda name: which_calls.append(name) or None, runner=runner)
    by_name = {agent.name: agent for agent in report.agents}

    assert which_calls == []
    assert by_name["ccg"].status == "available"
    assert by_name["ccg"].version is None
    assert by_name["ccd"].status == "missing"
    assert all(call[0] != "ccd" for call in runner.calls)
    assert ("bash", "-ic", "type -t ccg") in runner.calls


def test_ccm_secret_wrapper_suppresses_probe_output_in_report_json():
    specs = [AgentSpec("ccm", AgentKind.SHELL_FUNCTION, secrets=True)]
    runner = FakeCommandRunner({
        ("bash", "-ic", "type -t ccm"): CommandResult(0, "function\nMINIMAX_API_KEY=super-secret", "token=abc"),
    })

    report = run_doctor(specs=specs, which_fn=lambda _name: None, runner=runner)
    encoded = json.dumps(report.to_dict(), ensure_ascii=False)

    assert report.agents[0].status == "available"
    assert report.agents[0].version is None
    assert "super-secret" not in encoded
    assert "token=abc" not in encoded
    assert "MINIMAX_API_KEY" not in encoded


def test_codex_sandbox_health_detects_healthy_degraded_and_unavailable():
    healthy_runner = FakeCommandRunner({
        ("bwrap", "--unshare-user", "--uid", "0", "--gid", "0", "true"): CommandResult(0, "", ""),
        ("unshare", "-Ur", "true"): CommandResult(0, "", ""),
        ("unshare", "-n", "true"): CommandResult(0, "", ""),
    })
    healthy = check_codex_sandbox(which_fn=lambda name: f"/bin/{name}", runner=healthy_runner)
    assert healthy.status == "healthy"

    degraded_runner = FakeCommandRunner({
        ("bwrap", "--unshare-user", "--uid", "0", "--gid", "0", "true"): CommandResult(1, "", "bwrap: No permissions to create a new namespace"),
        ("unshare", "-Ur", "true"): CommandResult(0, "", ""),
        ("unshare", "-n", "true"): CommandResult(0, "", ""),
    })
    degraded = check_codex_sandbox(which_fn=lambda name: f"/bin/{name}", runner=degraded_runner)
    assert degraded.status == "degraded"
    assert "namespace" in degraded.detail.lower()
    assert not any("dangerously" in " ".join(call) for call in degraded_runner.calls)
    assert not any(call and call[0] == "codex" for call in degraded_runner.calls)

    unavailable = check_codex_sandbox(which_fn=lambda _name: None, runner=FakeCommandRunner({}))
    assert unavailable.status == "unavailable"


def test_subprocess_command_runner_uses_explicit_env(tmp_path):
    script = tmp_path / "print_env.py"
    script.write_text("import os; print(os.environ.get('PATH', ''))\n")

    runner = SubprocessCommandRunner(env={"PATH": "custom-path"})

    result = runner.run([sys.executable, str(script)], timeout=5)

    assert result.returncode == 0
    assert result.stdout.strip() == "custom-path"


def test_default_run_doctor_is_limited_to_four_operational_agents_and_secret_free():
    runner = FakeCommandRunner({
        ("/bin/codex", "--version"): CommandResult(0, "codex-cli 0.141.0\n", ""),
        ("bash", "-ic", "type -t ccd"): CommandResult(0, "function\n", ""),
        ("bash", "-ic", "type -t ccg"): CommandResult(0, "function\n", ""),
        ("bash", "-ic", "type -t ccm"): CommandResult(0, "function\n", "API_TOKEN=abc"),
        ("bwrap", "--unshare-user", "--uid", "0", "--gid", "0", "true"): CommandResult(1, "", "Operation not permitted"),
        ("unshare", "-Ur", "true"): CommandResult(0, "", ""),
        ("unshare", "-n", "true"): CommandResult(0, "", ""),
    })

    report = run_doctor(
        which_fn=lambda name: f"/bin/{name}" if name in {"codex", "bwrap", "unshare"} else None,
        runner=runner,
    )
    payload = report.to_dict()
    encoded = json.dumps(payload, ensure_ascii=False)

    assert [agent["name"] for agent in payload["agents"]] == ["ccd", "codex", "ccg", "ccm"]
    assert "claude" not in encoded
    assert "emd" not in encoded
    assert "abc" not in encoded
    assert "API_TOKEN" not in encoded
    assert payload["agents"][1]["sandbox"]["status"] == "degraded"


def test_default_which_discovers_user_local_bin_when_service_path_omits_it(tmp_path, monkeypatch):
    # Regression: hermes.service starts with a minimal PATH that omits
    # ~/.local/bin where `codex` is installed, so a raw shutil.which reports it
    # missing even though the bash -ic executor can launch it.
    import shutil

    from gateway.orchestrator.doctor import _default_which

    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    fake = local_bin / "codex"
    fake.write_text("#!/bin/sh\necho codex-cli 0.141.0\n")
    fake.chmod(0o755)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # simulate hermes.service PATH

    assert shutil.which("codex") is None  # the bug: bare PATH lookup misses it
    assert _default_which("codex") == str(fake)  # the fix: augmented lookup finds it


def test_default_run_doctor_discovers_user_local_codex_with_service_path(tmp_path, monkeypatch):
    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    codex = local_bin / "codex"
    codex.write_text("#!/bin/sh\necho codex-cli 0.141.0\n")
    codex.chmod(0o755)
    specs = [AgentSpec("codex", AgentKind.BINARY, ("codex", "--version"), sandbox=True)]
    runner = FakeCommandRunner({
        (str(codex), "--version"): CommandResult(0, "codex-cli 0.141.0\n", ""),
        ("unshare", "-Ur", "true"): CommandResult(1, "", "unshare: Operation not permitted"),
        ("unshare", "-n", "true"): CommandResult(0, "", ""),
    })

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # simulate hermes.service PATH

    report = run_doctor(specs=specs, runner=runner)
    agent = report.agents[0]

    assert agent.status == "degraded"
    assert agent.path == str(codex)
    assert agent.version == "codex-cli 0.141.0"
    assert "binary not found on PATH" not in agent.notes
    assert "sandbox degraded" in agent.notes


def test_codex_degraded_internal_sandbox_can_report_external_isolation_capability():
    spec = AgentSpec("codex", AgentKind.BINARY, ("codex", "--version"), sandbox=True, external_isolation=True)
    resolved = "/opt/agents/bin/codex"
    runner = FakeCommandRunner({
        (resolved, "--version"): CommandResult(0, "codex-cli 0.141.0\n", ""),
        ("unshare", "-Ur", "true"): CommandResult(1, "", "unshare: write failed /proc/self/uid_map: Operation not permitted"),
        ("unshare", "-n", "true"): CommandResult(0, "", ""),
        (resolved, "sandbox", "-P", ":danger-full-access", "true"): CommandResult(0, "", ""),
    })

    report = run_doctor(
        specs=[spec],
        which_fn=lambda name: resolved if name == "codex" else "/bin/unshare" if name == "unshare" else None,
        runner=runner,
    )
    agent = report.agents[0]
    payload = report.to_dict()["agents"][0]

    assert agent.status == "degraded"
    assert agent.execution_mode == "external-isolated"
    assert agent.external_isolation.status == "available"
    assert agent.external_isolation.mode == "danger-full-access"
    assert "sandbox degraded" in agent.notes
    assert "external isolation available" in agent.notes
    assert payload["execution_mode"] == "external-isolated"
    assert payload["external_isolation"]["status"] == "available"
    assert (resolved, "sandbox", "-P", ":danger-full-access", "true") in runner.calls


def test_binary_version_probe_uses_resolved_absolute_path():
    # Regression: the version probe must run the resolved absolute path, not the
    # bare name, so it does not depend on the (minimal) service PATH.
    spec = AgentSpec("codex", AgentKind.BINARY, ("codex", "--version"))
    resolved = "/opt/agents/bin/codex"
    runner = FakeCommandRunner({
        (resolved, "--version"): CommandResult(0, "codex-cli 0.141.0\n", ""),
    })

    report = run_doctor(
        specs=[spec],
        which_fn=lambda name: resolved if name == "codex" else None,
        runner=runner,
    )
    agent = report.agents[0]

    assert agent.status == "available"
    assert agent.version == "codex-cli 0.141.0"
    assert (resolved, "--version") in runner.calls
    assert ("codex", "--version") not in runner.calls  # never probe by bare name


def test_codex_version_probe_failure_stays_degraded_when_sandbox_degraded():
    spec = AgentSpec("codex", AgentKind.BINARY, ("codex", "--version"), sandbox=True)
    resolved = "/opt/agents/bin/codex"
    runner = FakeCommandRunner({
        (resolved, "--version"): CommandResult(127, "", "exec format error"),
        ("unshare", "-Ur", "true"): CommandResult(1, "", "unshare: Operation not permitted"),
        ("unshare", "-n", "true"): CommandResult(0, "", ""),
    })

    report = run_doctor(
        specs=[spec],
        which_fn=lambda name: resolved if name == "codex" else "/bin/unshare" if name == "unshare" else None,
        runner=runner,
    )
    agent = report.agents[0]

    assert agent.status == "degraded"
    assert agent.path == resolved
    assert agent.version is None
    assert any("version probe failed" in note and "exec format error" in note for note in agent.notes)
    assert "sandbox degraded" in agent.notes
    assert (resolved, "--version") in runner.calls
    assert ("codex", "--version") not in runner.calls
