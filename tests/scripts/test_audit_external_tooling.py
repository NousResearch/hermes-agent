from __future__ import annotations

from pathlib import Path
import json
import hashlib
import os
import subprocess
import sys
from types import SimpleNamespace

from scripts.audit_external_tooling import (
    _summarize_json_output,
    _distribution_realization_digest,
    build_commands,
    build_uv_export_command,
    run_audits,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    tool_root = tmp_path / "venv" / "bin"
    tool_root.mkdir(parents=True)
    site_packages = tool_root.parent / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    versions = {
        "zizmor": "1.30.0",
        "lint-imports": "2.14",
        "pip-audit": "2.10.1",
    }
    pins = {}
    for name, version in versions.items():
        executable = tool_root / name
        executable.write_text(f"#!/bin/sh\n# {name}\n", encoding="utf-8")
        executable.chmod(0o755)
        pins["import-linter" if name == "lint-imports" else name] = {
            "version": version,
            "sha256": f"sha256:{hashlib.sha256(executable.read_bytes()).hexdigest()}",
        }
        if name in {"lint-imports", "pip-audit"}:
            distribution_name = "import-linter" if name == "lint-imports" else "pip-audit"
            distribution_dir = site_packages / (
                "import_linter-2.14.dist-info"
                if name == "lint-imports"
                else "pip_audit-2.10.1.dist-info"
            )
            distribution_dir.mkdir()
            (distribution_dir / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: {version}\n",
                encoding="utf-8",
            )
            package_name = "import_linter.py" if name == "lint-imports" else "pip_audit.py"
            (site_packages / package_name).write_text(
                f"# {distribution_name}\n", encoding="utf-8"
            )
            (distribution_dir / "RECORD").write_text(
                f"{package_name},,\n"
                f"{distribution_dir.name}/METADATA,,\n"
                f"{distribution_dir.name}/RECORD,,\n"
                f"../../../bin/{name},,\n",
                encoding="utf-8",
            )
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\n# uv\n", encoding="utf-8")
    uv.chmod(0o755)
    lock = {"schema_version": 1, "tools": pins, "uv": {
        "version": "0.12.6",
        "sha256": f"sha256:{hashlib.sha256(uv.read_bytes()).hexdigest()}",
    }}
    for distribution_name in ("import-linter", "pip-audit"):
        digest = _distribution_realization_digest(tool_root, distribution_name)
        assert digest is not None
        lock["tools"][distribution_name]["distribution_sha256"] = digest
    (repo / ".audit-tool-lock.json").write_text(json.dumps(lock), encoding="utf-8")
    return repo, tool_root, uv


def test_zizmor_summary_preserves_counts_when_raw_capture_is_bounded():
    output = json.dumps(
        [
            {
                "ident": "github-app",
                "determinations": {"severity": "High", "confidence": "High"},
                "locations": [{}, {}],
            },
            {
                "ident": "github-app",
                "determinations": {"severity": "High", "confidence": "Medium"},
                "locations": [{}],
            },
        ]
    )

    assert _summarize_json_output("zizmor", output) == {
        "finding_groups": 2,
        "locations": 3,
        "by_severity": {"High": 2},
        "by_confidence": {"High": 1, "Medium": 1},
        "by_ident": {"github-app": 2},
        "high_confidence_high_severity": 1,
    }


def test_commands_are_pinned_read_only_and_offline_where_supported(tmp_path):
    tool_root = tmp_path / "bin"
    commands = build_commands(repo_root=tmp_path / "repo", tool_root=tool_root)

    assert [command.name for command in commands] == [
        "zizmor",
        "import-linter",
        "pip-audit",
    ]
    flattened = [argument for command in commands for argument in command.argv]
    assert "--fix" not in flattened
    assert "--offline" in commands[0].argv
    assert "--no-exit-codes" in commands[0].argv
    assert "--locked" not in commands[2].argv
    assert "-r" in commands[2].argv
    assert "--require-hashes" in commands[2].argv
    assert "--disable-pip" in commands[2].argv
    assert "--cache-dir" in commands[2].argv

    export_command = build_uv_export_command(
        requirements_path=tmp_path / "requirements.txt"
    )
    assert "--locked" in export_command
    assert "--cache-dir" in export_command


def test_receipt_binds_results_to_exact_git_identity(tmp_path):
    repo, tool_root, uv = _fixture(tmp_path)
    head = "a" * 40

    def fake_runner(argv, **kwargs):
        command = tuple(str(part) for part in argv)
        if command[:3] == ("git", "rev-parse", "HEAD"):
            return SimpleNamespace(returncode=0, stdout=f"{head}\n", stderr="")
        if command[:3] == ("git", "branch", "--show-current"):
            return SimpleNamespace(returncode=0, stdout="feature/audit\n", stderr="")
        if command[:3] == ("git", "status", "--porcelain"):
            return SimpleNamespace(returncode=0, stdout=" M local.txt\n", stderr="")
        if command[-1:] == ("--version",):
            executable = Path(command[0]).name
            versions = {
                "zizmor": "zizmor 1.30.0\n",
                "lint-imports": "import-linter 2.14\n",
                "pip-audit": "pip-audit 2.10.1\n",
                "uv": "uv 0.12.6\n",
            }
            return SimpleNamespace(returncode=0, stdout=versions[executable], stderr="")
        if Path(command[0]).name == "zizmor":
            return SimpleNamespace(returncode=0, stdout="[]\n", stderr="")
        if Path(command[0]).name == "pip-audit":
            return SimpleNamespace(
                returncode=0, stdout='{"dependencies": []}\n', stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    receipt = run_audits(repo_root=repo, tool_root=tool_root, runner=fake_runner, uv_path=uv)

    assert receipt["repository"]["head"] == head
    assert receipt["repository"]["branch"] == "feature/audit"
    assert receipt["repository"]["dirty"] is True
    assert receipt["ok"] is False
    assert len(receipt["audits"]) == 3
    assert all(row["stdout_sha256"].startswith("sha256:") for row in receipt["audits"])


def test_receipt_fails_when_observed_tool_version_differs(tmp_path):
    repo, tool_root, uv = _fixture(tmp_path)

    calls = []

    def fake_runner(argv, **kwargs):
        command = tuple(str(part) for part in argv)
        calls.append(command)
        if command[:2] == ("git", "rev-parse"):
            return SimpleNamespace(returncode=0, stdout=f"{'b' * 40}\n", stderr="")
        if command[:2] == ("git", "branch"):
            return SimpleNamespace(returncode=0, stdout="feature/audit\n", stderr="")
        if command[:2] == ("git", "status"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[-1:] == ("--version",):
            executable = Path(command[0]).name
            versions = {
                "zizmor": "zizmor 9.9.9\n",
                "lint-imports": "import-linter 2.14\n",
                "pip-audit": "pip-audit 2.10.1\n",
                "uv": "uv 0.12.6\n",
            }
            return SimpleNamespace(returncode=0, stdout=versions[executable], stderr="")
        if Path(command[0]).name == "zizmor":
            return SimpleNamespace(returncode=0, stdout="[]\n", stderr="")
        if Path(command[0]).name == "pip-audit":
            return SimpleNamespace(
                returncode=0, stdout='{"dependencies": []}\n', stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    receipt = run_audits(repo_root=repo, tool_root=tool_root, runner=fake_runner, uv_path=uv)

    zizmor = receipt["audits"][0]
    assert zizmor["observed_version"] == "9.9.9"
    assert zizmor["version_matches"] is False
    assert zizmor["audit_invoked"] is False
    assert not any(call and Path(call[0]).name == "zizmor" and "--offline" in call for call in calls)
    assert receipt["ok"] is False


def test_python_distribution_realization_changes_when_installed_file_changes(tmp_path):
    _repo, tool_root, _uv = _fixture(tmp_path)
    before = _distribution_realization_digest(tool_root, "import-linter")

    assert before is not None
    package_file = (
        tool_root.parent
        / "lib"
        / "python3.13"
        / "site-packages"
        / "import_linter.py"
    )
    package_file.write_text("# replaced installed package\n", encoding="utf-8")

    after = _distribution_realization_digest(tool_root, "import-linter")

    assert after is not None
    assert after != before


def test_receipt_refuses_changed_python_distribution_before_audit(tmp_path):
    repo, tool_root, uv = _fixture(tmp_path)
    package_file = (
        tool_root.parent
        / "lib"
        / "python3.13"
        / "site-packages"
        / "pip_audit.py"
    )
    package_file.write_text("# replaced installed package\n", encoding="utf-8")
    calls = []

    def fake_runner(argv, **kwargs):
        command = tuple(str(part) for part in argv)
        calls.append(command)
        if command[:2] == ("git", "rev-parse"):
            return SimpleNamespace(returncode=0, stdout=f"{'d' * 40}\n", stderr="")
        if command[:2] == ("git", "branch"):
            return SimpleNamespace(returncode=0, stdout="feature/audit\n", stderr="")
        if command[:2] == ("git", "status"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[-1:] == ("--version",):
            executable = Path(command[0]).name
            versions = {
                "zizmor": "zizmor 1.30.0\n",
                "lint-imports": "import-linter 2.14\n",
                "pip-audit": "pip-audit 2.10.1\n",
                "uv": "uv 0.12.6\n",
            }
            return SimpleNamespace(returncode=0, stdout=versions[executable], stderr="")
        if Path(command[0]).name == "zizmor":
            return SimpleNamespace(returncode=0, stdout="[]\n", stderr="")
        if Path(command[0]).name == "pip-audit":
            return SimpleNamespace(
                returncode=0, stdout='{"dependencies": []}\n', stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    receipt = run_audits(repo_root=repo, tool_root=tool_root, runner=fake_runner, uv_path=uv)

    pip_audit = receipt["audits"][2]
    assert pip_audit["distribution_realization_ok"] is False
    assert pip_audit["audit_invoked"] is False
    assert not any(
        call and Path(call[0]).name == "pip-audit" and "-r" in call for call in calls
    )
    assert receipt["ok"] is False


def test_malformed_scanner_payloads_fail_closed():
    assert _summarize_json_output("pip-audit", "{}") is None
    assert _summarize_json_output(
        "pip-audit", '{"dependencies": [{"name": "demo", "version": "1.0"}]}'
    ) is None
    assert _summarize_json_output("zizmor", "[{}]") is None


def test_zizmor_high_confidence_high_severity_finding_fails_policy(tmp_path):
    repo, tool_root, uv = _fixture(tmp_path)
    finding = json.dumps(
        [
            {
                "ident": "template-injection",
                "determinations": {"severity": "High", "confidence": "High"},
                "locations": [{}],
            }
        ]
    )

    def fake_runner(argv, **kwargs):
        command = tuple(str(part) for part in argv)
        if command[:2] == ("git", "rev-parse"):
            return SimpleNamespace(returncode=0, stdout=f"{'c' * 40}\n", stderr="")
        if command[:2] == ("git", "branch"):
            return SimpleNamespace(returncode=0, stdout="feature/audit\n", stderr="")
        if command[:2] == ("git", "status"):
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[-1:] == ("--version",):
            executable = Path(command[0]).name
            versions = {
                "zizmor": "zizmor 1.30.0\n",
                "lint-imports": "import-linter 2.14\n",
                "pip-audit": "pip-audit 2.10.1\n",
                "uv": "uv 0.12.6\n",
            }
            return SimpleNamespace(returncode=0, stdout=versions[executable], stderr="")
        if Path(command[0]).name == "zizmor":
            return SimpleNamespace(returncode=0, stdout=finding, stderr="")
        if Path(command[0]).name == "pip-audit":
            return SimpleNamespace(
                returncode=0, stdout='{"dependencies": []}\n', stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    receipt = run_audits(repo_root=repo, tool_root=tool_root, runner=fake_runner, uv_path=uv)

    zizmor = receipt["audits"][0]
    assert zizmor["returncode"] == 0
    assert zizmor["summary"]["high_confidence_high_severity"] == 1
    assert zizmor["policy_ok"] is False
    assert receipt["ok"] is False


def test_app_token_requires_complete_credential_pair():
    action = (REPO_ROOT / ".github" / "actions" / "get-app-token" / "action.yml").read_text()

    assert "PRIVATE_KEY: ${{ inputs.private-key }}" in action
    assert 'if [ -n "$CLIENT_ID" ] && [ -n "$PRIVATE_KEY" ]; then' in action


def test_script_runs_directly_from_its_file_path():
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "audit_external_tooling.py"), "--plan"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["read_only"] is True
