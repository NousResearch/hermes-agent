from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dflash_stability_canary.py"
SPEC = importlib.util.spec_from_file_location("dflash_stability_canary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
canary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = canary
SPEC.loader.exec_module(canary)

LOOP_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dflash_hardening_loop.py"
LOOP_SPEC = importlib.util.spec_from_file_location("dflash_hardening_loop", LOOP_MODULE_PATH)
assert LOOP_SPEC is not None and LOOP_SPEC.loader is not None
loop = importlib.util.module_from_spec(LOOP_SPEC)
sys.modules[LOOP_SPEC.name] = loop
LOOP_SPEC.loader.exec_module(loop)


def test_incomplete_tail_classifier_catches_short_connector_fragment():
    assert canary.looks_like_incomplete_tail("I see a lot of discord-res tasks (digest Discord content) and some")
    assert canary.looks_like_incomplete_tail("Now let me read the STATUS.md to find a task I can pick up")
    assert not canary.looks_like_incomplete_tail("CANARY_ONBOARD_OK")


def test_classify_result_requires_exact_marker():
    ok = canary.CommandResult(returncode=0, stdout="CANARY_ONBOARD_OK\n", stderr="", elapsed_s=1.0)
    noisy = canary.CommandResult(returncode=0, stdout="Done. CANARY_ONBOARD_OK\n", stderr="", elapsed_s=1.0)

    assert canary.classify_result(ok, "CANARY_ONBOARD_OK") is None
    assert canary.classify_result(noisy, "CANARY_ONBOARD_OK") == "marker-mismatch"
    assert canary.classify_result(noisy, "CANARY_ONBOARD_OK", strict_marker=False) is None


def test_classify_result_flags_auth_empty_timeout_and_nonzero():
    assert (
        canary.classify_result(
            canary.CommandResult(returncode=0, stdout="", stderr="", elapsed_s=1.0),
            "MARKER",
        )
        == "empty-final"
    )
    assert (
        canary.classify_result(
            canary.CommandResult(returncode=0, stdout="MARKER", stderr="Error code: 401 Invalid API key", elapsed_s=1.0),
            "MARKER",
        )
        == "auth-error"
    )
    assert (
        canary.classify_result(
            canary.CommandResult(returncode=124, stdout="", stderr="", elapsed_s=180.0, timed_out=True),
            "MARKER",
        )
        == "timeout"
    )
    assert (
        canary.classify_result(
            canary.CommandResult(returncode=2, stdout="MARKER", stderr="", elapsed_s=1.0),
            "MARKER",
        )
        == "nonzero-exit"
    )


def test_build_command_includes_model_provider_toolsets():
    case = canary.CanaryCase(name="unit", marker="OK", prompt="reply OK")
    cmd = canary.build_command("hermes", case, model="dflash", provider="taro", toolsets="terminal,file")

    assert cmd == [
        "hermes",
        "--provider",
        "taro",
        "--model",
        "dflash",
        "--toolsets",
        "terminal,file",
        "-z",
        "reply OK",
    ]


def test_materialize_case_marker_replaces_marker_with_nonce():
    case = canary.CanaryCase(name="unit", marker="OK", prompt="reply exactly OK")

    materialized = canary.materialize_case_marker(case, "run-1 case")

    assert materialized.marker == "OK_run_1_case"
    assert materialized.prompt == "reply exactly OK_run_1_case"


def test_materialize_case_marker_injects_source_root(tmp_path):
    case = canary.CanaryCase(name="unit", marker="OK", prompt="cd {source_root}; reply OK")

    materialized = canary.materialize_case_marker(case, "n1", source_root=tmp_path)

    assert str(tmp_path) in materialized.prompt
    assert materialized.prompt.endswith("reply OK_n1")


def test_run_case_uses_runner_and_sanitizes_prompt_from_logged_command(tmp_path):
    case = canary.CanaryCase(name="unit", marker="OK", prompt="private prompt")

    def runner(cmd, cwd, timeout_s):
        assert cmd[-1] == "private prompt"
        assert cwd == tmp_path
        assert timeout_s == 5.0
        return canary.CommandResult(returncode=0, stdout="OK\n", stderr="", elapsed_s=0.25)

    record = canary.run_case(
        case,
        cwd=tmp_path,
        hermes_bin="hermes",
        model="dflash",
        provider="taro",
        toolsets="terminal,file",
        timeout_s=5.0,
        strict_marker=True,
        runner=runner,
    )

    assert record["ok"] is True
    assert record["failure"] is None
    assert record["cmd"][-1] == "<prompt>"
    assert record["stdout"] == "OK\n"


def test_run_case_uses_nonce_marker(tmp_path):
    case = canary.CanaryCase(name="unit", marker="OK", prompt="reply OK")

    def runner(cmd, cwd, timeout_s):
        assert cmd[-1] == "reply OK_cycle_1"
        return canary.CommandResult(returncode=0, stdout="OK_cycle_1\n", stderr="", elapsed_s=0.25)

    record = canary.run_case(
        case,
        cwd=tmp_path,
        hermes_bin="hermes",
        model="dflash",
        provider="taro",
        toolsets="terminal,file",
        timeout_s=5.0,
        strict_marker=True,
        marker_nonce="cycle-1",
        runner=runner,
    )

    assert record["ok"] is True
    assert record["marker"] == "OK_cycle_1"
    assert record["marker_base"] == "OK"


def test_run_subprocess_timeout_requests_thread_dump(tmp_path):
    code = (
        "import faulthandler, signal, sys, time; "
        "faulthandler.enable(file=sys.stderr, all_threads=True); "
        "sig=getattr(signal, 'SIGUSR1', None); "
        "sig is not None and faulthandler.register(sig, file=sys.stderr, all_threads=True, chain=False); "
        "time.sleep(30)"
    )

    result = canary.run_subprocess(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        timeout_s=0.2,
    )

    assert result.timed_out is True
    assert result.returncode == 124
    assert "Current thread" in result.stderr


def test_hardening_loop_failure_task_id_is_stable_and_sanitized():
    record = {"case": "MeshBoard Onboard", "failure": "nonzero_exit"}

    assert (
        loop.failure_task_id(record, prefix="Hermes DFlash Hardening Loop")
        == "hermes-dflash-hardening-loop-meshboard-onboard-nonzero-exit"
    )


def test_hardening_loop_meshboard_command_omits_raw_stdout(tmp_path):
    record = {
        "case": "meshboard-onboard",
        "failure": "auth-error",
        "returncode": 1,
        "elapsed_s": 12.3,
        "stdout": "secret-ish raw model text",
        "stderr": "raw stderr",
    }

    cmd = loop.build_meshboard_failure_command(
        meshboard_root=tmp_path,
        task_id="hermes-dflash-hardening-loop-meshboard-onboard-auth-error",
        record=record,
        log_path=tmp_path / "evidence.jsonl",
        actor="ko-taro.hermes",
        parent_task="parent-task",
    )
    command_text = "\n".join(cmd)

    assert "secret-ish raw model text" not in command_text
    assert "raw stderr" not in command_text
    assert "--parent-task" in cmd
    assert "parent-task" in cmd
    assert str(tmp_path / "evidence.jsonl") in cmd
