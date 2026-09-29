import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

def test_run_task_kimi_omits_temperature():
    """Kimi models should NOT have client-side temperature overrides.

    The Kimi gateway selects the correct temperature server-side.
    """
    with patch("openai.OpenAI") as mock_openai:
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=[]))]
        )
        mock_openai.return_value = client

        from mini_swe_runner import MiniSWERunner

        runner = MiniSWERunner(
            model="kimi-for-coding",
            base_url="https://api.kimi.com/coding/v1",
            api_key="test-key",
            env_type="local",
            max_iterations=1,
        )
        runner._create_env = MagicMock()
        runner._cleanup_env = MagicMock()

        result = runner.run_task("2+2")

    assert result["completed"] is True
    assert "temperature" not in client.chat.completions.create.call_args.kwargs


REPO_ROOT = Path(__file__).resolve().parents[1]


def _runner_env():
    """Run the entry point with no provider credentials in reach."""
    env = dict(os.environ)
    for key in list(env):
        if any(marker in key.upper() for marker in ("OPENAI", "ANTHROPIC", "OPENROUTER", "GEMINI", "GOOGLE")):
            env.pop(key)
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def _run_script(args, tmp_path):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "mini_swe_runner.py"), *args],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=180, env=_runner_env(),
    )


def _run_entry_point_with_stub(tmp_path, stub, args):
    """Drive main() through fire with one method stubbed, so the exit code is the real one."""
    driver = tmp_path / "driver.py"
    driver.write_text("import mini_swe_runner as m\n" + stub + "\nimport fire\nfire.Fire(m.main)\n")
    return subprocess.run([sys.executable, str(driver), *args], cwd=str(tmp_path),
                          capture_output=True, text=True, timeout=180, env=_runner_env())


def test_entry_point_without_arguments_exits_two(tmp_path):
    proc = _run_script([], tmp_path)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "Please provide either --task or --prompts_file" in proc.stderr
    # Rejected before a runner (and its client) is built.
    assert "Mini-SWE Runner initialized" not in proc.stdout


def test_entry_point_with_empty_prompts_file_exits_one(tmp_path):
    prompts = tmp_path / "empty.jsonl"
    prompts.write_text("")
    proc = _run_script(["--prompts_file", str(prompts)], tmp_path)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "No prompts found" in proc.stderr
    assert "Mini-SWE Runner initialized" not in proc.stdout


def test_entry_point_reports_a_task_that_hit_the_iteration_ceiling(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    stub = ("m.MiniSWERunner.run_task = lambda self, task: "
            "{'completed': False, 'api_calls': 15, 'conversations': []}")
    proc = _run_entry_point_with_stub(
        tmp_path, stub,
        ["--task", "2+2", "--output_file", str(trajectory), "--api_key", "test-key"])
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert trajectory.exists()


def test_entry_point_reports_a_completed_task_as_success(tmp_path):
    trajectory = tmp_path / "trajectory.jsonl"
    stub = ("m.MiniSWERunner.run_task = lambda self, task: "
            "{'completed': True, 'api_calls': 1, 'conversations': []}")
    proc = _run_entry_point_with_stub(
        tmp_path, stub,
        ["--task", "2+2", "--output_file", str(trajectory), "--api_key", "test-key"])
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_entry_point_reports_a_finished_batch_as_success(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "2+2"}\n')
    stub = "m.MiniSWERunner.run_batch = lambda self, prompts, output_file: []"
    proc = _run_entry_point_with_stub(
        tmp_path, stub,
        ["--prompts_file", str(prompts), "--output_file", str(tmp_path / "out.jsonl"),
         "--api_key", "test-key"])
    assert proc.returncode == 0, proc.stdout + proc.stderr
