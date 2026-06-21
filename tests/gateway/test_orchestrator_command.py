import sys

from gateway.orchestrator.command import CommandResult, FakeCommandRunner, SubprocessCommandRunner


def test_fake_command_runner_returns_programmed_result_and_records_calls():
    runner = FakeCommandRunner({("tool", "--version"): CommandResult(0, "v1\n", "")})

    result = runner.run(["tool", "--version"], timeout=3)

    assert result.returncode == 0
    assert result.stdout == "v1\n"
    assert runner.calls == [("tool", "--version")]


def test_fake_command_runner_returns_missing_command_result_by_default():
    runner = FakeCommandRunner({})

    result = runner.run(["missing"], timeout=3)

    assert result.returncode == 127
    assert "not programmed" in result.stderr


def test_subprocess_command_runner_captures_successful_output():
    runner = SubprocessCommandRunner()

    result = runner.run([sys.executable, "-c", "print('hi')"], timeout=5)

    assert result.returncode == 0
    assert result.stdout.strip() == "hi"
    assert result.timed_out is False


def test_subprocess_command_runner_maps_timeout_to_result_without_raising():
    runner = SubprocessCommandRunner()

    result = runner.run([sys.executable, "-c", "import time; time.sleep(2)"], timeout=0.1)

    assert result.timed_out is True
    assert result.returncode == -1
