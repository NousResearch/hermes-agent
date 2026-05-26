"""Static checks for scripts/run_tests.sh invariants."""
from __future__ import annotations

from pathlib import Path


RUN_TESTS = Path(__file__).resolve().parents[2] / "scripts" / "run_tests.sh"


def _script() -> str:
    return RUN_TESTS.read_text(encoding="utf-8")


def test_run_tests_raises_file_descriptor_limit_for_full_suite():
    script = _script()
    assert "ulimit -n 4096" in script
    assert "Too many open files" in script


def test_run_tests_keeps_hermetic_credential_env_unset():
    script = _script()
    assert "*_API_KEY" in script
    assert "*_TOKEN" in script
    assert "unset \"$name\"" in script


def test_run_tests_no_arg_invocation_is_safe_on_bash_32():
    script = _script()
    executable_lines = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )
    assert "if [ \"$#\" -gt 0 ]; then" in script
    assert 'PYTEST_CMD+=("$@")' in script
    assert "ARGS=(" not in executable_lines
    assert '"${ARGS[@]}"' not in executable_lines
