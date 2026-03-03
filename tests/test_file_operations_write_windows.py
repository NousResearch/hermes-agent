from pathlib import Path

from tools.file_operations import ShellFileOperations


class _FakeLocalEnv:
    __module__ = "tools.environments.local"

    def __init__(self, cwd: str):
        self.cwd = cwd

    def execute(self, command: str, cwd: str = "", **kwargs):
        # Windows-local write path should not go through shell execution.
        raise AssertionError(f"execute() should not be called, got: {command}")


def test_write_file_windows_local_uses_native_io(tmp_path):
    env = _FakeLocalEnv(str(tmp_path))
    ops = ShellFileOperations(env)

    rel_path = r"agents\memory.md"
    content = "EOT check: write via native IO"
    result = ops.write_file(rel_path, content)

    assert result.error is None
    assert result.bytes_written > 0

    target = Path(ops._resolve_windows_path(rel_path))
    assert target.exists()
    assert target.read_text(encoding="utf-8") == content
