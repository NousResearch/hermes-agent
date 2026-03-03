from pathlib import Path

from tools.file_operations import ShellFileOperations


class _FakeLocalEnv:
    __module__ = "tools.environments.local"

    def __init__(self, cwd: str):
        self.cwd = cwd

    def execute(self, command: str, cwd: str = "", **kwargs):
        # Search path for Windows local should use native walk implementation.
        return {"output": "", "returncode": 0}


def test_search_content_with_glob_pattern_routes_to_file_search(tmp_path):
    (tmp_path / "a.html").write_text("<h1>A</h1>", encoding="utf-8")
    (tmp_path / "b.txt").write_text("hello", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.html").write_text("<p>C</p>", encoding="utf-8")

    env = _FakeLocalEnv(str(tmp_path))
    ops = ShellFileOperations(env)

    result = ops.search(pattern="*.html", path=str(tmp_path), target="content")
    payload = result.to_dict()

    assert "error" not in payload
    files = payload.get("files", [])
    assert len(files) == 2
    assert any(f.endswith("a.html") for f in files)
    assert any(f.endswith("c.html") for f in files)
