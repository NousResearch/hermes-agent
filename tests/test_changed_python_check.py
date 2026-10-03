from pathlib import Path

from scripts.check_changed_python import blocking_diagnostics, changed_lines


def test_changed_lines_respect_insertions_and_deletions():
    assert changed_lines(
        "@@ -2,0 +3,2 @@\n+x\n+y\n@@ -8,1 +10,0 @@\n-z\n@@ -14 +16 @@\n-a\n+b"
    ) == {3, 4, 16}


def test_old_diagnostics_do_not_block_but_changed_spans_do(tmp_path):
    entries = [
        {
            "filename": str(tmp_path / "file.py"),
            "location": {"row": 1},
            "end_location": {"row": 1},
            "code": "F821",
            "message": "old",
        },
        {
            "filename": str(tmp_path / "file.py"),
            "location": {"row": 3},
            "end_location": {"row": 5},
            "code": "F821",
            "message": "new",
        },
    ]
    result = blocking_diagnostics(entries, "ruff", {"file.py": {4}}, tmp_path)
    assert len(result) == 1 and "new" in result[0]


def test_type_errors_block_changed_lines_but_warnings_remain_advisory():
    entry = {
        "location": {"path": "file.py", "lines": {"begin": 3}},
        "description": "bad return",
        "check_name": "invalid-return-type",
        "severity": "major",
    }
    assert blocking_diagnostics([entry], "ty", {"file.py": {3}}, Path.cwd())
    assert not blocking_diagnostics(
        [{**entry, "severity": "minor"}], "ty", {"file.py": {3}}, Path.cwd()
    )


def test_locationless_tool_errors_fail_closed():
    assert blocking_diagnostics(
        [{"message": "invalid configuration"}], "ruff", {}, Path.cwd()
    )


def test_gate_checks_new_files_with_real_tools(tmp_path, monkeypatch, capsys):
    import subprocess
    from scripts.check_changed_python import main

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test Fixture",
            "-c",
            "user.email=fixture@example.test",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["check_changed_python.py"])
    file = tmp_path / "example.py"
    file.write_text("print(undefined_variable)\n", encoding="utf-8")
    assert main() == 1
    assert "F821" in capsys.readouterr().out
    file.write_text("print('valid')\n", encoding="utf-8")
    assert main() == 0
