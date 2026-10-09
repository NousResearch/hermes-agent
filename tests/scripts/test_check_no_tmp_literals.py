"""scripts/check_no_tmp_literals.py: literal /tmp paths are flagged; idioms, comments, markers and tests are not."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_tmp_literals.py"

def _load():
    spec = importlib.util.spec_from_file_location("check_no_tmp_literals", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _hits(text: str, suffix: str = ".py") -> list[int]:
    return [lineno for lineno, _ in _load()._iter_lines_with_hits(text, suffix)]

@pytest.mark.parametrize(
    "line",
    [
        'STORAGE_DIR = "/tmp/hermes-results"',
        'return "/tmp"',
        "LOG=/tmp/pinggy.log",
        'cwd="/tmp"',
        "Save the file to `/tmp/report.pdf` and return it.",
        'workdir="/tmp/issue-78"',
        "--output=/tmp/x.json",
        "Auto-allow workspace and /tmp edits",
    ],
)
def test_literal_tmp_paths_are_flagged(line):
    assert _hits(line, ".md") == [1]

@pytest.mark.parametrize(
    "line",
    [
        'SOCKET_DIR="${TMPDIR:-/tmp}/hermes"',  # shell fallback idiom
        'Path("/var/tmp")',
        'Path("/private/tmp")',
        "mounted as tmpfs",
        "tmp_path / 'x'",
        "~/tmp/scratch",
        "cd ./tmp && ls",
        "C:\\\\Users\\\\x\\\\tmp",
        "a/tmp/b",
        "the tmp dir",
    ],
)
def test_non_tmp_tokens_are_not_flagged(line):
    assert _hits(line, ".md") == []

def test_python_comments_and_docstrings_are_exempt_but_strings_are_not():
    src = (
        '"""Module docstring mentions /tmp on purpose.\n'
        "\n"
        "More /tmp here too.\n"
        '"""\n'
        "# a comment about /tmp\n"
        'PROMPT = "write scratch files to /tmp/out"  # trailing comment: /tmp\n'
        "def f():\n"
        '    """Function docstring: /tmp"""\n'
        '    return "/tmp"\n'
        'HASH_IN_STRING = "not a comment # /tmp"\n'
    )
    assert _hits(src, ".py") == [6, 9, 10]

def test_js_and_shell_comments_are_exempt():
    assert _hits("// world-shared /tmp dir\nconst p = '/tmp/x'\n", ".ts") == [2]
    assert _hits("const p = 1 // see /tmp\n", ".ts") == []
    assert _hits("#!/bin/sh\n# stage under /tmp\nLOG=/tmp/x.log\n", ".sh") == [3]

def test_markdown_prose_is_not_exempt():
    assert _hits("Frames are written to `/tmp` during capture.\n", ".md") == [1]

def test_inline_marker_on_same_or_previous_line_allows_one_hit():
    mod = _load()
    src = (
        f'Path("/tmp").resolve()  # {mod.MARKER} — macOS alias check\n'
        f"<!-- {mod.MARKER} — explains the anti-pattern -->\n"
        "Never write to /tmp on Termux.\n"
        'STILL = "/tmp/bad"\n'
    )
    assert _hits(src, ".md") == [4]

def test_scan_skips_tests_lockfiles_ci_and_translations(tmp_path):
    mod = _load()
    files = {
        "tools/a.py": 'X = "/tmp/a"\n',
        "skills/x/SKILL.md": "save to /tmp/out\n",
        "website/docs/guide.md": "cd /tmp\n",
        "website/i18n/zh/guide.md": "cd /tmp\n",
        "tests/test_a.py": 'X = "/tmp/a"\n',
        "tools/test_inline.py": 'X = "/tmp/a"\n',
        "tools/conftest.py": 'X = "/tmp/a"\n',
        "app/src/__tests__/a.ts": "const p = '/tmp'\n",
        "app/src/a.test.ts": "const p = '/tmp'\n",
        "app/src/a.ts": "const p = '/tmp'\n",
        ".github/workflows/ci.yml": "run: echo > /tmp/x\n",
        "Dockerfile": "RUN tar -C /tmp\n",
        "package-lock.json": '"resolved": "/tmp"\n',
        "node_modules/x/index.js": "'/tmp'\n",
        "evals/e.py": '"/tmp"\n',
        "vendor.rs": 'let p = "/tmp";\n',
    }
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    hits = mod.scan(root=tmp_path)
    assert sorted(hits) == ["app/src/a.ts", "skills/x/SKILL.md", "tools/a.py", "website/docs/guide.md"]
    assert all(len(v) == 1 for v in hits.values())

def test_baseline_entries_are_burned_down_not_grown(tmp_path, monkeypatch, capsys):
    mod = _load()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "a.py").write_text('X = "/tmp/a"\nY = "/tmp/b"\n', encoding="utf-8")
    (tmp_path / "tools" / "clean.py").write_text("X = 1\n", encoding="utf-8")
    monkeypatch.setattr(mod, "ROOT", tmp_path)

    monkeypatch.setattr(mod, "_BASELINE", {"tools/a.py": 2})
    assert mod.main([]) == 0

    monkeypatch.setattr(mod, "_BASELINE", {"tools/a.py": 1})  # grew: the new line is reported
    assert mod.main([]) == 1
    assert "tools/a.py:1" in capsys.readouterr().out

    monkeypatch.setattr(mod, "_BASELINE", {"tools/a.py": 3})  # shrank: advisory only, strict fails
    assert mod.main([]) == 0
    assert "stale" in capsys.readouterr().out
    assert mod.main(["--strict-baseline"]) == 1

    monkeypatch.setattr(mod, "_BASELINE", {"tools/a.py": 2, "tools/clean.py": 1})  # stale entry
    assert mod.main([]) == 0
    assert "tools/clean.py: listed in _BASELINE but clean" in capsys.readouterr().out
    assert mod.main(["--strict-baseline"]) == 1

    monkeypatch.setattr(mod, "_BASELINE", {"tools/a.py": 2})
    assert mod.main(["--all"]) == 1  # burn-down view ignores the baseline
