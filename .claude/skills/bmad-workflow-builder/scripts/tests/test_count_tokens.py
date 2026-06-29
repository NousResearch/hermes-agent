#!/usr/bin/env python3
"""Tests for count_tokens.py.

Covers the output schema, the tiktoken path and the forced-fallback path
agreeing within tolerance, the CLI over a file and over stdin, and argument
guards. Run with: python3 -m pytest test_count_tokens.py
(or plain `python3 test_count_tokens.py` to run a lightweight self-check).
"""
import builtins
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "count_tokens.py"

SAMPLE = (
    "The builder is platform-agnostic. Nothing assumes a single runtime, and no "
    "model list is ever hardcoded. Token counts replace line counts as the one "
    "length metric, with a chars-over-four fallback when tiktoken is absent.\n"
) * 8


def _load_module():
    spec = importlib.util.spec_from_file_location("count_tokens", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tiktoken_path():
    mod = _load_module()
    try:
        import tiktoken  # noqa: F401
    except Exception:
        # No tiktoken in this interpreter; the real path can't be exercised here.
        tokens, method = mod.count_tokens(SAMPLE)
        assert method == "fallback"
        assert tokens == len(SAMPLE) // 4
        return
    tokens, method = mod.count_tokens(SAMPLE)
    assert method == "tiktoken"
    assert isinstance(tokens, int)
    assert tokens > 0


def test_fallback_path_when_import_blocked():
    """Force the import of tiktoken to fail and confirm the fallback fires."""
    mod = _load_module()
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "tiktoken" or name.startswith("tiktoken."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocked_import
    try:
        tokens, method = mod.count_tokens(SAMPLE)
    finally:
        builtins.__import__ = real_import

    assert method == "fallback"
    assert tokens == len(SAMPLE) // 4


def test_paths_agree_within_tolerance():
    """tiktoken and chars//4 should be in the same order of magnitude.

    Skipped when tiktoken is not installed (nothing to compare against).
    """
    mod = _load_module()
    try:
        import tiktoken  # noqa: F401
    except Exception:
        return

    real_tokens, real_method = mod.count_tokens(SAMPLE)
    assert real_method == "tiktoken"

    fallback_tokens = len(SAMPLE) // 4

    # The chars//4 heuristic is a rough proxy; require it within +/-50% of the
    # real count so the fallback stays a usable budget gate, not a wild guess.
    lower = real_tokens * 0.5
    upper = real_tokens * 1.5
    assert lower <= fallback_tokens <= upper, (
        f"fallback {fallback_tokens} not within 50% of tiktoken {real_tokens}"
    )


def test_cli_file_output_schema(tmp_path):
    f = tmp_path / "sample.md"
    f.write_text(SAMPLE, encoding="utf-8")
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(f)],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    assert set(data.keys()) == {"tokens", "method"}
    assert isinstance(data["tokens"], int)
    assert data["method"] in ("tiktoken", "fallback")
    assert data["tokens"] > 0


def test_cli_stdin_output_schema():
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--stdin"],
        input=SAMPLE, capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    assert set(data.keys()) == {"tokens", "method"}
    assert isinstance(data["tokens"], int)
    assert data["method"] in ("tiktoken", "fallback")


def test_cli_file_and_stdin_agree():
    """The CLI over a file and over stdin produce the same count for same text."""
    import tempfile, os
    fd, name = tempfile.mkstemp(suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(SAMPLE)
        file_out = json.loads(subprocess.run(
            [sys.executable, str(SCRIPT), name],
            capture_output=True, text=True, check=True,
        ).stdout)
    finally:
        os.unlink(name)
    stdin_out = json.loads(subprocess.run(
        [sys.executable, str(SCRIPT), "--stdin"],
        input=SAMPLE, capture_output=True, text=True, check=True,
    ).stdout)
    assert file_out == stdin_out


def test_cli_requires_an_input():
    """No file and no --stdin is a usage error (exit 2 from argparse)."""
    res = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True, text=True,
    )
    assert res.returncode != 0


def _run_all():
    import tempfile
    failures = 0
    tests = [
        test_tiktoken_path,
        test_fallback_path_when_import_blocked,
        test_paths_agree_within_tolerance,
        test_cli_stdin_output_schema,
        test_cli_file_and_stdin_agree,
        test_cli_requires_an_input,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {e}")
    # tmp_path-based test handled separately
    with tempfile.TemporaryDirectory() as d:
        try:
            test_cli_file_output_schema(Path(d))
            print("PASS test_cli_file_output_schema")
        except Exception as e:
            failures += 1
            print(f"FAIL test_cli_file_output_schema: {e}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
