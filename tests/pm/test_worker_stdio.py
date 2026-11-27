"""Worker control traffic must never become a subprocess's standard input."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import textwrap

import pytest

from tests.pm._fixtures import isolated_python as isolated_python  # noqa: PLC0414 -- the self-alias is load-bearing: it suppresses F811 for the pytest fixture parameter shadowing this import


@pytest.mark.parametrize("streaming", [False, True])
def test_worker_children_get_eof_while_control_pipe_stays_open(tmp_path, isolated_python, streaming):
    root = Path(__file__).resolve().parents[2]
    # Inject an operation, not a process mock: main owns the real protocol reader
    # and the Python engine launches an actual child with inherited standard IO.
    script = textwrap.dedent(f"""
        import os, sys
        from pathlib import Path
        sys.path.insert(0, {str(root)!r})
        from pm import build_operations, worker
        from pm.environment import PythonEnvironment

        def probe(cache, ci=False):
            environment = PythonEnvironment(
                uv=Path(sys.executable), python=Path(sys.executable),
                destination=Path(cache) / 'unused', cache=Path(cache),
                env=dict(os.environ), output=sys.stderr if {streaming!r} else None,
            )
            result = environment._run(
                ['-I', '-c', "import sys; assert sys.stdin.read() == ''; print('EOF_OK')"],
                cwd=Path(cache), timeout=10,
            )
            if result.returncode:
                raise RuntimeError(result.stderr)
            assert 'EOF_OK' in result.stdout + result.stderr
            return 'child completed'

        build_operations.prune_cache = probe
        worker.main()
    """)
    request = {
        "id": "stdio-probe", "operation": "prune_cache",
        "arguments": {"cache": str(tmp_path)}, "callbacks": [], "packages": [],
        "context": {"repo": str(root), "lockfile": str(root / "pm/lock.json")},
    }
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "home"),
           "HERMES_RUNTIME_DIR": str(tmp_path / "tools")}
    with (tmp_path / "diagnostics.log").open("w+") as diagnostics:
        with subprocess.Popen([str(isolated_python), "-I", "-c", script],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=diagnostics, text=True, env=env) as process:
            assert process.stdin is not None and process.stdout is not None
            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                # communicate() would close stdin and erase the bug's precondition.
                process.wait(timeout=30)
                response = json.loads(process.stdout.read())
                diagnostics.seek(0)
                assert response.get("result") == "child completed", (response, diagnostics.read())
                assert process.returncode == 0
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)



def test_worker_progress_survives_a_non_utf8_stdio(tmp_path, isolated_python):
    """A piped Windows worker starts with cp1252 stdio; its "→ label…" progress must not fail the operation."""
    root = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(f"""
        import io, sys
        sys.path.insert(0, {str(root)!r})
        # What a Windows interpreter hands a worker whose standard streams are pipes.
        sys.stdout = io.TextIOWrapper(io.FileIO(1, 'w', closefd=False), encoding='cp1252', line_buffering=True)
        sys.stderr = io.TextIOWrapper(io.FileIO(2, 'w', closefd=False), encoding='cp1252', line_buffering=True)
        from pm import build_operations, worker

        def probe(cache, ci=False):
            print('  \u2192 Preparing build tools\u2026')
            print('stderr \u2192', file=sys.stderr)
            return 'progress written'

        build_operations.prune_cache = probe
        worker.main()
    """)
    request = {
        "id": "encoding-probe", "operation": "prune_cache",
        "arguments": {"cache": str(tmp_path)}, "callbacks": [], "packages": [],
        "context": {"repo": str(root), "lockfile": str(root / "pm/lock.json")},
    }
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "home"), "HERMES_RUNTIME_DIR": str(tmp_path / "tools")}
    completed = subprocess.run([str(isolated_python), "-I", "-c", script], input=json.dumps(request) + "\n",
                               capture_output=True, encoding="utf-8", env=env, timeout=60, check=False)
    response = json.loads(completed.stdout)
    assert response.get("result") == "progress written", (response, completed.stderr)
    assert "\u2192 Preparing build tools\u2026" in completed.stderr
