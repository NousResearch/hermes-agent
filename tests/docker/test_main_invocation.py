"""Harness: docker run <image> [cmd...] invocation patterns.

These tests MUST pass on the current tini-based image AND continue to
pass after the Phase 2 s6 migration. Any behavior drift is a regression.

The harness expects ``built_image`` and ``container_name`` fixtures from
``tests/docker/conftest.py``. When Docker isn't available every test
here is skipped at collection time.
"""
from __future__ import annotations

import subprocess




def test_chat_subcommand_passthrough(built_image: str) -> None:
    """``docker run <image> chat --help`` should exec ``hermes chat --help``.

    Uses ``--help`` so the call doesn't need an upstream model configured.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "chat", "--help"],
        capture_output=True, text=True, timeout=60,
        check=False,
    )
    assert r.returncode == 0
    combined = (r.stdout + r.stderr).lower()
    assert "chat" in combined or "usage" in combined




# Every top-level hermes subcommand that is also a program on the image PATH, i.e. a name the
# wrapper's `command -v "$1"` probe would exec instead of passing it to hermes.
_SHADOWED_SUBCOMMANDS = (
    "import shutil\n"
    "from hermes_cli.main import _build_cli_parser\n"
    "names = sorted(n for n in _build_cli_parser()[1].choices if shutil.which(n))\n"
    "print('SHADOWED:' + ' '.join(names))\n"
)


def test_subcommands_named_like_a_program_on_path_reach_hermes(built_image: str) -> None:
    """``docker run <image> mcp list`` must run ``hermes mcp list``, not the program named ``mcp``.

    The wrapper execs a first argument that is a program on PATH. ``mcp`` (the MCP SDK's
    console script in the venv) and ``login`` (util-linux) shadowed their subcommands: ``mcp``
    failed with the SDK's "typer is required" and ``login`` ran the system login.
    """
    probe = subprocess.run(
        ["docker", "run", "--rm", "-u", "hermes", "--entrypoint", "/opt/hermes/.venv/bin/python",
         built_image, "-c", _SHADOWED_SUBCOMMANDS],
        capture_output=True, text=True, timeout=120, check=False,
    )
    marker = [line for line in probe.stdout.splitlines() if line.startswith("SHADOWED:")]
    assert probe.returncode == 0 and marker, probe.stdout[-1000:] + probe.stderr[-2000:]
    for name in marker[0].removeprefix("SHADOWED:").split():
        r = subprocess.run(
            ["docker", "run", "--rm", built_image, name, "--help"],
            capture_output=True, text=True, timeout=60, check=False,
        )
        assert f"usage: hermes {name}" in r.stdout + r.stderr, (name, r.stdout[-1000:], r.stderr[-1000:])


def test_bash_pattern(built_image: str) -> None:
    """``docker run <image> bash -c 'echo ok'`` should exec bash directly."""
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "bash", "-c", "echo ok"],
        capture_output=True, text=True, timeout=30,
        check=False,
    )
    assert r.returncode == 0
    assert "ok" in r.stdout


def test_container_exit_code_matches_inner_exit(built_image: str) -> None:
    """The container exit code must match the inner process's exit code.

    Critical for CI: ``docker run <image> hermes batch ...`` returns a
    non-zero status when batch fails. Phase 2 (s6) must preserve this.
    """
    r = subprocess.run(
        ["docker", "run", "--rm", built_image, "sh", "-c", "exit 42"],
        capture_output=True, text=True, timeout=30,
        check=False,
    )
    assert r.returncode == 42
