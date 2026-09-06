"""Fixtures shared across hermes_cli tests."""

from __future__ import annotations

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_CJK_SRC = _REPO / "native" / "fts5_cjk" / "fts5_cjk.c"
_CJK_VENDOR = _REPO / "native" / "fts5_cjk" / "vendor"


@pytest.fixture(scope="session")
def cjk_so(tmp_path_factory):
    """Build the cjk_unicode61 loadable tokenizer, or skip when the toolchain cannot."""
    if shutil.which("gcc") is None or not _CJK_SRC.exists():
        pytest.skip("no C toolchain / tokenizer source")
    output = tmp_path_factory.mktemp("hermes-cli-fts5cjk") / "libfts5_cjk.so"
    try:
        subprocess.run(
            [
                "gcc", "-shared", "-fPIC", "-O2",
                f"-I{_CJK_VENDOR}", str(_CJK_SRC), "-o", str(output),
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"cjk tokenizer build failed: {(exc.stderr or '')[:200]}")
    probe = sqlite3.connect(":memory:")
    try:
        probe.enable_load_extension(True)
        probe.load_extension(str(output))
        probe.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError) as exc:
        pytest.skip(f"extension loading unavailable: {exc}")
    finally:
        probe.close()
    return output


@pytest.fixture
def all_assignees_spawnable(monkeypatch):
    """Pretend every assignee maps to a real Hermes profile.

    Most dispatcher tests use synthetic assignees ("alice", "bob") that
    don't correspond to actual profile directories on disk. Without this
    patch, the dispatcher's profile-exists guard (PR #20105) routes
    those tasks into ``skipped_nonspawnable`` instead of spawning, which
    would break tests that assert spawn behavior.
    """
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)


@pytest.fixture(autouse=True)
def _suppress_concurrent_hermes_gate(request, monkeypatch):
    """Default ``_detect_concurrent_hermes_instances`` to ``[]`` for every test.

    The Windows update path now refuses to proceed when another
    ``hermes.exe`` is detected (issue #26670). On a developer's Windows
    machine running the test suite via ``hermes`` itself, this would
    flag the running agent as a concurrent instance and abort every
    ``cmd_update`` test. Tests that want to exercise the gate explicitly
    re-patch ``_detect_concurrent_hermes_instances`` with their own
    return value — autouse here gives a clean default without touching
    the rest of the suite.

    Tests that need to call the REAL function (e.g. unit tests for the
    helper itself) opt out with ``@pytest.mark.real_concurrent_gate``.
    """
    if request.node.get_closest_marker("real_concurrent_gate"):
        return
    try:
        from hermes_cli import main as _cli_main
    except Exception:
        return
    # raising=False: under pytest's per-test spawn isolation, a concurrent
    # xdist worker importing a module that transitively touches hermes_cli.main
    # can briefly expose a partially-initialized module object here — one where
    # _detect_concurrent_hermes_instances isn't defined yet. A bare setattr
    # would raise AttributeError and error the (unrelated) test. The attribute
    # always exists once main.py finishes importing, so a no-op when it's
    # transiently absent is the correct, race-free default.
    monkeypatch.setattr(
        _cli_main,
        "_detect_concurrent_hermes_instances",
        lambda *_a, **_k: [],
        raising=False,
    )
