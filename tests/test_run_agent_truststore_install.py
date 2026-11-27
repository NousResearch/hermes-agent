"""Regression for #126808 — importing the agent runner must install platform trust.

Trust is installed by every Hermes *entry point* (the ``hermes`` CLI, the
``hermes-agent`` console script, ``tui_gateway``, ``acp_adapter``). A process that
embeds the agent the way ``run_agent``'s own docstring advertises
(``from run_agent import AIAgent``) reaches none of them, so trust was installed
lazily instead — after the provider SDK had already been imported. ``botocore``
caches ``ssl.SSLContext`` at its own import time, so an injection that lands after
that import leaves every later ``botocore.httpsession.create_urllib3_context()``
recursing until the stack is exhausted.

The contract: importing ``run_agent`` puts the platform store in force before any
provider SDK import can capture the pre-injection class.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("truststore")
pytest.importorskip("botocore")

CHECKOUT_ROOT = Path(__file__).resolve().parents[1]


def _run_child(body: str, tmp_path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(CHECKOUT_ROOT)
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(tmp_path),
        env=env,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    return child


def test_importing_run_agent_installs_the_platform_trust_store(tmp_path):
    """No entry point ran and nothing asked for a client: the import is the guarantee."""
    _run_child(
        """
        import ssl

        import run_agent  # noqa: F401
        import truststore

        assert ssl.SSLContext is truststore.SSLContext, (
            "importing run_agent left the stdlib SSLContext in place, so a provider "
            "SDK imported next caches the pre-injection class and recurses once "
            "trust is installed"
        )
        """,
        tmp_path,
    )


def test_platform_trust_precedes_the_provider_sdk_import(tmp_path):
    """The reported failure: the SDK import beats the (lazy) trust installation."""
    _run_child(
        """
        import run_agent  # noqa: F401  — trust must be in force before the SDK import
        from botocore.httpsession import create_urllib3_context

        from agent.ssl_verify import resolve_httpx_verify

        resolve_httpx_verify()  # the lazy install path the bug leaned on
        create_urllib3_context()  # RecursionError when the SDK cached the old class
        """,
        tmp_path,
    )
