"""A managed worker trusts the OS certificate store before its first connection."""


def test_managed_worker_trusts_the_platform_store_before_any_connection(tmp_path):
    """The worker is spawned as a bare ``python -m agent.managed_worker`` and connects its
    profile's MCP servers before ``run_agent`` is imported, so no later import can install
    platform trust in time. ``main()`` must install it before it reads its bootstrap; a
    worker that did not would verify against OpenSSL's bundled roots and reject a corporate
    TLS-inspection CA that every other entry point trusts.

    The child is stopped at its first protocol read (empty stdin), which is before any
    network or tool discovery, and reports whether ``ssl.SSLContext`` was injected by then.
    """
    import os
    import subprocess
    import sys

    script = """
import ssl, sys
from agent import managed_worker
def stop(_stream):
    sys.stderr.write('INJECTED=%s\\n' % (ssl.SSLContext.__module__ != 'ssl'))
    raise EOFError('stop before bootstrap')
managed_worker.read_frame = stop
managed_worker.main()
"""
    env = os.environ.copy()
    env.update(HOME=str(tmp_path), HERMES_HOME=str(tmp_path / "home"))
    child = subprocess.run([sys.executable, "-s", "-c", script], env=env, stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=60, check=False)
    assert "INJECTED=True" in child.stderr, child.stderr
