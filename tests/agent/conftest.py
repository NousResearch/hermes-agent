"""Agent-test hermeticity fixtures.

The global hermetic fixture in ``tests/conftest.py`` blanks credential env
vars and redirects ``HERMES_HOME``, but it deliberately does NOT redirect
``HOME`` (doing so broke CI subprocesses). That leaves one machine-specific
leak for ``test_anthropic_adapter.py``:

``agent.anthropic_adapter.read_claude_code_credentials`` reads the macOS
Keychain entry ``"Claude Code-credentials"`` (via ``security
find-generic-password``) BEFORE the ``~/.claude/.credentials.json`` file.
The adapter tests stub ``Path.home`` for the *file* source, but nothing
intercepts the Keychain source. On a developer Mac running Claude Code
>= 2.1.114 the real OAuth token leaks past the ``Path.home`` stub and fails
~14 assertions that expect "no creds resolved". CI (Linux, no Keychain) and
Macs without Claude Code never see it — so it's invisible until you run the
suite on a logged-in Mac (e.g. the gateway host).

Default the Keychain source to empty for every agent test; the handful of
tests that exercise Keychain resolution set it explicitly.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _block_real_claude_keychain(monkeypatch):
    """Make anthropic_adapter see a non-Darwin platform by default so the
    macOS Keychain Claude-Code credential read can't fire.

    ``read_claude_code_credentials`` early-returns when
    ``platform.system() != "Darwin"``, so defaulting the adapter's view of
    the platform to "Linux" blocks the real ``security find-generic-password``
    call without touching the credential-resolution functions themselves.

    Tests that *exercise* Keychain behaviour re-patch
    ``agent.anthropic_adapter.platform.system`` to ``"Darwin"`` (and mock
    ``subprocess.run``) inside the test body — those context-manager patches
    apply after this fixture and win, so this does not interfere with them.
    It only neutralises the ambient real-Keychain leak on a dev Mac.
    """
    try:
        import agent.anthropic_adapter as _aa
    except Exception:
        return
    monkeypatch.setattr(_aa.platform, "system", lambda: "Linux", raising=False)
