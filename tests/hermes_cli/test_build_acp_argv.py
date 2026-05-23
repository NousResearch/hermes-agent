"""Argv-forwarding regression tests for ``hermes acp`` (issue #30571).

``hermes acp`` is a thin shim that re-dispatches into ``hermes-acp``
(``acp_adapter.entry.main``) with a synthesised argv built from the
parsed argparse Namespace plus the current process env. The interesting
contract is the ``HERMES_PROFILE`` → ``--profile <name>`` forwarding:

* Set + non-default → forward ``--profile <name>``
* Set to literal ``default`` → do NOT forward (default is implicit)
* Unset / empty → do NOT forward

Without this, direct ``hermes-acp`` invocations (used by editor configs
like Zed agent commands) would not honour the parent ``hermes -p <name>
acp`` profile selection, breaking config.yaml / .env / skills isolation.
"""

from __future__ import annotations

from argparse import Namespace

from hermes_cli.main import _build_acp_argv


class TestBuildAcpArgvFlags:
    """The non-profile flags pass through as plain switches."""

    def test_no_flags_returns_empty(self):
        argv = _build_acp_argv(Namespace(), env={})
        assert argv == []

    def test_version_flag(self):
        argv = _build_acp_argv(Namespace(acp_version=True), env={})
        assert argv == ["--version"]

    def test_check_flag(self):
        argv = _build_acp_argv(Namespace(check=True), env={})
        assert argv == ["--check"]

    def test_setup_flag(self):
        argv = _build_acp_argv(Namespace(setup=True), env={})
        assert argv == ["--setup"]

    def test_setup_browser_flag(self):
        argv = _build_acp_argv(Namespace(setup_browser=True), env={})
        assert argv == ["--setup-browser"]

    def test_assume_yes_flag(self):
        argv = _build_acp_argv(Namespace(assume_yes=True), env={})
        assert argv == ["--yes"]

    def test_multiple_flags_preserve_order(self):
        argv = _build_acp_argv(
            Namespace(setup_browser=True, assume_yes=True), env={}
        )
        assert argv == ["--setup-browser", "--yes"]


class TestBuildAcpArgvProfileForwarding:
    """HERMES_PROFILE → ``--profile`` forwarding rules (the #30571 fix)."""

    def test_profile_set_forwards_flag(self):
        argv = _build_acp_argv(Namespace(), env={"HERMES_PROFILE": "coder"})
        assert argv == ["--profile", "coder"]

    def test_profile_default_does_not_forward(self):
        """``HERMES_PROFILE=default`` is the implicit no-op case."""
        argv = _build_acp_argv(Namespace(), env={"HERMES_PROFILE": "default"})
        assert argv == []

    def test_profile_unset_does_not_forward(self):
        argv = _build_acp_argv(Namespace(), env={})
        assert "--profile" not in argv

    def test_profile_empty_does_not_forward(self):
        argv = _build_acp_argv(Namespace(), env={"HERMES_PROFILE": ""})
        assert "--profile" not in argv

    def test_profile_whitespace_does_not_forward(self):
        """Whitespace-only HERMES_PROFILE is treated as unset."""
        argv = _build_acp_argv(Namespace(), env={"HERMES_PROFILE": "   "})
        assert "--profile" not in argv

    def test_profile_strips_surrounding_whitespace(self):
        argv = _build_acp_argv(Namespace(), env={"HERMES_PROFILE": "  coder  "})
        assert argv == ["--profile", "coder"]

    def test_profile_forwarded_after_other_flags(self):
        """``--profile`` is appended last so the receiving argparse sees a
        stable shape regardless of which other flags are set."""
        argv = _build_acp_argv(
            Namespace(check=True), env={"HERMES_PROFILE": "coder"}
        )
        assert argv == ["--check", "--profile", "coder"]

    def test_env_defaults_to_os_environ(self, monkeypatch):
        """Omitting ``env`` should fall back to ``os.environ`` so the
        production call site (``cmd_acp``) behaves the same."""
        monkeypatch.setenv("HERMES_PROFILE", "coder")
        argv = _build_acp_argv(Namespace())
        assert argv == ["--profile", "coder"]

    def test_env_defaults_to_os_environ_unset(self, monkeypatch):
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        argv = _build_acp_argv(Namespace())
        assert "--profile" not in argv
