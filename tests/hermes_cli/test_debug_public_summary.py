"""The PUBLIC debug paste is an allowlisted summary; full logs only reach private storage (#46006).

Subtractive redaction scrubs credentials but not IPs, room IDs, display names or message bodies,
so every surface that publishes to the public paste (``hermes debug share``, the ``/debug`` gateway
command, the dashboard ``POST /api/ops/debug-share``) must upload no log line text at all. The
private ``--nous`` bundle still carries the redacted tails.
"""

import gzip
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# PII that subtractive redaction does not recognise (message text, names, IDs, IPs, paths).
_MESSAGE = "please wire 4000 to my landlord at 12 Elm Street"
_DISPLAY_NAME = "Alice Wonderland"
_ROOM_ID = "!roomXYZ123:matrix.example.org"
_IP = "203.0.113.77"
_HOME_PATH = "/home/alice/secret-project"
_PII = (_MESSAGE, _DISPLAY_NAME, _ROOM_ID, _IP, _HOME_PATH, "landlord", "Elm Street")


@pytest.fixture
def pii_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    logs = home / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        "model:\n  default: secret-model-name\n  provider: openrouter\n"
        f"terminal:\n  cwd: {_HOME_PATH}\n"
        "custom_providers:\n  - name: alice-private-endpoint\n"
        "toolsets: [web, alice-custom-toolset]\n"
        "alice_unknown_key: hello\n")
    (logs / "agent.log").write_text(
        f"2026-04-12 17:00:00,001 INFO [sess-1] gateway.platforms.matrix: inbound from "
        f"{_DISPLAY_NAME} in {_ROOM_ID}: {_MESSAGE}\n"
        f"2026-04-12 17:00:01,002 WARNING [sess-1] agent.tool_executor: Tool terminal returned "
        f"error: cd {_HOME_PATH} failed for {_DISPLAY_NAME}\n"
        f"2026-04-12 17:00:02,003 ERROR [sess-1] agent.conversation_loop: API call failed: "
        f"openai.RateLimitError: HTTP 429 from {_IP} while answering '{_MESSAGE}'\n"
        "Traceback (most recent call last):\n"
        f'  File "{_HOME_PATH}/x.py", line 1, in <module>\n'
        f"openai.RateLimitError: Error code: 429 - quota for {_DISPLAY_NAME}\n")
    (logs / "errors.log").write_text(
        f"2026-04-12 17:00:02,003 ERROR gateway.run: ConnectionResetError talking to {_IP}\n")
    (logs / "gateway.log").write_text(
        f"2026-04-12 17:00:03,004 INFO gateway.run: {_DISPLAY_NAME} said {_MESSAGE}\n")
    return home


def _assert_no_pii(text: str) -> None:
    for needle in _PII:
        assert needle not in text, f"public paste leaked {needle!r}"
    for log_line in ("inbound from", "Tool terminal returned", "API call failed", "Traceback"):
        assert log_line not in text, f"public paste carries log line text: {log_line!r}"


def _capture_uploads():
    uploaded: list[str] = []

    def _upload(content, expiry_days=1):
        uploaded.append(content)
        return f"https://paste.rs/p{len(uploaded)}"

    return uploaded, _upload


def test_cli_public_share_uploads_only_the_allowlisted_summary(pii_home, capsys):
    from hermes_cli.debug import run_debug_share

    uploaded, upload = _capture_uploads()
    with patch("hermes_cli.debug.upload_to_pastebin", side_effect=upload), \
            patch("hermes_cli.debug._schedule_auto_delete"), \
            patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)):
        # --no-redact must not be able to put logs on the public paste either.
        for no_redact in (False, True):
            run_debug_share(SimpleNamespace(lines=200, expire=1, local=False, nous=False,
                                            yes=True, no_redact=no_redact))

    assert len(uploaded) == 2  # one summary paste per share, no full-log pastes
    for text in uploaded:
        _assert_no_pii(text)
        # The useful, allowlisted signal is still there.
        assert "RateLimitError" in text and "ConnectionResetError" in text
        assert "429" in text
        assert "agent.tool_executor" in text
        assert "terminal" in text and "model" in text  # config key names
        assert "secret-model-name" not in text and "alice-custom-toolset" not in text
        assert "alice_unknown_key" not in text and "alice-private-endpoint" not in text


def test_dashboard_builder_uploads_only_the_summary(pii_home):
    from hermes_cli.debug import build_debug_share

    uploaded, upload = _capture_uploads()
    with patch("hermes_cli.debug.upload_to_pastebin", side_effect=upload), \
            patch("hermes_cli.debug._schedule_auto_delete"), \
            patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)):
        result = build_debug_share(log_lines=200, redact=False)

    assert list(result.urls) == ["Report"]
    assert len(uploaded) == 1
    _assert_no_pii(uploaded[0])


@pytest.mark.asyncio
async def test_gateway_debug_uploads_only_the_summary(pii_home):
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}

    async def _inline(fn, *args):
        return fn(*args)

    runner._run_in_executor_with_context = _inline
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c1", user_id="u1", chat_type="dm")
    uploaded: list[str] = []
    with patch("hermes_cli.debug.upload_to_pastebin",
               lambda report: uploaded.append(report) or "https://paste.rs/x"), \
            patch("hermes_cli.debug._schedule_auto_delete"), \
            patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)):
        await runner._handle_debug_command(MessageEvent(text="/debug upload", source=source))

    assert len(uploaded) == 1
    _assert_no_pii(uploaded[0])
    assert "RateLimitError" in uploaded[0]


def test_nous_share_still_carries_redacted_log_tails(pii_home):
    from hermes_cli.debug import run_debug_share

    with patch("hermes_cli.diagnostics_upload.share_to_nous",
               return_value={"id": "x", "viewUrl": "https://v"}) as share, \
            patch("hermes_cli.debug.upload_to_pastebin") as paste, \
            patch("hermes_cli.debug._sweep_expired_pastes", return_value=(0, 0)):
        run_debug_share(SimpleNamespace(lines=200, expire=1, local=False, nous=True,
                                        yes=True, no_redact=False))

    paste.assert_not_called()
    envelope = json.loads(gzip.decompress(share.call_args[0][0]))
    assert _MESSAGE in envelope["files"]["agent.log"]
    assert _MESSAGE in envelope["files"]["report"]


def test_lines_bounds_the_scanned_tail(tmp_path, monkeypatch):
    """``--lines`` is the window the summary counts, as the help and docs say."""
    from hermes_cli.debug_summary import collect_public_summary

    logs = tmp_path / ".hermes" / "logs"
    logs.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (logs / "agent.log").write_text("".join(
        f"2026-04-12 17:00:00,001 WARNING agent.conversation_loop: row {i}\n" for i in range(300)))

    text = collect_public_summary(log_lines=50)
    assert "50 lines scanned; WARNING 50" in text
    assert "x50" in text and "x300" not in text


def test_message_derived_names_never_reach_the_summary():
    """Identifier-shaped tokens from message text or a runtime-built logger name are not printed;
    real module paths and builtin exception names still are."""
    from hermes_cli.debug_summary import _log_counts

    log = (
        "2026-04-12 17:00:00,001 WARNING agent.john_smith_42_main_st_password_hunter2: x\n"
        "2026-04-12 17:00:01,002 ERROR agent.conversation_loop: "
        "home.jsmith.projects.TopSecretError and SecretProjectError, then ConnectionResetError\n"
        "httpx.ReadTimeout: timed out\n")
    by_logger, exceptions, _codes, _n = _log_counts(log)
    loggers = {name for _level, name in by_logger}
    assert loggers == {"agent.*", "agent.conversation_loop"}
    assert set(exceptions) == {"(other)", "ConnectionResetError", "httpx.ReadTimeout"}
    assert "john_smith" not in repr(by_logger) and "jsmith" not in repr(exceptions)


def test_known_module_prefix_does_not_launder_message_text():
    """A fabricated class after a real package name is still message text, not a diagnosis."""
    from hermes_cli.debug_summary import _log_counts

    private_token = "PrivateName427ElmStreet"
    log = (
        "2026-04-12 17:00:00,001 WARNING agent.conversation_loop: "
        f"received httpx.{private_token}Error while httpx.ReadTimeout was raised\n"
    )
    _, exceptions, _, _ = _log_counts(log)
    assert exceptions == {"(other)": 1, "httpx.ReadTimeout": 1}
    assert private_token not in repr(exceptions)


def test_exception_check_keeps_real_classes_and_runs_no_module_code():
    """Hermes and dependency exception classes stay named; verifying a log-derived path must not
    import it (``acp_adapter.__main__`` starts the ACP server on import)."""
    import acp_adapter.entry
    from hermes_cli.debug_summary import _exception_label

    with patch.object(acp_adapter.entry, "main") as server_main:
        assert _exception_label("acp_adapter.__main__.StartupError") == "(other)"
    server_main.assert_not_called()
    assert _exception_label("agent.chat_completion_helpers.ProviderStreamError") == \
        "agent.chat_completion_helpers.ProviderStreamError"
    assert _exception_label("openai.RateLimitError") == "openai.RateLimitError"
    assert _exception_label("asyncio.CancelledError") == "asyncio.CancelledError"
    assert _exception_label("agent.chat_completion_helpers.Jsmith42Error") == "(other)"


def test_info_lines_and_unknown_logger_roots_name_nothing():
    """INFO lines (where adapters log message text) are not mined, and a logger outside the known
    roots (one named after an MCP server or a user) is counted as (other), never printed."""
    from hermes_cli.debug_summary import _log_counts

    log = (
        "2026-04-12 17:00:00,001 INFO agent.conversation_loop: ConnectionResetError HTTP 502\n"
        "2026-04-12 17:00:01,002 WARNING alice_private_server: x\n")
    by_logger, exceptions, codes, _n = _log_counts(log)
    assert {name for _level, name in by_logger} == {"(other)"}
    assert not exceptions and not codes


def test_custom_platform_plugin_names_are_counted_not_printed(pii_home):
    """Plugins register platform identifiers at runtime under any name (a person, a phone number);
    only platforms core ships (built-in members and bundled plugins) are named."""
    from gateway.platform_registry import PlatformEntry, platform_registry
    from hermes_cli.debug_summary import _platform_names

    private = "alice_private_oncall_15551234567"
    (pii_home / "config.yaml").write_text(
        "platforms:\n"
        "  telegram: {enabled: true, token: '123:abc'}\n"
        "  irc: {enabled: true, token: x}\n"
        f"  {private}: {{enabled: true, token: x}}\n")
    platform_registry.register(PlatformEntry(
        name=private, label="x", adapter_factory=lambda _cfg: None, check_fn=lambda: True,
        source="plugin"))
    try:
        assert _platform_names() == "irc, telegram, +1 custom"
    finally:
        platform_registry.unregister(private)


def test_symlinked_source_outside_the_package_does_not_vouch_for_a_class(
        tmp_path, monkeypatch):
    """A module file symlinked out of a known package (or a star-import sibling that is) can hold
    any text; a class name quoted there is not verified, and empty dotted segments never match."""
    import sys

    from hermes_cli import debug_summary as ds

    outside = tmp_path / "private.py"
    outside.write_text("# AliceOncall15551234567Error\n")
    pkg = tmp_path / "site" / "zzpkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("from .stars import *\nclass RealError(Exception): ...\n")
    (pkg / "linked.py").symlink_to(outside)
    (pkg / "stars.py").symlink_to(outside)
    (pkg / "real.py").write_text("class RealError(Exception): ...\n")
    monkeypatch.syspath_prepend(str(pkg.parent))
    monkeypatch.setattr(ds, "_KNOWN_LOGGER_ROOTS", ds._KNOWN_LOGGER_ROOTS | {"zzpkg"})
    monkeypatch.delitem(sys.modules, "zzpkg", raising=False)
    ds._resolve_module.cache_clear()
    ds._module_names.cache_clear()
    try:
        assert ds._exception_label("zzpkg.real.RealError") == "zzpkg.real.RealError"
        assert ds._exception_label("zzpkg.linked.AliceOncall15551234567Error") == "(other)"
        assert ds._exception_label("zzpkg.AliceOncall15551234567Error") == "(other)"
        assert ds._module_path("zzpkg...real") == "zzpkg.*"
        assert ds._module_path("agent...conversation_loop") == "agent.*"
    finally:
        ds._resolve_module.cache_clear()
        ds._module_names.cache_clear()
