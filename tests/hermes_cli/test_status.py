from types import SimpleNamespace

from hermes_cli.status import show_status


def test_show_status_includes_tavily_key(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-1234567890abcdef")

    show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Tavily" in output
    assert "tvly...cdef" in output


def test_show_status_termux_gateway_section_skips_systemctl(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("TERMUX_VERSION", "0.118.3")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)
    monkeypatch.setattr(status_mod.shutil, "which", lambda name: None)

    def _unexpected_systemctl(*args, **kwargs):
        raise AssertionError("systemctl should not be called in the Termux status view")

    monkeypatch.setattr(status_mod.subprocess, "run", _unexpected_systemctl)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Manager:      Termux / manual process" in output
    assert "Start with:   hermes gateway" in output
    assert "systemd (user)" not in output


def test_show_status_reports_nous_auth_error(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(
        auth_mod,
        "get_nous_auth_status",
        lambda: {
            "logged_in": False,
            "portal_base_url": "https://portal.nousresearch.com",
            "access_expires_at": "2026-04-20T01:00:51+00:00",
            "agent_key_expires_at": "2026-04-20T04:54:24+00:00",
            "has_refresh_token": True,
            "error": "Refresh session has been revoked",
        },
        raising=False,
    )
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Nous Portal   ✗ not logged in (run: hermes auth add nous --type oauth)" in output
    assert "Error:      Refresh session has been revoked" in output
    assert "Access exp:" in output
    assert "Key exp:" in output


def test_show_status_reports_vercel_backend_contract(monkeypatch, capsys, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "vercel_sandbox")
    monkeypatch.setenv("TERMINAL_VERCEL_RUNTIME", "python3.13")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "oidc-token")
    monkeypatch.setattr(status_mod.importlib.util, "find_spec", lambda name: object() if name == "vercel" else None)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"terminal": {"backend": "vercel_sandbox"}}, raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "find_gateway_pids", lambda exclude_pids=None: [], raising=False)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Backend:      vercel_sandbox" in output
    assert "Runtime:      python3.13" in output
    assert "Auth:" in output and "OIDC token via VERCEL_OIDC_TOKEN" in output
    assert "Auth detail:  mode: OIDC" in output
    assert "Auth detail:  active env: VERCEL_OIDC_TOKEN" in output
    assert "oidc-token" not in output
    assert "snapshot filesystem" in output
    assert "live processes do not survive" in output


def _minimal_status_test_setup(monkeypatch, tmp_path):
    from hermes_cli import status as status_mod
    import hermes_cli.auth as auth_mod
    import hermes_cli.gateway as gateway_mod

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(status_mod, "get_env_path", lambda: tmp_path / ".env", raising=False)
    monkeypatch.setattr(status_mod, "get_hermes_home", lambda: tmp_path, raising=False)
    monkeypatch.setattr(status_mod, "load_config", lambda: {"model": "gpt-5.4"}, raising=False)
    monkeypatch.setattr(status_mod, "resolve_requested_provider", lambda requested=None: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "resolve_provider", lambda requested=None, **kwargs: "openai-codex", raising=False)
    monkeypatch.setattr(status_mod, "provider_label", lambda provider: "OpenAI Codex", raising=False)
    monkeypatch.setattr(auth_mod, "get_nous_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_codex_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_qwen_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(auth_mod, "get_minimax_oauth_auth_status", lambda: {}, raising=False)
    monkeypatch.setattr(gateway_mod, "get_gateway_runtime_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("skip gateway snapshot")), raising=False)
    return status_mod


def test_show_status_reports_passwordless_sudo(monkeypatch, capsys, tmp_path):
    status_mod = _minimal_status_test_setup(monkeypatch, tmp_path)

    class _Result:
        returncode = 0

    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(status_mod.shutil, "which", lambda name: "/usr/bin/sudo" if name == "sudo" else None)
    monkeypatch.setattr(status_mod.subprocess, "run", lambda *args, **kwargs: _Result())

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Sudo:         ✓ enabled (non-interactive)" in output


def test_show_status_reports_sudo_password(monkeypatch, capsys, tmp_path):
    status_mod = _minimal_status_test_setup(monkeypatch, tmp_path)

    monkeypatch.setenv("SUDO_PASSWORD", "secret")

    def _unexpected_subprocess(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when SUDO_PASSWORD is set")

    monkeypatch.setattr(status_mod.subprocess, "run", _unexpected_subprocess)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Sudo:         ✓ enabled (SUDO_PASSWORD)" in output


def test_show_status_reports_disabled_when_passwordless_probe_fails(monkeypatch, capsys, tmp_path):
    status_mod = _minimal_status_test_setup(monkeypatch, tmp_path)

    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(status_mod.shutil, "which", lambda name: "/usr/bin/sudo" if name == "sudo" else None)

    def _raise_probe_failure(*args, **kwargs):
        raise OSError("sudo probe failed")

    monkeypatch.setattr(status_mod.subprocess, "run", _raise_probe_failure)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Sudo:         ✗ disabled" in output


def test_show_status_reports_disabled_when_passwordless_probe_returns_nonzero(monkeypatch, capsys, tmp_path):
    status_mod = _minimal_status_test_setup(monkeypatch, tmp_path)

    class _Result:
        returncode = 1

    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(status_mod.shutil, "which", lambda name: "/usr/bin/sudo" if name == "sudo" else None)
    monkeypatch.setattr(status_mod.subprocess, "run", lambda *args, **kwargs: _Result())

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Sudo:         ✗ disabled" in output


def test_show_status_reports_disabled_when_sudo_is_missing(monkeypatch, capsys, tmp_path):
    status_mod = _minimal_status_test_setup(monkeypatch, tmp_path)

    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    monkeypatch.setattr(status_mod.shutil, "which", lambda name: None)

    def _unexpected_probe(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when sudo is missing")

    monkeypatch.setattr(status_mod.subprocess, "run", _unexpected_probe)

    status_mod.show_status(SimpleNamespace(all=False, deep=False))

    output = capsys.readouterr().out
    assert "Sudo:         ✗ disabled" in output
