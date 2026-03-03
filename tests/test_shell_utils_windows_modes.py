from tools.environments import shell_utils


def _set_windows(monkeypatch):
    monkeypatch.setattr(shell_utils.os, "name", "nt", raising=False)
    # Reset mode-log state between tests so behavior is deterministic.
    shell_utils._last_logged_mode = None


def test_get_local_shell_mode_auto_prefers_wsl(monkeypatch):
    _set_windows(monkeypatch)
    monkeypatch.setenv("HERMES_WINDOWS_SHELL", "auto")
    monkeypatch.setattr(shell_utils, "_wsl_available", lambda: True)
    monkeypatch.setattr(shell_utils, "_powershell_executable", lambda: r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")

    assert shell_utils.get_local_shell_mode() == "wsl"


def test_get_local_shell_mode_powershell_falls_back_to_cmd(monkeypatch):
    _set_windows(monkeypatch)
    monkeypatch.setenv("HERMES_WINDOWS_SHELL", "powershell")
    monkeypatch.setattr(shell_utils, "_powershell_executable", lambda: "")

    assert shell_utils.get_local_shell_mode() == "cmd"


def test_get_local_shell_mode_switching_follows_env(monkeypatch):
    """Terminal switching: mode must follow HERMES_WINDOWS_SHELL (no stale cache)."""
    _set_windows(monkeypatch)
    monkeypatch.setattr(shell_utils, "_wsl_available", lambda: True)
    monkeypatch.setattr(shell_utils, "_powershell_executable", lambda: "powershell.exe")

    monkeypatch.setenv("HERMES_WINDOWS_SHELL", "wsl")
    shell_utils._last_logged_mode = None
    assert shell_utils.get_local_shell_mode() == "wsl"

    monkeypatch.setenv("HERMES_WINDOWS_SHELL", "powershell")
    shell_utils._last_logged_mode = None
    assert shell_utils.get_local_shell_mode() == "powershell"

    monkeypatch.setenv("HERMES_WINDOWS_SHELL", "wsl")
    shell_utils._last_logged_mode = None
    assert shell_utils.get_local_shell_mode() == "wsl"


def test_build_local_subprocess_invocation_wsl_payload(monkeypatch):
    _set_windows(monkeypatch)
    monkeypatch.setattr(shell_utils, "get_local_shell_mode", lambda: "wsl")
    monkeypatch.setattr(shell_utils, "_wsl_executable", lambda: "wsl.exe")

    args, kwargs, mode = shell_utils.build_local_subprocess_invocation(
        command="echo hello",
        work_dir=r"C:\tmp",
    )

    assert mode == "wsl"
    assert args[:3] == ["wsl.exe", "-e", "bash"]
    assert args[3] == "-lc"
    assert "cd /mnt/c/tmp && echo hello" == args[4]
    assert kwargs["shell"] is False


def test_build_local_subprocess_invocation_powershell_payload(monkeypatch):
    _set_windows(monkeypatch)
    monkeypatch.setattr(shell_utils, "get_local_shell_mode", lambda: "powershell")
    monkeypatch.setattr(shell_utils, "_powershell_executable", lambda: "powershell.exe")
    monkeypatch.setattr(shell_utils.os.path, "isdir", lambda p: True)

    args, kwargs, mode = shell_utils.build_local_subprocess_invocation(
        command="echo hello",
        work_dir=r"C:\hermes\workspace\\",
    )

    assert mode == "powershell"
    assert args[0].lower().endswith("powershell.exe")
    assert "-Command" in args
    ps_command = args[-1]
    assert "Set-Location -LiteralPath 'C:\\hermes\\workspace'; echo hello" == ps_command
    assert kwargs["shell"] is False

