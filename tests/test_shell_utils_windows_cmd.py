from tools.environments import shell_utils


def test_cmd_mode_uses_cmd_d_flag_and_no_shell(monkeypatch):
    monkeypatch.setattr(shell_utils, "get_local_shell_mode", lambda: "cmd")
    monkeypatch.setattr(shell_utils.os, "name", "nt", raising=False)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    args, kwargs, mode = shell_utils.build_local_subprocess_invocation(
        command="echo hello",
        work_dir=r"C:\tmp",
    )

    assert mode == "cmd"
    assert kwargs["shell"] is False
    assert args[0].lower().endswith("cmd.exe")
    lowered = [a.lower() for a in args]
    assert "/d" in lowered
    assert "/c" in lowered
    # Working dir is set via Popen cwd=, not embedded "cd /d" (avoids quote-escaping breakage)
    assert kwargs.get("cwd", "").lower().rstrip("\\").endswith("c:\\tmp")
    assert any("echo hello" in part.lower() for part in args)


def test_cmd_mode_keeps_percent_env_var_syntax(monkeypatch):
    monkeypatch.setattr(shell_utils, "get_local_shell_mode", lambda: "cmd")
    monkeypatch.setattr(shell_utils.os, "name", "nt", raising=False)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    args, _, mode = shell_utils.build_local_subprocess_invocation(
        command="echo %COMSPEC%",
        work_dir=r"C:\tmp",
    )

    assert mode == "cmd"
    payload = " ".join(args)
    assert "%COMSPEC%" in payload


def test_cmd_mode_strips_trailing_backslash_in_cd(monkeypatch):
    monkeypatch.setattr(shell_utils, "get_local_shell_mode", lambda: "cmd")
    monkeypatch.setattr(shell_utils.os, "name", "nt", raising=False)
    monkeypatch.setenv("COMSPEC", r"C:\Windows\System32\cmd.exe")

    args, kwargs, mode = shell_utils.build_local_subprocess_invocation(
        command="echo hello",
        work_dir=r"C:\hermes\workspace\\",
    )

    assert mode == "cmd"
    # cwd is set via Popen; safe_cwd must not end with single \ (breaks Windows).
    cwd = kwargs.get("cwd", "")
    assert cwd
    assert cwd.endswith(":\\") or not cwd.endswith("\\")


def test_windows_path_safe_for_quotes_keeps_drive_root():
    assert shell_utils._windows_path_safe_for_quotes(r"C:\\") == "C:\\"
    assert shell_utils._windows_path_safe_for_quotes(r"C:\tmp\\") == r"C:\tmp"
