def test_update_python_dependencies_prefers_dotvenv_with_uv(monkeypatch, tmp_path, capsys):
    import hermes_cli.main as main_mod

    project_root = tmp_path
    venv_python = project_root / ".venv" / ("Scripts" if main_mod.sys.platform == "win32" else "bin") / ("python.exe" if main_mod.sys.platform == "win32" else "python")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    (project_root / "pyproject.toml").write_text("[project]\nname='hermes-agent'\nversion='0.0.0'\n")

    calls = []

    def fake_run(cmd, cwd=None, check=None, **kwargs):
        calls.append({"cmd": cmd, "cwd": cwd, "check": check, "kwargs": kwargs})

    monkeypatch.setattr(main_mod.shutil, "which", lambda name: "/usr/local/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    main_mod._update_python_dependencies(project_root)

    out = capsys.readouterr().out
    assert "→ Updating Python dependencies..." in out
    assert calls == [
        {
            "cmd": [
                "/usr/local/bin/uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "-e",
                ".",
                "--quiet",
            ],
            "cwd": project_root,
            "check": True,
            "kwargs": {},
        }
    ]


def test_update_python_dependencies_uses_project_venv_pip_without_uv(monkeypatch, tmp_path):
    import hermes_cli.main as main_mod

    project_root = tmp_path
    venv_bin = project_root / "venv" / ("Scripts" if main_mod.sys.platform == "win32" else "bin")
    venv_pip = venv_bin / ("pip.exe" if main_mod.sys.platform == "win32" else "pip")
    venv_pip.parent.mkdir(parents=True)
    venv_pip.write_text("")

    calls = []

    def fake_run(cmd, cwd=None, check=None, **kwargs):
        calls.append({"cmd": cmd, "cwd": cwd, "check": check, "kwargs": kwargs})

    monkeypatch.setattr(main_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    main_mod._update_python_dependencies(project_root)

    assert calls == [
        {
            "cmd": [str(venv_pip), "install", "-e", ".", "--quiet"],
            "cwd": project_root,
            "check": True,
            "kwargs": {},
        }
    ]


def test_update_python_dependencies_bootstraps_pip_when_venv_has_only_python(monkeypatch, tmp_path):
    import hermes_cli.main as main_mod

    project_root = tmp_path
    venv_python = project_root / "venv" / ("Scripts" if main_mod.sys.platform == "win32" else "bin") / ("python.exe" if main_mod.sys.platform == "win32" else "python")
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")

    calls = []

    def fake_run(cmd, cwd=None, check=None, **kwargs):
        calls.append({"cmd": cmd, "cwd": cwd, "check": check, "kwargs": kwargs})

    monkeypatch.setattr(main_mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    main_mod._update_python_dependencies(project_root)

    assert calls == [
        {
            "cmd": [str(venv_python), "-m", "ensurepip", "--upgrade"],
            "cwd": project_root,
            "check": True,
            "kwargs": {},
        },
        {
            "cmd": [str(venv_python), "-m", "pip", "install", "-e", ".", "--quiet"],
            "cwd": project_root,
            "check": True,
            "kwargs": {},
        },
    ]


def test_update_python_dependencies_falls_back_to_uv_sync_when_no_project_venv(monkeypatch, tmp_path):
    import hermes_cli.main as main_mod

    project_root = tmp_path
    (project_root / "pyproject.toml").write_text("[project]\nname='hermes-agent'\nversion='0.0.0'\n")

    calls = []

    def fake_run(cmd, cwd=None, check=None, **kwargs):
        calls.append({"cmd": cmd, "cwd": cwd, "check": check, "kwargs": kwargs})

    monkeypatch.setattr(main_mod.shutil, "which", lambda name: "/usr/local/bin/uv" if name == "uv" else None)
    monkeypatch.setattr(main_mod.subprocess, "run", fake_run)

    main_mod._update_python_dependencies(project_root)

    assert calls == [
        {
            "cmd": ["/usr/local/bin/uv", "sync", "--quiet"],
            "cwd": project_root,
            "check": True,
            "kwargs": {},
        }
    ]
