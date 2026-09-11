"""Windows npm.cmd must see the checkout directory, not the launch cwd.

Launching ``hermes desktop`` / ``hermes dashboard`` from another project (e.g.
D:\\Work) leaves the Win32 process directory there. Python's ``Popen(cwd=)``
does not override it for ``npm.cmd``, so npm reads the wrong package.json
(``Missing script: "pack"`` / ``No workspaces found``) and the desktop pack
path misdiagnoses that as a blocked Electron download (#46785).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_cli._subprocess_compat import npm_failure_is_wrong_cwd, pinned_win32_cwd
from hermes_cli import main_desktop


def test_pinned_win32_cwd_chdirs_only_on_windows() -> None:
    """The Windows branch chdirs into the checkout and restores; POSIX is a no-op."""
    seen: list[str] = []

    def chdir(path: str) -> None:
        seen.append(path)

    with pinned_win32_cwd(
        r"C:\hermes\apps\desktop",
        is_windows=True,
        chdir=chdir,
        getcwd=lambda: r"D:\Work",
    ):
        pass
    assert seen == [r"C:\hermes\apps\desktop", r"D:\Work"]

    seen.clear()
    with pinned_win32_cwd("/tmp/desktop", is_windows=False, chdir=chdir, getcwd=lambda: "/"):
        pass
    assert seen == []


def test_desktop_pack_skips_electron_mirror_on_wrong_package_json(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """A missing pack script is the wrong package.json, not GitHub blocking Electron."""
    desktop_dir = tmp_path / "apps" / "desktop"
    desktop_dir.mkdir(parents=True)
    staging = desktop_dir / ".staging-pack"
    staging.mkdir()
    redownloads: list[object] = []
    pack_envs: list[dict] = []

    def fake_run(cmd, **kwargs):
        pack_envs.append(dict(kwargs.get("env") or {}))
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr='npm error Missing script: "pack"\n',
        )

    monkeypatch.setattr(main_desktop.subprocess, "run", fake_run)
    monkeypatch.setattr(main_desktop, "_desktop_packaged_executable_in", lambda *_a, **_k: None)
    monkeypatch.setattr(main_desktop, "_electron_dist_ok", lambda *_a, **_k: False)
    monkeypatch.setattr(
        main_desktop,
        "_redownload_electron_dist",
        lambda *_a, **_k: redownloads.append(1) or False,
    )
    monkeypatch.setattr(main_desktop, "_purge_electron_build_cache", lambda *_a, **_k: [])
    monkeypatch.setattr(main_desktop, "_stop_desktop_processes_locking_build", lambda *_a, **_k: [])

    result = main_desktop._run_desktop_pack_with_recovery(
        desktop_dir, ["npm.cmd", "run", "pack"], {}, {}, staging,
    )

    assert result.returncode == 1
    assert len(pack_envs) == 1
    assert redownloads == []
    out = capsys.readouterr().out
    assert "Skipping the npmmirror retry" in out
    assert "Re-downloading via a public mirror" not in out
    assert npm_failure_is_wrong_cwd(result.stderr)
    assert npm_failure_is_wrong_cwd("npm error No workspaces found: --workspace=ui-tui --workspace=web")
    assert not npm_failure_is_wrong_cwd("electron failed to download")
