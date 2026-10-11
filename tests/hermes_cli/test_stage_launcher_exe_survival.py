"""A native launcher exe that cannot be re-minted must survive the ``.cmd``.

``stage_launcher`` publishes a ``.cmd`` fallback when distlib is unavailable
(a uv-managed base has neither pip nor distlib). It used to delete any
existing ``{name}.exe`` beside the fallback so the stale exe could not shadow
the new command — but the exe's absolute path is pinned by consumers that
never go through PATH resolution: an NSSM gateway service with
``Application = <home>\\bin\\hermes.exe`` dies on the next ``sc start`` and
``hermes update`` ends in the resume crash (#126279).

The exe is only removed when it is verifiably dead: not one of our launcher
archives, the obsolete pre-pm venv trampoline, a launcher whose embedded
interpreter is gone, or one booting another install's root. A bootable
launcher for this install stays: it resolves the install's code at runtime.
"""

import io
from pathlib import Path
from zipfile import ZipFile

import pytest

from hermes_cli import _launchers


def _fake_exe(path: Path, script: str, interpreter: Path) -> None:
    """A distlib-shaped launcher exe: loader prefix, interpreter shebang, zip."""
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("__main__.py", script)
    prefix = b"MZ fake loader\n#!" + str(interpreter).encode("utf-8") + b" -I\n"
    path.write_bytes(prefix + payload.getvalue())


def _older_script(root: Path) -> str:
    """A previous revision of the bootstrap script for the same install."""
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(root.resolve())!r})\n"
        "from hermes_cli.main import main\n"
        "sys.exit(main())\n"
    )


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A Windows-shaped bin dir plus a materialized store interpreter."""
    root = tmp_path / "hermes-agent"
    root.mkdir()
    store_python = tmp_path / "store" / "python.exe"
    store_python.parent.mkdir()
    store_python.write_bytes(b"MZ fake store python")
    out_dir = tmp_path / "bin"
    out_dir.mkdir()
    monkeypatch.setattr(_launchers, "_is_windows", lambda: True)
    monkeypatch.setattr(_launchers, "_load_script_maker", lambda: None)
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda _root: store_python)
    return root, out_dir, store_python


def test_bootable_native_launcher_survives_cmd_fallback(staged):
    root, out_dir, store_python = staged
    exe = out_dir / "hermes.exe"
    _fake_exe(exe, _older_script(root), store_python)

    staged_path = _launchers.stage_launcher("hermes", root, out_dir)

    assert staged_path == out_dir / "hermes.cmd"
    assert (out_dir / "hermes.cmd").is_file()
    # The service's Application target survives the fallback.
    assert exe.exists()


def test_launcher_with_dead_interpreter_is_replaced(staged, tmp_path):
    root, out_dir, _store_python = staged
    exe = out_dir / "hermes.exe"
    _fake_exe(exe, _older_script(root), tmp_path / "repinned-away" / "python.exe")

    staged_path = _launchers.stage_launcher("hermes", root, out_dir)

    assert staged_path == out_dir / "hermes.cmd"
    assert not exe.exists()


def test_launcher_booting_another_install_is_replaced(staged, tmp_path):
    root, out_dir, store_python = staged
    other_root = tmp_path / "moved-install"
    other_root.mkdir()
    exe = out_dir / "hermes.exe"
    _fake_exe(exe, _older_script(other_root), store_python)

    staged_path = _launchers.stage_launcher("hermes", root, out_dir)

    assert staged_path == out_dir / "hermes.cmd"
    assert not exe.exists()


def test_venv_trampoline_is_replaced(staged):
    root, out_dir, _store_python = staged
    venv_python = root / "venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    trampoline = out_dir / "hermes.exe"
    trampoline.write_bytes(b"MZ " + str(venv_python).encode("utf-8"))

    staged_path = _launchers.stage_launcher("hermes", root, out_dir)

    assert staged_path == out_dir / "hermes.cmd"
    assert not trampoline.exists()


def test_foreign_exe_is_replaced(staged):
    root, out_dir, _store_python = staged
    foreign = out_dir / "hermes.exe"
    foreign.write_bytes(b"MZ not a launcher")

    staged_path = _launchers.stage_launcher("hermes", root, out_dir)

    assert staged_path == out_dir / "hermes.cmd"
    assert not foreign.exists()


def test_missing_exe_still_stages_the_cmd_fallback(staged):
    root, out_dir, _store_python = staged

    staged_path = _launchers.stage_launcher("hermes", root, out_dir)

    assert staged_path == out_dir / "hermes.cmd"
    assert not (out_dir / "hermes.exe").exists()


def test_launcher_exe_replaceable_verdicts(staged):
    """The kept exe embeds this install's root; the fallback never replaces it."""
    root, out_dir, store_python = staged
    exe = out_dir / "hermes.exe"
    _fake_exe(exe, _older_script(root), store_python)

    assert _launchers._launcher_exe_replaceable(exe, root) is False
    # A copy whose shebang points at a vanished interpreter is replaceable.
    dead = out_dir / "dead.exe"
    _fake_exe(dead, _older_script(root), Path("X:\\vanished\\python.exe"))
    assert _launchers._launcher_exe_replaceable(dead, root) is True
