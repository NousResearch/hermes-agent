"""A profile a plugin runs elsewhere: this host loads no env file and no secret source from its home.

While ``<root>/profiles_run_elsewhere/<profile>.json`` exists, the profile's home is written by the
backend the plugin runs (for example a sandbox). ``load_hermes_dotenv`` runs at the import of every
``hermes`` start, before any command can refuse, so a home it trusted would let that backend put
variables into, and run programs in, every host process started under the profile:

- the home's ``.env`` and ``.op.env`` are not read, sanitized or published;
- no ``secrets:`` source of its ``config.yaml`` is resolved (``command`` runs a shell command,
  ``onepassword.binary_path`` a program, with no model involved);
- the config-backend bootstrap reads no selector or plane credential from them;
- a host that serves other profiles resolves no secret source for it either
  (``hydrate_profile_secret_sources``).

The project ``.env``, the administrator-managed ``.env`` and the terminal bridge are the host's own
and still apply. A process that is denied the marker folder is the backend the plugin runs, and
loads its own files as before.

The home's ``config.yaml`` is that backend's file too. Importing ``gateway.run`` copies settings of
the launch home's ``config.yaml`` into the process environment, every top-level scalar under its
own name among them; for a marked home it copies none.

The stand-in helpers here only append a line to a file under ``tmp_path`` and print a value.
"""
from __future__ import annotations

import errno
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import env_loader
from hermes_cli import profiles_run_elsewhere as pre

PLUGIN = "sandbox-plugin"
_KEYS = (
    "MHE_DOTENV_KEY", "MHE_SHELL_KEY", "MHE_OP_ENV_KEY", "MHE_OP_KEY", "MHE_CMD_KEY",
    "MHE_PROJECT_KEY", "MHE_MANAGED_KEY", "OP_SERVICE_ACCOUNT_TOKEN", "HERMES_CONFIG_BACKEND",
)


@pytest.fixture(autouse=True)
def _clean_process_env(monkeypatch):
    """Each test starts and ends without this file's variables and with no home marked applied."""
    env_loader.reset_secret_source_cache()
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    yield
    env_loader.reset_secret_source_cache()
    for key in _KEYS:
        os.environ.pop(key, None)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A Hermes root with the default profile and two named profiles, ``ops`` and ``dev``."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    for name in ("ops", "dev"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _mark(root: Path, profile: str = "ops") -> None:
    pre.mark_profile_run_elsewhere(profile, PLUGIN, root=root)


def _unmark(root: Path, profile: str = "ops") -> None:
    assert pre.clear_profile_run_elsewhere(profile, PLUGIN, root=root)
    _forget()


def _forget() -> None:
    """A fresh process: nothing applied for any home, none of this file's variables set."""
    env_loader.reset_secret_source_cache()
    for key in _KEYS:
        os.environ.pop(key, None)


def _probe(home: Path, name: str, value: str = "resolved-by-probe") -> tuple[Path, Path]:
    """A program in the profile's own workspace, where its backend can write one: it appends a
    line to ``<name>.ran`` beside itself and prints *value*. Returns ``(program, ran file)``."""
    folder = home / "workspace"
    folder.mkdir(exist_ok=True)
    ran = folder / f"{name}.ran"
    if os.name == "nt":
        program = folder / f"{name}.cmd"
        program.write_text(f'@echo off\r\necho ran>>"{ran}"\r\necho {value}\r\n', encoding="utf-8")
    else:
        program = folder / name
        program.write_text(f'#!/bin/sh\necho ran >> "{ran}"\necho {value}\n', encoding="utf-8")
        program.chmod(0o755)
    return program, ran


def _secrets(home: Path, secrets: dict) -> None:
    (home / "config.yaml").write_text(json.dumps({"secrets": secrets}), encoding="utf-8")


def _onepassword(home: Path) -> Path:
    """Enable the 1Password source with ``binary_path`` naming a probe; returns its ran file."""
    program, ran = _probe(home, "op-probe")
    _secrets(home, {"onepassword": {"enabled": True, "binary_path": str(program), "cache_ttl_seconds": 0,
                                     "env": {"MHE_OP_KEY": "op://vault/item/field"}}})
    return ran


# ── a marked home ──────────────────────────────────────────────────────────────────────────────


def test_marked_home_dotenv_values_do_not_reach_the_process_env(root, monkeypatch):
    """Not a new variable, and not an override of one this process already has."""
    ops = root / "profiles" / "ops"
    (ops / ".env").write_text("MHE_DOTENV_KEY=from-home\nMHE_SHELL_KEY=from-home\n", encoding="utf-8")
    monkeypatch.setenv("MHE_SHELL_KEY", "from-shell")
    _mark(root)

    loaded = env_loader.load_hermes_dotenv(hermes_home=ops)

    assert loaded == []
    assert "MHE_DOTENV_KEY" not in os.environ
    assert os.environ["MHE_SHELL_KEY"] == "from-shell"


def test_marked_home_op_env_is_not_loaded(root):
    ops = root / "profiles" / "ops"
    (ops / ".op.env").write_text("OP_SERVICE_ACCOUNT_TOKEN=planted\nMHE_OP_ENV_KEY=planted\n", encoding="utf-8")
    _mark(root)

    env_loader.load_hermes_dotenv(hermes_home=ops)

    assert "OP_SERVICE_ACCOUNT_TOKEN" not in os.environ
    assert "MHE_OP_ENV_KEY" not in os.environ


def test_marked_home_onepassword_binary_path_is_not_executed(root):
    """``secrets.onepassword.binary_path`` is run without a shell on every OS."""
    ops = root / "profiles" / "ops"
    ran = _onepassword(ops)
    _mark(root)

    env_loader.load_hermes_dotenv(hermes_home=ops)

    assert not ran.exists()
    assert "MHE_OP_KEY" not in os.environ


@pytest.mark.platforms("posix")  # the command source needs /bin/sh
def test_marked_home_secrets_command_helper_is_not_run(root):
    ops = root / "profiles" / "ops"
    ran = ops / "command-helper.ran"
    _secrets(ops, {"command": {"enabled": True, "command": f'echo ran >> "{ran}"; echo MHE_CMD_KEY=from-helper'}})
    _mark(root)

    env_loader.load_hermes_dotenv(hermes_home=ops)
    assert not ran.exists()
    assert "MHE_CMD_KEY" not in os.environ

    _unmark(root)
    env_loader.load_hermes_dotenv(hermes_home=ops)
    assert ran.read_text(encoding="utf-8").count("ran") == 1
    assert os.environ["MHE_CMD_KEY"] == "from-helper"


def test_marked_home_asks_no_secret_source_at_all(root, monkeypatch):
    """Whatever source the home's ``secrets:`` block enables, the registry is not asked: the
    ``command`` helper cannot be run on Windows, and a plugin's source is not known here."""
    from agent.secret_sources import registry

    ops = root / "profiles" / "ops"
    _secrets(ops, {"command": {"enabled": True, "command": "exit 0"}})
    asked: list[Path] = []
    real_apply_all = registry.apply_all

    def apply_all(cfg, home_path, **kwargs):
        asked.append(Path(home_path))
        return real_apply_all(cfg, home_path, **kwargs)

    monkeypatch.setattr(registry, "apply_all", apply_all)
    _mark(root)
    env_loader.load_hermes_dotenv(hermes_home=ops)
    assert asked == []

    _unmark(root)
    env_loader.load_hermes_dotenv(hermes_home=ops)
    assert asked == [ops]


def test_marked_home_env_file_is_not_rewritten(root):
    """The loader normalizes a ``.env`` it is about to read (here: Notepad's UTF-16) by rewriting
    the file. A marked home's file is not this host's to rewrite."""
    ops = root / "profiles" / "ops"
    utf16 = "MHE_DOTENV_KEY=from-home\n".encode("utf-16")
    (ops / ".env").write_bytes(utf16)
    _mark(root)

    env_loader.load_hermes_dotenv(hermes_home=ops)
    assert (ops / ".env").read_bytes() == utf16

    _unmark(root)
    env_loader.load_hermes_dotenv(hermes_home=ops)
    assert (ops / ".env").read_bytes().split() == [b"MHE_DOTENV_KEY=from-home"]
    assert os.environ["MHE_DOTENV_KEY"] == "from-home"


def test_marked_process_home_loads_nothing_without_an_explicit_home(root, monkeypatch):
    """What ``hermes -p ops ...`` and a plugin's own host process run at import: the process home."""
    ops = root / "profiles" / "ops"
    (ops / ".env").write_text("MHE_DOTENV_KEY=from-home\n", encoding="utf-8")
    ran = _onepassword(ops)
    monkeypatch.setenv("HERMES_HOME", str(ops))
    _mark(root)

    env_loader.load_hermes_dotenv()

    assert "MHE_DOTENV_KEY" not in os.environ
    assert not ran.exists()


def test_marked_default_profile_loads_nothing_from_the_root_home(root):
    (root / ".env").write_text("MHE_DOTENV_KEY=from-home\n", encoding="utf-8")
    _mark(root, "default")

    env_loader.load_hermes_dotenv(hermes_home=root)
    assert "MHE_DOTENV_KEY" not in os.environ

    # Another profile of the same root is not marked and loads its own files.
    dev = root / "profiles" / "dev"
    (dev / ".env").write_text("MHE_DOTENV_KEY=from-dev\n", encoding="utf-8")
    env_loader.load_hermes_dotenv(hermes_home=dev)
    assert os.environ["MHE_DOTENV_KEY"] == "from-dev"


def test_marked_home_keeps_the_hosts_own_layers(root, tmp_path, monkeypatch):
    """The project ``.env``, the administrator-managed ``.env`` and the terminal bridge are not
    the profile backend's to write, and still apply."""
    from hermes_cli import managed_scope

    ops = root / "profiles" / "ops"
    (ops / ".env").write_text("MHE_DOTENV_KEY=from-home\n", encoding="utf-8")
    project_env = tmp_path / "project.env"
    project_env.write_text("MHE_PROJECT_KEY=from-project\n", encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text("MHE_MANAGED_KEY=from-managed\n", encoding="utf-8")
    monkeypatch.setattr(managed_scope, "get_managed_dir", lambda: managed)
    monkeypatch.setenv("MHE_PROJECT_KEY", "from-shell")
    bridged: list[Path] = []
    monkeypatch.setattr(env_loader, "_reapply_terminal_config_bridge", bridged.append)
    _mark(root)

    loaded = env_loader.load_hermes_dotenv(hermes_home=ops, project_env=project_env)

    assert loaded == [project_env]
    # The marked home's .env counts as absent, so the project .env overrides the shell as it does
    # for a home that has none: whether that file exists is not the profile backend's to decide.
    assert os.environ["MHE_PROJECT_KEY"] == "from-project"
    assert os.environ["MHE_MANAGED_KEY"] == "from-managed"
    assert bridged == [ops]
    assert "MHE_DOTENV_KEY" not in os.environ


def test_marked_home_env_file_does_not_select_the_config_backend(root):
    """The config backend is chosen from the dotenv layers before the first config read. A marked
    home's ``.env`` is not one of them, so it cannot move host processes to another backend."""
    ops = root / "profiles" / "ops"
    (ops / ".env").write_text("HERMES_CONFIG_BACKEND=remote\n", encoding="utf-8")
    assert env_loader.selected_config_backend_name(ops) == "remote"

    _mark(root)
    assert env_loader.selected_config_backend_name(ops) == "file"
    env_loader.load_hermes_dotenv(hermes_home=ops)  # boots the backend its own bootstrap selects
    assert "HERMES_CONFIG_BACKEND" not in os.environ


def test_one_dotenv_load_asks_once_whether_its_home_is_marked(root, monkeypatch):
    """The question costs a path resolve and a folder listing, and every ``hermes`` start asks it at
    import: once per load, with the answer handed on to the load's two config-backend bootstraps."""
    ops = root / "profiles" / "ops"
    (ops / ".env").write_text("MHE_DOTENV_KEY=from-home\n", encoding="utf-8")
    asked: list[Path] = []
    real = pre.run_elsewhere_for_home

    def run_elsewhere_for_home(home):
        asked.append(Path(home))
        return real(home)

    monkeypatch.setattr(pre, "run_elsewhere_for_home", run_elsewhere_for_home)

    env_loader.load_hermes_dotenv(hermes_home=ops)

    assert asked == [ops]
    assert os.environ["MHE_DOTENV_KEY"] == "from-home"


def test_marked_profile_secret_sources_are_not_resolved_by_a_host_that_serves_it(root):
    """A host process of another profile (the host gateway, a dashboard) resolves a served
    profile's sources into that profile's scope. For a marked profile it resolves none."""
    ops = root / "profiles" / "ops"
    ran = _onepassword(ops)
    _mark(root)

    assert env_loader.hydrate_profile_secret_sources(ops) == {}
    assert not ran.exists()

    _unmark(root)
    assert env_loader.hydrate_profile_secret_sources(ops) == {"MHE_OP_KEY": "resolved-by-probe"}
    assert ran.exists()


# ── not marked ─────────────────────────────────────────────────────────────────────────────────


def test_unmarked_home_loads_env_files_and_secret_sources(root):
    """The control: with no marker the same home loads all three, and a marker for another
    profile changes nothing for this one."""
    ops = root / "profiles" / "ops"
    (ops / ".env").write_text("MHE_DOTENV_KEY=from-home\n", encoding="utf-8")
    (ops / ".op.env").write_text("MHE_OP_ENV_KEY=from-op-env\n", encoding="utf-8")
    ran = _onepassword(ops)
    _mark(root, "dev")

    loaded = env_loader.load_hermes_dotenv(hermes_home=ops)

    assert loaded == [ops / ".env"]
    assert os.environ["MHE_DOTENV_KEY"] == "from-home"
    assert os.environ["MHE_OP_ENV_KEY"] == "from-op-env"
    assert ran.read_text(encoding="utf-8").count("ran") == 1
    assert os.environ["MHE_OP_KEY"] == "resolved-by-probe"


def test_process_denied_the_marker_folder_loads_its_own_files(root, monkeypatch):
    """Inside the backend the plugin runs, the marker folder is denied (or not there at all): that
    process is the one the profile's files are for."""
    ops = root / "profiles" / "ops"
    (ops / ".env").write_text("MHE_DOTENV_KEY=from-home\n", encoding="utf-8")
    (ops / ".op.env").write_text("MHE_OP_ENV_KEY=from-op-env\n", encoding="utf-8")
    ran = _onepassword(ops)
    _mark(root)
    folder = pre.run_elsewhere_dir(root)
    real_listdir = os.listdir

    def listdir(path):
        if Path(path) == folder:
            raise PermissionError(errno.EACCES, "Access is denied", str(path))
        return real_listdir(path)

    monkeypatch.setattr(pre.os, "listdir", listdir)

    env_loader.load_hermes_dotenv(hermes_home=ops)

    assert os.environ["MHE_DOTENV_KEY"] == "from-home"
    assert os.environ["MHE_OP_ENV_KEY"] == "from-op-env"
    assert ran.exists()
    assert os.environ["MHE_OP_KEY"] == "resolved-by-probe"


def test_unreadable_marker_folder_loads_no_profiles_home(root):
    """A marker folder that cannot be read (here a regular file in its place) refuses every
    profile, because the host cannot tell which are marked. For the same reason it takes no env
    file and no secret source from any profile home until the folder can be read."""
    dev = root / "profiles" / "dev"
    (dev / ".env").write_text("MHE_DOTENV_KEY=from-home\n", encoding="utf-8")
    ran = _onepassword(dev)
    (root / "profiles_run_elsewhere").write_text("not a folder\n", encoding="utf-8")

    env_loader.load_hermes_dotenv(hermes_home=dev)
    assert "MHE_DOTENV_KEY" not in os.environ
    assert not ran.exists()

    (root / "profiles_run_elsewhere").unlink()
    env_loader.load_hermes_dotenv(hermes_home=dev)
    assert os.environ["MHE_DOTENV_KEY"] == "from-home"
    assert ran.exists()


# ── other paths to the same home ───────────────────────────────────────────────────────────────


def _short_path(path: Path) -> str:
    """The 8.3 form of *path* (the same text where the volume keeps no short names)."""
    import ctypes

    buffer = ctypes.create_unicode_buffer(1024)
    length = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    return buffer.value if 0 < length < len(buffer) else str(path)


def _spell(form: str, home: Path, tmp_path: Path) -> str:
    if form == "forward_slashes":
        return home.as_posix()
    if form == "dot_segments":
        return str(home.parent / "dev" / ".." / home.name)
    if form == "trailing_separator":
        return str(home) + os.sep
    if form == "relative":
        return os.path.relpath(home, Path.cwd())
    if form == "link":
        link = tmp_path / "ops-link"
        try:
            os.symlink(home, link, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            if os.name != "nt":
                pytest.skip(f"cannot create a directory symlink here: {exc}")
            import _winapi

            _winapi.CreateJunction(str(home), str(link))
        return str(link)
    if form == "upper_case_id":
        return str(home.parent / home.name.upper())
    if form == "upper_case_path":
        return str(home).upper()
    if form == "lower_case_drive":
        return str(home)[0].lower() + str(home)[1:]
    if form == "extended_length":
        return "\\\\?\\" + str(home)
    if form == "short_name":
        return _short_path(home)
    raise AssertionError(form)


_WINDOWS_ONLY = pytest.mark.platforms("windows")


@pytest.mark.parametrize("form", [
    "forward_slashes", "dot_segments", "trailing_separator", "relative", "link",
    pytest.param("upper_case_id", marks=_WINDOWS_ONLY),
    pytest.param("upper_case_path", marks=_WINDOWS_ONLY),
    pytest.param("lower_case_drive", marks=_WINDOWS_ONLY),
    pytest.param("extended_length", marks=_WINDOWS_ONLY),
    pytest.param("short_name", marks=_WINDOWS_ONLY),
])
def test_marked_home_is_recognized_through_another_spelling_of_its_path(root, tmp_path, monkeypatch, form):
    """``HERMES_HOME`` reaches a host process as its launcher wrote it: with either separator, a
    trailing one, ``..`` segments, relative, through a link or junction, and on Windows in another
    letter case, with a ``\\\\?\\`` prefix or as an 8.3 name. Each spelling opens the same ``.env``
    (first half), so each must read the same marker (second half)."""
    ops = root / "profiles" / "ops"
    (ops / ".env").write_text("MHE_DOTENV_KEY=from-home\n", encoding="utf-8")
    monkeypatch.chdir(root)
    spelled = _spell(form, ops, tmp_path)

    env_loader.load_hermes_dotenv(hermes_home=spelled)
    assert os.environ["MHE_DOTENV_KEY"] == "from-home"

    _forget()
    _mark(root)
    assert env_loader.load_hermes_dotenv(hermes_home=spelled) == []
    assert "MHE_DOTENV_KEY" not in os.environ


_CONFIG_NAMES = ("MHE_CONFIG_SCALAR", "LD_PRELOAD", "NODE_OPTIONS", "HERMES_TIMEZONE")


def _env_after_gateway_import(home: Path) -> dict:
    """What a fresh host process started under *home* holds for ``_CONFIG_NAMES`` once it has
    imported ``gateway.run``, as ``hermes gateway status`` and the gateway process do."""
    code = ("import json, os, gateway.run; "
            f"print('MHE=' + json.dumps({{n: os.environ.get(n) for n in {_CONFIG_NAMES!r}}}))")
    env = {k: v for k, v in os.environ.items() if k not in _CONFIG_NAMES}
    env["HERMES_HOME"] = str(home)
    done = subprocess.run([sys.executable, "-c", code], cwd=str(Path(__file__).resolve().parents[2]), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
                          check=False)
    lines = [line for line in done.stdout.splitlines() if line.startswith("MHE=")]
    assert lines, f"the child did not report (exit {done.returncode}): {done.stderr[-600:]}"
    return json.loads(lines[-1][len("MHE="):])


def test_marked_home_config_yaml_puts_no_variable_into_a_host_process(root):
    """``gateway.run`` copies settings of the home's ``config.yaml`` into the process environment
    when it is imported: every top-level scalar under the scalar's own name where that name is not
    set yet (``bridge_config_scalars``), and named settings such as ``timezone``. For a marked home
    the backend the plugin runs writes that file, so this is the ``.env`` rule again: a host
    process started under the home takes no variable from it. Otherwise that backend chooses
    ``LD_PRELOAD``, ``NODE_OPTIONS``, ``PYTHONPATH`` or a proxy for the host process and for every
    program it starts."""
    ops = root / "profiles" / "ops"
    planted = {"MHE_CONFIG_SCALAR": "from-config", "LD_PRELOAD": "/marked-home/x.so",
               "NODE_OPTIONS": "--require /marked-home/x.js", "timezone": "Antarctica/Troll"}
    (ops / "config.yaml").write_text(json.dumps(planted), encoding="utf-8")
    exported = {**{name: planted[name] for name in _CONFIG_NAMES[:3]}, "HERMES_TIMEZONE": planted["timezone"]}

    assert _env_after_gateway_import(ops) == exported  # a profile that runs here: unchanged

    _mark(root)
    assert _env_after_gateway_import(ops) == dict.fromkeys(_CONFIG_NAMES)
