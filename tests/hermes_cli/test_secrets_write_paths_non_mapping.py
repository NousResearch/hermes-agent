"""``hermes secrets`` commands that write config.yaml must survive a non-mapping ``secrets:`` section.

The read paths already treat ``secrets: bitwarden`` / ``[bitwarden]`` / ``42`` / null and a scalar
``secrets.<source>`` as "not configured". The write paths (bitwarden setup/disable, 1Password
setup/set/remove/disable) used to chain ``setdefault`` on that value and exit with an
``AttributeError``. These tests write a real config.yaml into a temp home, run the command, and check
what the user gets: the exit code, the message, and the config.yaml left behind.
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

import hermes_yaml

REPO = Path(__file__).resolve().parents[2]
SHAPES = {
    "scalar": "secrets: bitwarden\n",
    "list": "secrets: [bitwarden]\n",
    "int": "secrets: 42\n",
    "null": "secrets:\n",
    "source-scalar": "secrets:\n  {source}: true\n",
}
REF = "op://Vault/Item/field"
TOKEN_ENVS = ("BWS_ACCESS_TOKEN", "BWS_SERVER_URL", "OP_SERVICE_ACCOUNT_TOKEN")


def _config(shape: str, source: str) -> str:
    return SHAPES[shape].format(source=source)


def _raw(home: Path) -> dict:
    data = hermes_yaml.safe_load((home / "config.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert isinstance(data.get("secrets", {}), dict), data
    return data


def _section(home: Path, source: str) -> dict:
    section = _raw(home).get("secrets", {}).get(source, {})
    assert isinstance(section, dict), section
    return section


def _env_file(home: Path) -> str:
    path = home / ".env"
    return path.read_text(encoding="utf-8") if path.exists() else ""


@pytest.fixture
def home(tmp_path, monkeypatch):
    for name in TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "200")
    path = tmp_path / ".hermes"
    path.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(path))
    return path


def _bw_setup(monkeypatch):
    import agent.secret_sources.bitwarden as bw_module
    from hermes_cli import secrets_cli

    monkeypatch.setattr(sys, "stdin", io.StringIO())
    monkeypatch.setattr(bw_module, "find_bws", lambda **_kw: Path("/fake/bws"))
    monkeypatch.setattr(bw_module, "fetch_bitwarden_secrets", lambda **_kw: ({"ZZ_BW_KEY": "v"}, []))
    monkeypatch.setattr(secrets_cli, "_bws_version", lambda _binary: "bws 1.0.0")
    return secrets_cli.cmd_setup(argparse.Namespace(
        access_token="0.zz-token", server_url="https://vault.bitwarden.eu", project_id="proj-zz"))


def _op_setup(monkeypatch):
    from hermes_cli import onepassword_secrets_cli as op_cli

    monkeypatch.setattr(op_cli.op_src, "find_op", lambda _path="": Path("/fake/op"))
    monkeypatch.setattr(op_cli, "_op_version", lambda _binary: "2.30.0")
    return op_cli.cmd_setup(argparse.Namespace(account=None, token_env=None, token="ops_zz", binary_path=None))


def _op(name, **kwargs):
    def run(_monkeypatch):
        from hermes_cli import onepassword_secrets_cli as op_cli

        return getattr(op_cli, name)(argparse.Namespace(**kwargs))
    return run


def _bw_disable(_monkeypatch):
    from hermes_cli import secrets_cli

    return secrets_cli.cmd_disable(argparse.Namespace())


def _check_bw_setup(home, out):
    assert "Bitwarden Secrets Manager is enabled." in out
    section = _section(home, "bitwarden")
    assert section["enabled"] is True
    assert section["project_id"] == "proj-zz"
    assert section["server_url"] == "https://vault.bitwarden.eu"
    assert "BWS_ACCESS_TOKEN=0.zz-token" in _env_file(home)


def _check_disabled(source):
    def check(home, out):
        from hermes_cli._secrets_common import section_cfg
        from hermes_cli.config import load_config

        assert "Disabled." in out
        assert not _section(home, source).get("enabled")
        assert not section_cfg(load_config(), source).get("enabled")
    return check


def _check_op_setup(home, out):
    assert "1Password secret source is enabled." in out
    assert _section(home, "onepassword")["enabled"] is True
    assert "OP_SERVICE_ACCOUNT_TOKEN=ops_zz" in _env_file(home)


def _check_op_set(home, out):
    assert "mapped ZZ_OP_KEY" in out
    assert _section(home, "onepassword")["env"] == {"ZZ_OP_KEY": REF}


COMMANDS = {
    # name: (source, runner, expected rc, check(home, output))
    "bitwarden-setup": ("bitwarden", _bw_setup, 0, _check_bw_setup),
    "bitwarden-disable": ("bitwarden", _bw_disable, 0, _check_disabled("bitwarden")),
    "op-setup": ("onepassword", _op_setup, 0, _check_op_setup),
    "op-set": ("onepassword", _op("cmd_set", env_var="ZZ_OP_KEY", reference=REF), 0, _check_op_set),
    "op-remove": ("onepassword", _op("cmd_remove", env_var="ZZ_OP_KEY"), 1, None),
    "op-disable": ("onepassword", _op("cmd_disable"), 0, _check_disabled("onepassword")),
}


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("command", list(COMMANDS))
def test_write_commands_handle_a_malformed_section(home, monkeypatch, capsys, command, shape):
    source, runner, expected_rc, check = COMMANDS[command]
    before = _config(shape, source)
    (home / "config.yaml").write_text(before, encoding="utf-8")

    rc = runner(monkeypatch)

    out = capsys.readouterr().out
    assert rc == expected_rc, out
    if check is None:  # remove: nothing to unmap, so the file is left exactly as it was
        assert "ZZ_OP_KEY is not mapped." in out
        assert (home / "config.yaml").read_text(encoding="utf-8") == before
    else:
        check(home, out)


def test_op_set_keeps_a_valid_sibling_section(home, capsys):
    from hermes_cli import onepassword_secrets_cli as op_cli

    (home / "config.yaml").write_text(
        "secrets:\n  bitwarden:\n    enabled: true\n    project_id: keep-me\n  onepassword: true\n",
        encoding="utf-8")

    assert op_cli.cmd_set(argparse.Namespace(env_var="ZZ_OP_KEY", reference=REF)) == 0

    assert _section(home, "bitwarden") == {"enabled": True, "project_id": "keep-me"}
    assert _section(home, "onepassword")["env"] == {"ZZ_OP_KEY": REF}


def test_section_cfg_for_write_replaces_only_non_mappings():
    from hermes_cli._secrets_common import section_cfg_for_write

    for cfg in ({"secrets": "bitwarden"}, {"secrets": ["bitwarden"]}, {"secrets": None}, {}):
        section = section_cfg_for_write(cfg, "bitwarden")
        assert section == {}
        assert cfg["secrets"]["bitwarden"] is section
    cfg = {"secrets": {"bitwarden": True, "onepassword": {"enabled": True}}}
    section_cfg_for_write(cfg, "bitwarden")["enabled"] = False
    assert cfg == {"secrets": {"bitwarden": {"enabled": False}, "onepassword": {"enabled": True}}}
    valid = {"enabled": True, "project_id": "p"}
    assert section_cfg_for_write({"secrets": {"bitwarden": valid}}, "bitwarden") is valid


FAKE_BWS = '#!/bin/sh\nif [ "$1" = "--version" ]; then echo "bws 1.0.0"; exit 0; fi\n' \
           'echo \'[{"key": "ZZ_BW_KEY", "value": "v"}]\'\n'
FAKE_OP = '#!/bin/sh\necho "2.30.0"\n'


@pytest.mark.parametrize("shape,argv,expected_rc", [
    ("scalar", ["secrets", "bitwarden", "disable"], 0),
    ("list", ["secrets", "bitwarden", "setup", "--access-token", "0.zz-token",
              "--server-url", "https://vault.bitwarden.eu", "--project-id", "proj-zz"], 0),
    ("int", ["secrets", "op", "set", "ZZ_OP_KEY", REF], 0),
    ("null", ["secrets", "op", "remove", "ZZ_OP_KEY"], 1),
    ("source-scalar", ["secrets", "op", "setup", "--token", "ops_zz"], 0),
    ("source-scalar", ["secrets", "op", "disable"], 0),
])
def test_real_cli_write_commands_on_a_malformed_section(tmp_path, shape, argv, expected_rc):
    source = "bitwarden" if argv[1] == "bitwarden" else "onepassword"
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(_config(shape, source), encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in (("bws", FAKE_BWS), ("op", FAKE_OP)):
        (bin_dir / name).write_text(script, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("HERMES_") and k not in TOKEN_ENVS}
    env.update(HOME=str(tmp_path), USERPROFILE=str(tmp_path), HERMES_HOME=str(home), COLUMNS="200",
               PATH=os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")]),
               PYTHONPATH=os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")])))

    proc = subprocess.run([sys.executable, "-m", "hermes_cli.main", *argv], env=env, cwd=tmp_path,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180, check=False)

    output = proc.stdout + proc.stderr
    assert proc.returncode == expected_rc, output
    assert "Traceback" not in output
    assert "has no attribute" not in output
    if argv[2] == "remove":  # nothing to unmap: the file is left exactly as it was
        assert "ZZ_OP_KEY is not mapped." in output
        assert (home / "config.yaml").read_text(encoding="utf-8") == _config(shape, source)
        return
    section = _section(home, source)
    if argv[2] == "setup":
        assert section["enabled"] is True
    elif argv[2] == "disable":
        assert not section.get("enabled")
    else:
        assert section["env"] == {"ZZ_OP_KEY": REF}
