from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.config import get_managed_system, is_managed, recommended_update_command
from hermes_cli.main import cmd_update
from tools.skills_hub_official import OptionalSkillSource


def test_recommended_update_command_defaults_to_hermes_update(monkeypatch):
    monkeypatch.delenv("HERMES_MANAGED", raising=False)

    # Also short-circuit the .managed marker path — CI runners may have an
    # ambient ~/.hermes/.managed if a prior test left HERMES_HOME pointing
    # somewhere with that marker, which would make get_managed_update_command()
    # return "Update your Nix flake input ..." instead of falling through to
    # detect_install_method().
    with patch("hermes_cli.config.get_managed_update_command", return_value=None), \
         patch("hermes_cli.config.detect_install_method", return_value="git"):
        assert recommended_update_command() == "hermes update"


@pytest.mark.parametrize("false_value", ["false", "0", "no", "off", "FALSE"])
def test_get_managed_system_false_values(monkeypatch, false_value):
    """An explicit opt-out is not a package manager named "false" (#12864)."""
    monkeypatch.setenv("HERMES_MANAGED", false_value)

    assert get_managed_system() is None
    assert not is_managed()
    with patch("hermes_cli.config.detect_install_method", return_value="git"):
        assert recommended_update_command() == "hermes update"


def test_optional_skill_source_honors_env_override(monkeypatch, tmp_path):
    optional_dir = tmp_path / "optional-skills"
    optional_dir.mkdir()
    monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(optional_dir))

    source = OptionalSkillSource()

    assert source._optional_dir == optional_dir


# ---------------------------------------------------------------------------
# Refusals on a managed install exit non-zero
# ---------------------------------------------------------------------------


@pytest.fixture
def nixos_managed(tmp_path, monkeypatch):
    """A NixOS-module install: HERMES_MANAGED comes from the service environment."""
    import hermes_cli.image_provenance as ip

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED", "nixos")
    monkeypatch.setattr(ip, "IMAGE_PROVENANCE_PATH", tmp_path / "absent.json")
    # Without the managed state the calls below would open an editor or install a service.
    assert is_managed()
    return home


def _update_args(**overrides):
    return SimpleNamespace(**{"plan": False, "check": False, "list_venv_holders": False, "branch": None,
                              "gateway": False, **overrides})


@pytest.mark.parametrize("check", [False, True], ids=["update", "update --check"])
@pytest.mark.parametrize("stamp", ["git", "docker"])
def test_update_on_managed_install_is_refused_by_contract(nixos_managed, monkeypatch, capsys, check, stamp):
    """`hermes update && restart` must stop here: exit 2 and a `refused` receipt, like docker/nix/apt.

    A stale install stamp can name any method; the package manager still owns the install."""
    from hermes_cli import update_cmd
    from hermes_cli.update_receipt import read_latest_receipt

    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda *a, **k: stamp)
    monkeypatch.setattr(update_cmd, "_cmd_update_check", lambda **_: pytest.fail("--check ran on a managed install"))
    monkeypatch.setattr("hermes_cli.main._install_hangup_protection",
                        lambda **_: pytest.fail("the update ran on a managed install"))

    with pytest.raises(SystemExit) as excinfo:
        cmd_update(_update_args(check=check))

    assert excinfo.value.code == 2
    assert "managed by nixos" in capsys.readouterr().out
    receipt = read_latest_receipt()
    assert receipt["outcome"] == "refused"
    assert receipt["stop_reason"] == "managed"


@pytest.mark.parametrize("stamp", [None, "git"], ids=["no stamp", "stale git stamp"])
def test_update_plan_runs_on_managed_install(nixos_managed, monkeypatch, capsys, stamp):
    """--plan is read-only; on a managed install it reports "not updatable in place" instead of a refusal,
    and never points at the `hermes update` path the managed guard refuses."""
    import hermes_cli.update_inventory as ui

    if stamp:
        monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda *a, **k: stamp)

    def _install_shape_only():
        # Leaves out the running-gateway scan, which would read this machine's process table.
        plan = ui.UpdatePlan()
        ui._collect_install_shape(plan)
        return plan

    monkeypatch.setattr(ui, "collect_runtime_inventory", _install_shape_only)

    cmd_update(_update_args(plan=True))

    captured = capsys.readouterr()
    assert "Install: nixos" in captured.out
    assert "NOT updatable in place" in captured.out
    assert "Update via: hermes update" not in captured.out
    assert "Cannot update" not in captured.out + captured.err
    assert not (nixos_managed / "logs" / "update_receipts").exists()


_MANAGED_CLI_REFUSALS = {
    "config set": ("hermes_cli.config", "set_config_value", ("model.default", "foo")),
    "config unset": ("hermes_cli.config", "unset_config_value", ("model.default",)),
    "config edit": ("hermes_cli.config", "edit_config", ()),
    "setup": ("hermes_cli.setup", "run_setup_wizard", (SimpleNamespace(),)),
    "gateway setup": ("hermes_cli.gateway_setup_wizard", "gateway_setup", ()),
    "gateway install": ("hermes_cli.gateway", "_cmd_install", (SimpleNamespace(),)),
    "gateway uninstall": ("hermes_cli.gateway", "_cmd_uninstall", (SimpleNamespace(),)),
}


@pytest.mark.parametrize("command", sorted(_MANAGED_CLI_REFUSALS))
def test_managed_refusal_exits_1(nixos_managed, capsys, command):
    """main() turns a handler's None into exit 0, so a refusal that only returns reads as success."""
    import importlib

    module, attr, args = _MANAGED_CLI_REFUSALS[command]
    with pytest.raises(SystemExit) as excinfo:
        getattr(importlib.import_module(module), attr)(*args)

    assert excinfo.value.code == 1
    assert "managed by nixos" in capsys.readouterr().err
    assert not (nixos_managed / "config.yaml").exists()
