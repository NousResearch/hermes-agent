"""`hermes update --plan --json` — machine-readable update plan.

With --plan --json, update prints the pre-update inventory as a JSON object
with install_method, profiles, runtimes, expected_sha, and expected_version.
"""

import json
import types

import pytest


@pytest.fixture()
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def test_plan_json_parses(tmp_home, monkeypatch, capsys):
    from hermes_cli.update_inventory import UpdatePlan

    fake_plan = UpdatePlan()
    fake_plan.install_method = "git"
    fake_plan.updatable_in_place = True
    fake_plan.expected_sha = "abc1234"
    fake_plan.profiles = ["default"]

    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory",
        lambda: fake_plan,
    )

    from hermes_cli.main import cmd_update

    args = types.SimpleNamespace(
        plan=True,
        json_output=True,
        check=False,
        gateway=False,
        branch=None,
        yes=False,
    )
    cmd_update(args)
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert doc["install_method"] == "git"
    assert doc["expected_sha"] == "abc1234"
    assert "default" in doc["profiles"]


def test_plan_json_has_runtimes_key(tmp_home, monkeypatch, capsys):
    from hermes_cli.update_inventory import UpdatePlan

    fake_plan = UpdatePlan()
    fake_plan.install_method = "git"
    fake_plan.runtimes = []

    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory",
        lambda: fake_plan,
    )

    from hermes_cli.main import cmd_update

    args = types.SimpleNamespace(
        plan=True,
        json_output=True,
        check=False,
        gateway=False,
        branch=None,
        yes=False,
    )
    cmd_update(args)
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert "runtimes" in doc
    assert isinstance(doc["runtimes"], list)


def test_plan_json_empty_plan(tmp_home, monkeypatch, capsys):
    from hermes_cli.update_inventory import UpdatePlan

    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory",
        lambda: UpdatePlan(),
    )

    from hermes_cli.main import cmd_update

    args = types.SimpleNamespace(
        plan=True,
        json_output=True,
        check=False,
        gateway=False,
        branch=None,
        yes=False,
    )
    cmd_update(args)
    out = capsys.readouterr().out
    doc = json.loads(out)
    assert "install_method" in doc
    assert "profiles" in doc
