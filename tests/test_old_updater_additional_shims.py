"""Retired import names hand off old updater work without claiming success."""

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import os
import sys
from types import SimpleNamespace

import pytest

from tests.compat.old_updater_support import (
    fresh_child as fresh_child,
    no_external_work as no_external_work,
)


@pytest.mark.parametrize("command,profile", [
    ("", None), ("hermes -p ops gateway run", "ops"), ("hermes --profile=ops gateway run", "ops"),
])
def test_live_profile_parser_does_no_external_work(command, profile, no_external_work):
    from gateway.status import profile_flag_value

    # This classifier still protects current gateway identity checks.
    assert profile_flag_value(command) == profile


@pytest.mark.parametrize("refresh", [False, True])
def test_retired_code_identity_is_unknown(refresh, no_external_work, monkeypatch):
    from hermes_cli import version_info
    from hermes_cli.build_info import get_code_identity

    monkeypatch.setattr(version_info, "get_code_identity", no_external_work)
    identity = get_code_identity(refresh=refresh)
    assert identity == {"sha": None, "short_sha": None, "version": None, "source": "unknown"}
    identity["sha"] = "caller mutation"
    assert get_code_identity(refresh)["sha"] is None


def test_retired_constants_reload_handoffs_old_gateway_recovery(fresh_child, monkeypatch):
    import hermes_constants

    # Shipped get_python_path uses this fallback when its constants module is stale.
    monkeypatch.delattr(hermes_constants, "venv_python_path")
    before = dict(vars(hermes_constants))
    with fresh_child.exits():
        try:
            from hermes_constants import venv_python_path
        except ImportError:
            from hermes_cli.managed_uv import _reload_hermes_constants
            venv_python_path = _reload_hermes_constants().venv_python_path
        pytest.fail(f"old recovery continued with {venv_python_path}")
    assert vars(hermes_constants) == before


@pytest.mark.parametrize("kwargs", [{}, {"timeout": 120, "capture_output": False}])
def test_retired_pip_install_handoffs_before_reporting_success(kwargs, fresh_child):
    from hermes_cli.tools_config import _pip_install

    with fresh_child.exits():
        result = _pip_install(["--quiet", "honcho-ai"], **kwargs)
        pytest.fail(f"retired installer returned a result: {result}")


def test_retired_root_handoffs_before_inventing_portable_git_path(fresh_child):
    from hermes_cli.update_cmd import get_default_hermes_root

    with fresh_child.exits():
        get_default_hermes_root() / "git" / "mingw64" / "libexec" / "git-core" / "git.exe"


@pytest.mark.parametrize("prompt", [True, False])
def test_retired_ensure_reports_unavailable_without_installing(prompt, no_external_work):
    from tools.lazy_deps import ensure

    # Dependency-unavailable callers must not mistake a no-op for readiness.
    with pytest.raises(ImportError, match="relaunch"):
        ensure("memory.honcho", prompt=prompt)


def test_live_dingtalk_dependencies_use_pm_not_retired_installer(monkeypatch):
    from plugins.platforms.dingtalk import adapter
    from pm import extras

    requested = []

    def unavailable(extra):
        requested.append(extra)
        raise ImportError("dependency unavailable")

    monkeypatch.setattr(adapter, "DINGTALK_STREAM_AVAILABLE", False)
    monkeypatch.setattr(extras, "ensure_import", unavailable)
    assert adapter.ensure_dingtalk_deps() is False
    assert requested == ["dingtalk"]


@pytest.mark.parametrize("specs", [[], ["honcho-ai"]])
def test_retired_install_specs_handoffs_before_reporting_success(specs, fresh_child):
    from tools.lazy_deps import install_specs

    before = list(specs)
    with fresh_child.exits():
        result = install_specs(specs, timeout=120)
        pytest.fail(f"retired installer returned a result: {result}")
    assert specs == before


@pytest.mark.parametrize("handled", [False, True], ids=["unacknowledged", "child-completed"])
def test_historical_payload_survives_bridge_and_cleanup_requires_ack(handled, fresh_child, monkeypatch):
    from hermes_cli import update_receipt
    from hermes_cli.managed_uv import ensure_uv

    @dataclass
    class HistoricalPlan:
        profiles: list[str]
        snapshots: dict[str, str]

    # These names belong to real old updater frames, not parameters added to the
    # shim. Preserve opaque data and arguments, including spaces and Unicode.
    had_desktop_app_before_update = True
    pre_update_snapshot_id = "snapshot before pull"
    pre_update_version = "old-version"
    gateway_mode = True
    assume_yes = False  # Old frame state takes precedence over argv's --yes.
    _pre_update_plan = HistoricalPlan(["ops team", "日本"], {"ops team": "snapshot before pull"})
    _windows_gateway_resume = {"resume_needed": True, "profiles": {"ops team": "old-pid"}}
    receipt = SimpleNamespace(data={"update_id": "original-id", "steps": [{"name": "pull", "ok": True}]})
    slot = ContextVar("test_historical_receipt", default=receipt)
    monkeypatch.setattr(update_receipt, "_current", slot)
    argv = ["hermes", "--profile", "ops team", "update", "--gateway", "--yes"]
    monkeypatch.setattr(sys, "argv", argv)
    expected = deepcopy({
        "desktop": had_desktop_app_before_update,
        "pre_update_snapshot_id": pre_update_snapshot_id,
        "pre_update_version": pre_update_version,
        "gateway_mode": gateway_mode,
        "assume_yes": assume_yes,
        "windows_resume": _windows_gateway_resume,
        "plan": {"profiles": _pre_update_plan.profiles, "snapshots": _pre_update_plan.snapshots},
        "receipt": receipt.data,
        "argv": argv,
    })
    fresh_child.result = {"resume_handled": handled, "receipt_handled": handled}
    with fresh_child.exits():
        ensure_uv()
    request = fresh_child.requests[0]
    assert {key: request[key] for key in expected} == expected
    assert _pre_update_plan.profiles == expected["plan"]["profiles"]
    assert _pre_update_plan.snapshots == expected["plan"]["snapshots"]
    assert receipt.data == expected["receipt"]
    assert sys.argv == expected["argv"]
    assert slot.get() is (None if handled else receipt)
    assert _windows_gateway_resume == {**expected["windows_resume"], "resume_needed": not handled}


def test_retired_subprocess_run_handoffs_instead_of_running_powershell(fresh_child):
    from hermes_cli import _subprocess_compat

    with fresh_child.exits():
        _subprocess_compat.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-c", "irm https://astral.sh/uv/install.ps1 | iex"],
            env=dict(os.environ), check=True, capture_output=True,
        )
