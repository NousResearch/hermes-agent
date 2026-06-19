"""Tests for the ai-employee-org Hermes plugin."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "ai-employee-org"


def load_plugin():
    package_name = "ai_employee_org_test_plugin"
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        package_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


class _FakeContext:
    def __init__(self) -> None:
        self.cli_commands = {}
        self.commands = {}

    def register_cli_command(self, name, **kwargs):
        self.cli_commands[name] = kwargs

    def register_command(self, name, **kwargs):
        self.commands[name] = kwargs


def test_register_exposes_cli_and_slash():
    module = load_plugin()
    ctx = _FakeContext()
    module.register(ctx)
    assert "ai-employees" in ctx.cli_commands
    assert "ai-employees" in ctx.commands


def test_plugin_dir_and_stack_file_exist():
    module = load_plugin()
    assert (module.core.plugin_dir() / "plugin.yaml").is_file()
    assert module.core.stack_file().is_file()
    assert (module.core.skill_source_dir() / "SKILL.md").is_file()


def test_list_installers_has_five_scripts():
    module = load_plugin()
    installers = module.cron_install.list_installers()
    assert len(installers) == 5
    assert "install-job-seeker-cron.py" in installers


def test_status_shape(tmp_path, monkeypatch):
    module = load_plugin()
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(module.core, "get_hermes_home", lambda: home)
    monkeypatch.setattr(module.core, "display_hermes_home", lambda: str(home))
    monkeypatch.setattr(module.core, "_profile_exists", lambda _name: False)

    payload = module.core.status()
    assert payload["plugin"] == "ai-employee-org"
    assert "profiles" in payload
    assert set(payload["profiles"]) == {
        "secretary",
        "job-recruiter",
        "job-seeker",
        "self-improver",
        "delivery-worker",
    }


def test_install_all_dry_run():
    module = load_plugin()
    payload = module.core.install_all(dry_run=True)
    assert payload["success"] is True
    assert payload["dry_run"] is True
    assert payload["steps"]["cron"] == "would_install"


def test_job_seeker_build_job_accepts_telegram_origin():
    script = PLUGIN_DIR / "scripts" / "install-job-seeker-cron.py"
    spec = importlib.util.spec_from_file_location("job_seeker_cron", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    job = mod.build_job(
        profile="job-seeker",
        telegram_origin={"platform": "telegram", "chat_id": "999", "chat_name": "T", "thread_id": None},
    )
    assert job["origin"]["chat_id"] == "999"
    assert job["no_agent"] is True
    assert job["script"]


def test_cron_install_dry_run(tmp_path, monkeypatch):
    module = load_plugin()
    jobs_path = tmp_path / "cron" / "jobs.json"
    jobs_path.parent.mkdir(parents=True)
    jobs_path.write_text('{"jobs": []}', encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = module.cron_install.install_all_crons(
        telegram_chat_id="12345",
        dry_run=True,
    )
    assert result["ok"] is True
    for name, detail in result["jobs"].items():
        assert detail["ok"] is True, f"{name}: {detail}"
    seeker = result["jobs"]["install-job-seeker-cron.py"]
    sample = json.loads(seeker["stdout"])
    assert sample.get("origin", {}).get("chat_id") == "12345"
