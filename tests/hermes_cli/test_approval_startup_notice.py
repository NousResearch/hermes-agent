"""Startup provenance must describe the approval gate actually in force (#106504)."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_REPO = Path(__file__).resolve().parents[2]
_PROBE = r'''
import json
import os
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace

case = json.loads(sys.argv[1])
home = Path(os.environ["HERMES_HOME"])
from hermes_cli.env_loader import load_hermes_dotenv
load_hermes_dotenv(hermes_home=home, project_env=case["project"], load_external_secrets=False)

if case.get("flag"):
    # The version path isolates the setter boundary, not the final chat-launch precedence.
    from hermes_cli.main import _prepare_agent_startup
    _prepare_agent_startup(SimpleNamespace(command="chat" if case.get("chat_imports") else "version", yolo=True))
if case.get("chat_imports"):
    # These real startup imports may reload dotenv before approval freezes its process policy.
    import cli
    import run_agent

from tools import approval
from hermes_cli.banner import _banner_left_lines, build_welcome_banner
from rich.console import Console
from rich.text import Text

def snapshot():
    lines = _banner_left_lines("test/model", "test-cwd", None, None, "test",
                               accent="white", dim="white")
    rendered = []
    for width in (60, 80, 120):
        output = StringIO()
        build_welcome_banner(
            console=Console(file=output, width=width, color_system=None),
            model="test/model", cwd="test-cwd", tools=[], enabled_toolsets=["terminal"],
            get_toolset_for_tool=lambda name: None, provider="test",
            availability={}, skills_by_category={},
        )
        rendered.append(output.getvalue())
    return {
        "active": approval.is_approval_bypass_active_for_session("startup-probe"),
        "frozen": approval._YOLO_MODE_FROZEN,
        "text": "\n".join(line.plain if isinstance(line, Text) else Text.from_markup(line).plain for line in lines),
        "rendered": rendered,
    }

seen = [snapshot()]
for next_home in case.get("reload_homes", []):
    os.environ["HERMES_HOME"] = next_home
    load_hermes_dotenv(hermes_home=next_home, project_env=case["project"], load_external_secrets=False)
    seen.append(snapshot())
print("STARTUP_PROBE=" + json.dumps(seen))
'''


def _probe(tmp_path, *, shell=None, profile=None, project=None, mode="manual",
           managed_mode=None, managed_env=None, flag=False, chat_imports=False, reload_values=()):
    home = tmp_path / "profile[local]"
    home.mkdir()
    (home / "config.yaml").write_text(f"approvals:\n  mode: {mode}\n", encoding="utf-8")
    if profile is not None:
        (home / ".env").write_text(profile, encoding="utf-8")
    project_env = tmp_path / "project.env"
    if project is not None:
        project_env.write_text(project, encoding="utf-8")
    managed = tmp_path / "managed"
    managed.mkdir()
    if managed_mode is not None:
        (managed / "config.yaml").write_text(
            f"approvals:\n  mode: {managed_mode}\n", encoding="utf-8")
    if managed_env is not None:
        (managed / ".env").write_text(managed_env, encoding="utf-8")

    # Only location/runtime variables cross into the child: never developer credentials/config.
    env = {key: os.environ[key] for key in (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP", "LANG", "TZ",
    ) if key in os.environ}
    env.update({"HOME": str(tmp_path), "USERPROFILE": str(tmp_path),
                "HERMES_HOME": str(home), "HERMES_MANAGED_DIR": str(managed),
                "PYTHONUTF8": "1", "HERMES_MULTIPLEX_PROFILES": "0"})
    if shell is not None:
        env["HERMES_YOLO_MODE"] = shell
    reload_homes = []
    if reload_values:
        sibling = tmp_path / "profile-b"
        sibling.mkdir()
        (sibling / "config.yaml").write_text("approvals:\n  mode: manual\n", encoding="utf-8")
        (sibling / ".env").write_text(f"HERMES_YOLO_MODE={reload_values[0]}\n", encoding="utf-8")
        reload_homes = [str(sibling), str(home), str(home)]
    case = {"project": str(project_env), "flag": flag, "chat_imports": chat_imports,
            "reload_homes": reload_homes}
    result = subprocess.run([sys.executable, "-c", _PROBE, json.dumps(case)],
                            cwd=_REPO, env=env, text=True, encoding="utf-8",
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = next(line.removeprefix("STARTUP_PROBE=") for line in result.stdout.splitlines()
                   if line.startswith("STARTUP_PROBE="))
    return json.loads(payload), {"profile": home, "project": project_env, "managed": managed}


@pytest.mark.parametrize("settings,active,source", [
    pytest.param({"shell": "true"}, True, "process", id="shell"),
    pytest.param({"shell": "false"}, False, None, id="false-is-not-bypass"),
    pytest.param({"shell": "0"}, False, None, id="zero-is-not-bypass"),
    pytest.param({"shell": "0", "profile": "HERMES_YOLO_MODE=1\n"}, True, "profile", id="profile-over-shell"),
    pytest.param({"shell": "1", "profile": "HERMES_YOLO_MODE=1\n", "project": "HERMES_YOLO_MODE=1\n"}, True, "profile", id="same-value-still-has-one-writer"),
    pytest.param({"shell": "1", "profile": "HERMES_YOLO_MODE=0\n"}, False, None, id="profile-disables-shell"),
    pytest.param({"shell": "0", "project": "HERMES_YOLO_MODE=1\n"}, True, "project", id="project-without-profile"),
    pytest.param({"profile": "HERMES_YOLO_MODE=1\n", "project": "HERMES_YOLO_MODE=0\n"}, True, "profile", id="profile-over-project"),
    pytest.param({"profile": "UNRELATED=value\n", "project": "HERMES_YOLO_MODE=1\n"}, True, "project", id="project-fills-gap"),
    pytest.param({"shell": "false", "mode": "'off'"}, True, "config", id="config-off-despite-false-env"),
    pytest.param({"mode": "off"}, True, "config", id="yaml-off-boolean"),
    pytest.param({"mode": "manual", "managed_mode": "off"}, True, "managed-config", id="managed-config-off"),
    pytest.param({"profile": "HERMES_YOLO_MODE=0\n", "managed_env": "HERMES_YOLO_MODE=1\n"}, True, "managed-env", id="managed-env-wins"),
    pytest.param({"profile": "HERMES_YOLO_MODE=1\n", "managed_env": "HERMES_YOLO_MODE=1\n"}, True, "managed-env", id="managed-same-value-wins"),
    pytest.param({"profile": "HERMES_YOLO_MODE=1\n", "mode": "off"}, True, "profile+config", id="both-independent-bypasses"),
    pytest.param({"flag": True}, True, "flag", id="yolo-setter-boundary"),
    pytest.param({"flag": True, "profile": "HERMES_YOLO_MODE=1\n"}, True, "flag", id="yolo-setter-replaces-dotenv-writer"),
    pytest.param({"flag": True, "chat_imports": True}, True, "flag", id="chat-imports-with-yolo"),
    pytest.param({"flag": True, "chat_imports": True, "profile": "HERMES_YOLO_MODE=1\n"}, True, "profile", id="chat-imports-reload-dotenv-writer"),
    pytest.param({"flag": True, "chat_imports": True, "profile": "HERMES_YOLO_MODE=0\n"}, False, None, id="chat-imports-preserve-existing-false-policy"),
    pytest.param({}, False, None, id="normal-startup"),
])
def test_notice_identifies_the_source_of_the_effective_bypass(tmp_path, settings, active, source):
    seen, paths = _probe(tmp_path, **settings)
    state = seen[0]
    assert state["active"] is active
    notice = state["text"]
    assert ("bypass" in notice.lower()) is active, notice
    for rendered in state["rendered"]:
        # Joining folded rows must retain the full source, including literal Rich brackets.
        # The right column finishes above the notice, so it does not split the source's rows.
        compact = "".join(char for char in rendered if not char.isspace() and char not in "│╭╮╰╯─")
        assert ("bypass" in compact.lower()) is active, rendered
        for line in notice.splitlines():
            if "bypass" in line.lower():
                assert "".join(line.split()) in compact, rendered
    if source is None:
        return
    file_sources = {"profile": paths["profile"] / ".env", "project": paths["project"],
                    "managed-env": paths["managed"] / ".env",
                    "config": paths["profile"] / "config.yaml",
                    "managed-config": paths["managed"] / "config.yaml"}
    expected = set(source.split("+"))
    for name, path in file_sources.items():
        assert (str(path) in notice) is (name in expected), notice
    for contributor in source.split("+"):
        if contributor == "flag":
            assert "--yolo" in notice
        elif contributor == "process":
            assert "HERMES_YOLO_MODE" in notice and "environment" in notice.lower()
        elif contributor in {"config", "managed-config"}:
            assert "approvals.mode" in notice and "off" in notice
            owner = paths["managed"] if contributor == "managed-config" else paths["profile"]
            assert str(owner / "config.yaml") in notice
        else:
            assert "HERMES_YOLO_MODE" in notice
            owner = paths["profile"] / ".env" if contributor == "profile" else paths["project"]
            if contributor == "managed-env":
                owner = paths["managed"] / ".env"
            assert str(owner) in notice


@pytest.mark.parametrize("initial,later,flag", [("1", "0", False), ("0", "1", False), ("1", "0", True)])
def test_dotenv_reloads_do_not_relabel_the_frozen_process_bypass(tmp_path, initial, later, flag):
    seen, paths = _probe(tmp_path, profile=f"HERMES_YOLO_MODE={initial}\n", flag=flag, reload_values=(later,))
    active = initial == "1" or flag
    for state in seen:
        assert state["active"] is active
        assert state["frozen"] is active
        assert ("bypass" in state["text"].lower()) is active, state["text"]
        if active:
            assert ("--yolo" if flag else str(paths["profile"] / ".env")) in state["text"]
            assert "profile-b" not in state["text"]
