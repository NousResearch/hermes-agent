"""Configured features survive every dependency rebuild (PM lifecycle, failure class 1).

A PM install's dependency environment is rebuilt as a NEW generation whenever its inputs change:
a release that moves ``uv.lock`` (``hermes update``), ``hermes pm repair``, and the first
``hermes update`` of a main-era install (the legacy ``venv/`` is replaced by a PM generation). The
user-visible contract is that whatever feature the user had working before the rebuild still
imports in the generation selected after it:

* an extra the user installed with the documented command (``hermes pm install --extra telegram``);
* the MCP client (an HTTP MCP server in config.yaml must still connect: ``hermes mcp test``
  against a real local streamable-HTTP server);
* a gateway platform enabled in config.yaml (Discord): the update installs the platform's extra
  instead of only warning that the SDK is missing (#124228);
* extras a main-era venv carried (``[all]`` plus ``[messaging]``) across the
  legacy-venv migration to the first PM generation.

Every check imports in the SELECTED generation's own interpreter (facts.json), booted as the
launcher boots it, or drives the real CLI.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from tests.e2e.core.mcp_plugins._helpers import HttpMcpServer
from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.pm import _pm as P
from tests.e2e.core.upgrade.test_upgrade_path import _RETRY_PREFIX, _refs, make_leg
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]

MCP_NAME = "e2e-http"
MANAGED = ("telegram", "discord", "mcp.client.streamable_http")


def _mcp_test(sb: I.Sandbox, server: HttpMcpServer) -> str:
    """``hermes mcp test`` against the live server; returns a failure description or ''."""
    before = server.log.read_text(encoding="utf-8").count('"initialize"') if server.log.exists() else 0
    cp = sb.cli("mcp", "test", MCP_NAME, timeout=180)
    after = server.log.read_text(encoding="utf-8").count('"initialize"') if server.log.exists() else 0
    if cp.returncode != 0 or "Connected" not in cp.stdout or after <= before:
        return f"`hermes mcp test {MCP_NAME}` did not connect (server saw {after - before} initialize)\n{I.describe(cp)}"
    return ""


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the pm features suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory, provider):
    """One install taken through a dependency-changing ``hermes update`` and a ``pm repair``."""
    root = tmp_path_factory.mktemp("pm-features")
    server = HttpMcpServer(root, "pm-features", MCPE2E_CANARY="CANARY-pm-features").start()
    try:
        sb, origin = P.install_head(root)
        P.configure(
            sb, provider.base_url,
            extra=("platforms:\n  discord:\n    enabled: true\n"
                   f"mcp_servers:\n  {MCP_NAME}:\n    url: {server.url}\n    connect_timeout: 30\n    timeout: 30\n"),
            env_extra="DISCORD_BOT_TOKEN=fake-e2e-discord-token\n")
        P.ok(sb.cli("pm", "install", "--extra", "telegram"), "the documented extra install failed")
        before = {"generation": P.selected_generation(sb), "imports": P.managed_imports(sb, *MANAGED),
                  "mcp": _mcp_test(sb, server)}
        assert before["imports"]["telegram"] == "ok" and before["mcp"] == "", (
            f"harness: `hermes pm install --extra telegram` exited 0 but left no SDK / MCP broken: {before}")
        P.publish_dependency_release(origin, root, 1)
        up = P.update(sb)
        after_update = {"generation": P.selected_generation(sb), "imports": P.managed_imports(sb, *MANAGED),
                        "mcp": _mcp_test(sb, server)}
        repair = sb.cli("pm", "repair")
        after_repair = {"generation": P.selected_generation(sb), "imports": P.managed_imports(sb, *MANAGED),
                        "mcp": _mcp_test(sb, server)}
        yield {"sb": sb, "before": before, "update": up, "after_update": after_update,
               "repair": repair, "after_repair": after_repair}
    finally:
        server.stop()


def test_update_rebuild_keeps_installed_extra_and_mcp(rebuilt):
    sb, up, after = rebuilt["sb"], rebuilt["update"], rebuilt["after_update"]
    assert up.returncode == 0, "hermes update failed:\n" + P.diagnostics(sb, up)
    assert after["generation"] != rebuilt["before"]["generation"], (
        "harness: the dependency release did not make the update build a new generation\n" + P.diagnostics(sb, up))
    assert after["imports"]["telegram"] == "ok", (
        f"the extra installed with `hermes pm install --extra telegram` is gone after `hermes update`: "
        f"{after['imports']}\n" + P.diagnostics(sb, up))
    assert after["imports"]["mcp.client.streamable_http"] == "ok", after["imports"]
    assert after["mcp"] == "", "HTTP MCP server no longer connects after `hermes update`:\n" + after["mcp"]


def test_repair_keeps_installed_extra_and_mcp(rebuilt):
    sb, rp, after = rebuilt["sb"], rebuilt["repair"], rebuilt["after_repair"]
    assert rp.returncode == 0, "hermes pm repair failed on a healthy install:\n" + P.diagnostics(sb, rp)
    assert after["generation"] != rebuilt["after_update"]["generation"], (
        "`hermes pm repair` reported success but did not rebuild the environment\n" + P.diagnostics(sb, rp))
    assert after["imports"]["telegram"] == "ok", (
        f"the extra installed with `hermes pm install --extra telegram` is gone after `hermes pm repair`: "
        f"{after['imports']}\n" + P.diagnostics(sb, rp))
    assert after["mcp"] == "", "HTTP MCP server no longer connects after `hermes pm repair`:\n" + after["mcp"]


def test_configured_gateway_platform_has_its_sdk_after_update(rebuilt):
    sb, up, after = rebuilt["sb"], rebuilt["update"], rebuilt["after_update"]
    assert up.returncode == 0, P.diagnostics(sb, up)
    assert after["imports"]["discord"] == "ok", (
        f"configured discord platform has no SDK after `hermes update`: {after['imports']['discord']}\n"
        + P.diagnostics(sb, up))


# ---------------------------------------------------------------------------
# Legacy venv -> first PM generation
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def migrated(tmp_path_factory, provider):
    """Release N-1 installed main-era style (``venv/`` from N-1's own uv.lock with ``[all]``) plus
    the ``[messaging]`` extra (Telegram, Discord, Slack SDKs) the main-era docs installed, then
    updated to HEAD."""
    if not _refs().base_tag:
        pytest.skip("no release tag reachable before HEAD (fetch tags)")
    leg = make_leg(tmp_path_factory.mktemp("pm-legacy") / "leg", None)
    no_cfg = leg.root / "uv-config"
    uv_env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "UV_NO_CONFIG", "UV_CONFIG_FILE")}
    uv_env.update(UV_PROJECT_ENVIRONMENT=str(leg.install / "venv"), XDG_CONFIG_HOME=str(no_cfg),
                  XDG_CONFIG_DIRS=str(no_cfg))
    uv = I.real_uv()
    assert uv is not None
    cp = subprocess.run([uv, "sync", "-q", "--locked", "--extra", "all", "--extra", "messaging",
                         "--python", str(leg.install / "venv" / "bin" / "python")],
                        cwd=str(leg.install), env=uv_env, capture_output=True, text=True, timeout=1800)
    assert cp.returncode == 0, f"harness: messaging install into the N-1 venv failed:\n{cp.stderr[-4000:]}"
    with (leg.install / "pyproject.toml").open("rb") as fh:
        assert "messaging" in tomllib.load(fh)["project"]["optional-dependencies"], "harness: N-1 has no messaging extra"
    (leg.hermes_home / "config.yaml").write_text(I.provider_config(provider.base_url, None), encoding="utf-8")
    (leg.hermes_home / ".env").write_text(f"OPENAI_API_KEY={I.FAKE_KEY}\n", encoding="utf-8")
    probe = "import telegram, discord, mcp.client.streamable_http; print('ok')"
    pre = leg.run("-c", probe, argv0=leg.python)
    I.git("update-ref", "refs/heads/main", _refs().head, cwd=leg.origin)
    up = leg.run(*_RETRY_PREFIX[1:], leg.hermes, "update", "--yes", "--branch", "main",
                 argv0=_RETRY_PREFIX[0], timeout=P.UPDATE_TIMEOUT)
    return {"leg": leg, "pre": pre, "update": up}


def test_legacy_venv_features_carry_into_the_first_pm_generation(migrated):
    leg, up = migrated["leg"], migrated["update"]
    assert migrated["pre"].returncode == 0, "harness: N-1 venv cannot import its features:\n" + H.describe(migrated["pre"])
    assert up.returncode == 0, "hermes update from a main-era venv failed:\n" + H.describe(up)
    assert I.git("rev-parse", "HEAD", cwd=leg.install) == _refs().head, H.describe(up)
    selected = Path(leg.python)
    assert "installs" in selected.parts, f"update left the install on the legacy venv: {selected}\n{H.describe(up)}"
    code = ("import importlib, json\nout = {}\nfor m in ('telegram', 'discord', 'mcp.client.streamable_http'):\n"
            "    try:\n        importlib.import_module(m); out[m] = 'ok'\n"
            "    except BaseException as e:\n        out[m] = f'{type(e).__name__}: {e}'\nprint(json.dumps(out))\n")
    cp = leg.run("-c", code, argv0=str(selected))
    assert cp.returncode == 0, H.describe(cp)
    lost = [m for m in ("telegram", "discord") if f'"{m}": "ok"' not in cp.stdout]
    assert not lost, (
        f"messaging SDKs the main-era venv carried are gone from the first PM generation: {lost} {cp.stdout}\n"
        + H.describe(up))
    assert '"mcp.client.streamable_http": "ok"' in cp.stdout, cp.stdout
