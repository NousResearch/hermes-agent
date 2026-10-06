"""Stamps, receipts and doctor tell the truth about a PM install (PM lifecycle, failure class 3).

One real install, taken through a dependency-changing ``hermes update`` so the selected generation
is not the installer's. Then:

* every ``hermes`` the install ships agrees on what the install is. The selected generation's own
  console script (``<gen>/venv/bin/hermes``) is on PATH for every child a Hermes process spawns
  (``activate_dependencies`` prepends that ``bin``), so the agent's terminal, workers and scripts
  resolve ``hermes`` to it. It must report the checkout as the install and be able to check for
  updates; it reports the workspace copy instead (gated on #122425 and #122627);
* ``hermes doctor`` on that healthy install reports nothing wrong with the command installation
  (#124050 is the false positive class) and ``hermes pm status`` reports the update as a success;
* real drift is caught and healed: a user uninstalls fastapi (the ``web`` extra the dashboard
  imports) from the selected environment. ``hermes doctor`` must say so, and ``hermes pm repair``
  must bring the dashboard's import back;
* a dependency update that cannot resolve fails loudly, ``hermes pm status`` reports it as failed,
  and the previous generation stays selected and working.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from tests.e2e.core._pending_fixes import known_failure
from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.pm import _pm as P
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [
    pytest.mark.platforms("linux"),
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]


def _section(doctor_out: str, title: str) -> list[str]:
    """The lines of one ``◆ <title>`` section of ``hermes doctor`` output."""
    lines, inside = [], False
    for line in doctor_out.splitlines():
        if line.startswith("◆ "):
            inside = line[2:].strip() == title
            continue
        if inside and line.strip():
            lines.append(line)
    return lines


@pytest.fixture(scope="module")
def provider():
    with FakeLLMServer(default_text="fake reply for the pm stamps suite") as srv:
        yield srv


@pytest.fixture(scope="module")
def updated(tmp_path_factory, provider):
    root = tmp_path_factory.mktemp("pm-stamps")
    sb, origin = P.install_head(root)
    P.configure(sb, provider.base_url)
    installed_gen = P.selected_generation(sb)
    target = P.publish_dependency_release(origin, root, 1)
    up = P.update(sb)
    P.ok(up, "hermes update failed")
    assert P.selected_generation(sb) != installed_gen, "harness: update did not select a new generation"
    assert I.git("rev-parse", "HEAD", cwd=sb.checkout) == target
    return {"sb": sb, "target": target, "update": up, "origin": origin, "root": root}


def _venv_hermes(sb: I.Sandbox) -> str:
    return str(P.selected_generation(sb) / "venv" / "bin" / "hermes")


def test_managed_env_hermes_can_check_for_updates(updated):
    sb = updated["sb"]
    exe = _venv_hermes(sb)
    assert Path(exe).is_file(), f"harness: selected generation ships no hermes console script: {exe}"
    cp = sb.run([exe, "update", "--check"], timeout=300)
    assert cp.returncode == 0 and "Not a git repository" not in cp.stdout + cp.stderr, (
        "`hermes update --check` from the managed environment: " + (cp.stdout + cp.stderr).strip()[-400:]
        + "\n" + I.describe(cp))


def test_managed_env_hermes_reports_the_checkout_as_the_install(updated):
    sb = updated["sb"]
    cp = P.ok(sb.run([_venv_hermes(sb), "--version"], timeout=300))
    shown = re.search(r"Install directory: (.+)", cp.stdout)
    method = re.search(r"Install method: (.+)", cp.stdout)
    with known_failure(r"managed-environment hermes reports install .*/environments/[0-9a-f]+/workspace",
                       "gated on #122425: the workspace copy carries no install metadata"):
        assert shown and shown.group(1).strip() == str(sb.checkout) and method and method.group(1).strip() == "git", (
            f"managed-environment hermes reports install {shown and shown.group(1)} "
            f"(method {method and method.group(1)}), not the checkout {sb.checkout}:\n{cp.stdout}")


def test_doctor_on_a_healthy_pm_install_reports_no_command_installation_problem(updated):
    sb = updated["sb"]
    cp = sb.cli("doctor", timeout=300)
    assert I.TRACEBACK not in cp.stdout + cp.stderr, I.describe(cp)
    section = _section(cp.stdout, "Command Installation")
    assert section, "doctor printed no Command Installation section:\n" + I.describe(cp)
    bad = [line for line in section if line.lstrip().startswith(("⚠", "✗"))]
    assert not bad, f"hermes doctor reports a launcher problem on a healthy PM install: {bad}\n" + I.describe(cp)
    assert f"Hermes entry point exists ({sb.checkout / 'hermes'})" in cp.stdout, "\n".join(section)


def test_pm_status_receipt_reports_the_successful_update(updated):
    sb = updated["sb"]
    receipt = json.loads(P.ok(sb.cli("pm", "status")).stdout)
    code = receipt.get("exit_code", receipt.get("pm_exit_code"))
    assert code == 0 and receipt.get("outcome") in ("ok", "success"), (
        f"`hermes pm status` does not report the last (successful) update as a success: {receipt}")


@pytest.fixture(scope="module")
def drifted(updated):
    """The user removes fastapi (the ``web`` extra) from the selected environment by hand."""
    sb = updated["sb"]
    assert P.managed_imports(sb, "fastapi")["fastapi"] == "ok", "harness: fastapi not installed to begin with"
    uv = next((sb.hermes_home / "tools").glob("uv-*/uv"), None) or I.real_uv()
    P.ok(sb.run([str(uv), "pip", "uninstall", "--python", sb.python, "fastapi"]), "harness: uninstall failed")
    assert P.managed_imports(sb, "fastapi")["fastapi"] != "ok", "harness: fastapi still importable"
    doctor = sb.cli("doctor", timeout=300)
    repair = sb.cli("pm", "repair")
    return {"sb": sb, "doctor": doctor, "repair": repair}


def test_doctor_reports_web_extra_drift(drifted):
    cp = drifted["doctor"]
    assert I.TRACEBACK not in cp.stdout + cp.stderr, I.describe(cp)
    flagged = [line for line in cp.stdout.splitlines()
               if re.search(r"(?i)fastapi|dashboard|web extra|\bweb\b.*(missing|not installed)", line)
               and line.lstrip().startswith(("⚠", "✗"))]
    assert flagged, ("hermes doctor is silent about fastapi missing from the selected environment "
                     f"(rc={cp.returncode})\n" + I.describe(cp))


def test_pm_repair_heals_the_drift(drifted):
    sb, rp = drifted["sb"], drifted["repair"]
    assert rp.returncode == 0, "hermes pm repair failed on a drifted environment:\n" + P.diagnostics(sb, rp)
    imports = P.managed_imports(sb, "fastapi", "hermes_cli.web_server")
    assert set(imports.values()) == {"ok"}, (
        f"`hermes pm repair` exited 0 but the dashboard still cannot import: {imports}\n" + P.diagnostics(sb, rp))


def test_failed_dependency_update_is_reported_as_failed(updated):
    """A release whose uv.lock uv rejects: the update must fail loudly, the receipt must say so, and
    the install must keep running on the generation it had."""
    sb = updated["sb"]
    before = P.selected_generation(sb)
    origin, scratch = updated["origin"], updated["root"]
    lock = I.git("show", "main:uv.lock", cwd=origin) + '\n[[package]]\nname = "e2e-broken"\n'
    I.publish_commit(origin, scratch, "release: e2e broken lockfile", {"uv.lock": lock})
    up = P.update(sb)
    receipt = json.loads(P.ok(sb.cli("pm", "status")).stdout)
    assert up.returncode != 0, "`hermes update` exited 0 on a release whose uv.lock uv rejects\n" + P.diagnostics(sb, up)
    assert receipt.get("outcome") not in ("ok", "success"), (
        f"`hermes pm status` reports the failed update as a success: {receipt}\n" + P.diagnostics(sb, up))
    assert P.selected_generation(sb) == before, "a failed update switched the selected generation"
    imports = P.managed_imports(sb, "pydantic", "openai")
    assert set(imports.values()) == {"ok"}, f"the install no longer works after a failed update: {imports}"
    cp = sb.cli("--version")
    assert cp.returncode == 0 and I.TRACEBACK not in cp.stdout + cp.stderr, I.describe(cp)
