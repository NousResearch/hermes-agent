"""Package mirrors: with a pip index mirror and an npm registry configured, nothing contacts the
public indexes, and the locked install paths accept the mirror.

This is the network CN/IR/corporate users actually have: ``pip.conf`` ``index-url`` and
``~/.npmrc`` ``registry`` point at an internal mirror, and pypi.org, files.pythonhosted.org and
registry.npmjs.org are unreachable. Every command runs in a private network namespace; the mirror
is a plain-HTTP server bridged onto the namespace's loopback (what users configure), the proxy
is the only other egress and refuses (and logs) every public index. The mirror holds a small set
pre-fetched at fixture time and verified against the pins: the PM runtime's locked wheels
(``pm/uv.lock``) and the npm tarball PM provisions.

Classes: an update that ships a PM runtime change restages it with ``uv sync --frozen`` from
the committed lock while pip's mirror is bridged into uv (#124418, #123943, #122112), and PM's
npm-hosted tool downloads (#123132: the configured npm registry serves the pinned tarball).
"""

from __future__ import annotations

import json
import shutil

import pytest

from tests.e2e.core.upgrade import _helpers as H
from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade.network import _netedge as N
from tests.e2e.core.upgrade.network import _seed as S

pytestmark = [
    pytest.mark.platforms("linux"),
    # Every `hermes update` here targets a throwaway sandboxed install, never the real checkout.
    pytest.mark.live_system_guard_bypass,
    pytest.mark.skipif(H.sandbox_required_reason() is not None, reason=str(H.sandbox_required_reason())),
    pytest.mark.skipif(N.netns_required_reason() is not None, reason=str(N.netns_required_reason())),
    pytest.mark.skipif(shutil.which("git") is None, reason="git required"),
    pytest.mark.skipif(I.real_uv() is None, reason="uv required"),
]


@pytest.fixture(scope="module")
def inst(tmp_path_factory):
    return S.seed_install(tmp_path_factory.mktemp("net-mirrors"))


def _public_index_hits(edge: S.Edge) -> list[str]:
    return [str(h) for h in edge.proxy.hits if h.host in S.PUBLIC_INDEXES]


def test_pip_mirror_update_restages_pm_runtime_through_the_mirror(inst):
    """The new upstream commit changes the PM runtime's inputs (``pm/pyproject.toml``), so the
    update must restage that runtime from ``pm/uv.lock``. pip.conf names the mirror; the public
    index is unreachable. The update must land, restage through the mirror, and never ask pypi."""
    pm = inst.sb.checkout / "pm"
    wheels: dict[str, list] = {}
    for name, url, sha in S.lock_wheels(pm / "uv.lock"):
        path = S.prefetch(url, sha, inst.root / "pypi-mirror" / url.rsplit("/", 1)[1])
        wheels.setdefault(name, []).append((path, sha))
    mirror = N.HttpSite(S.simple_index_app(wheels))
    pip_conf = inst.sb.home / ".config" / "pip" / "pip.conf"
    pip_conf.parent.mkdir(parents=True, exist_ok=True)
    pip_conf.write_text(f"[global]\nindex-url = {mirror.url}/simple/\n", encoding="utf-8")
    pyproject = (pm / "pyproject.toml").read_text(encoding="utf-8")
    new = inst.publish("pm-runtime-input", {"pm/pyproject.toml": pyproject + "\n# e2e: PM runtime input change\n"})
    before = inst.state()
    edge = inst.edge(sites=[mirror])
    try:
        r = inst.hermes("update", "--yes", edge=edge, timeout=900)
    finally:
        edge.close()
        pip_conf.unlink()
    public = _public_index_hits(edge)
    # #124418/#125657: the stage is --frozen, so the bridged mirror can no
    # longer veto the shipped pm/uv.lock — the update must now succeed.
    assert r.rc == 0 and inst.head() == new, "update with a pip mirror configured failed\n" + r.report(inst)
    after = inst.state()
    assert after["pm_runtime"] != before["pm_runtime"], "the PM runtime was not restaged for its new inputs\n" + r.report(inst)
    assert not public, f"a public index was contacted despite the mirror: {public}\n" + r.report(inst)


def test_npm_registry_mirror_serves_pm_npm_download(inst):
    """``~/.npmrc`` names the mirror; the npm tool's store entry is missing, so
    `hermes pm install npm` has to fetch its pinned tarball: from the mirror, never from
    registry.npmjs.org."""
    tool = S.installed_tool(inst, "npm")
    version = tool["version"]
    tarball = f"npm-{version}.tgz"
    blob = S.prefetch(f"https://registry.npmjs.org/npm/-/{tarball}", tool["artifacts"][0],
                      inst.root / "npm-mirror" / tarball)
    packument = json.dumps({"name": "npm", "dist-tags": {"latest": version}, "versions": {version: {
        "name": "npm", "version": version, "dist": {"tarball": f"MIRROR/npm/-/{tarball}"}}}})
    routes: dict = {f"/npm/-/{tarball}": blob.read_bytes()}
    mirror = N.HttpSite(N.static_app(routes))
    routes["/npm"] = packument.replace("MIRROR", mirror.url).encode()
    npmrc = inst.sb.home / ".npmrc"
    npmrc.write_text(f"registry={mirror.url}/\n", encoding="utf-8")
    try:
        with S.tool_missing(inst, "npm") as tool:
            edge = inst.edge(sites=[mirror])
            try:
                r = inst.hermes("pm", "install", "npm", edge=edge, timeout=600)
            finally:
                edge.close()
            public = _public_index_hits(edge)
            assert not public and r.rc == 0, (
                f"npm provisioning with a registry mirror failed or reached the public registry {public}\n"
                + r.report(inst))
            assert (inst.sb.hermes_home / "tools" / tool["entry"]).is_dir(), "npm was not re-provisioned\n" + r.report(inst)
            assert any(h.path.endswith(tarball) and h.status == 200 for h in mirror.hits), (
                "the tarball was not served by the mirror\n" + r.report(inst))
    finally:
        npmrc.unlink()
