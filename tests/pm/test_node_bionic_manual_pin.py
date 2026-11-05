"""#134029: `hermes pm update node` must not request a termux .deb the pool
has never shipped.

Nodejs.latest_versions handed nodejs.org's index to every target, including
linux-arm64-bionic — whose artifacts come from termux's apt pool, a separate
supplier that lags nodejs.org by weeks (nodejs.org served 26.10.0 while the
pool's newest nodejs deb was 26.4.0-1). The resolver picked 26.10.0 for every
target, and the pin step's hash_url() died with HTTP 404 on
packages.termux.dev, failing the whole node pin ("every pin failed; lockfile
untouched"). Like Python's bionic row, the termux nodejs .deb is a manual
pin: bionic now reports no auto-update source, so a resolve skips it and
_pin_artifacts retains the existing bionic row untouched.
"""

from pm import cli, packages
from pm.registry import get_package
from pm.store import ALL_TARGETS
from pm.update import resolve_package

TERMUX_ROW = {
    "url": "https://packages.termux.dev/apt/termux-main/pool/main/n/nodejs/"
    "nodejs_26.4.0-1_aarch64.deb",
    "sha256": "0" * 64,
}


def test_node_bionic_is_a_manual_pin(monkeypatch):
    monkeypatch.setattr(packages, "node_latest_versions", lambda: ["26.10.0"])
    assert get_package("node").latest_versions("linux-arm64-bionic") == []
    assert get_package("node").latest_versions("linux-x64") == ["26.10.0"]


def test_node_resolve_and_pin_never_touch_the_termux_pool(monkeypatch):
    monkeypatch.setattr(packages, "node_latest_versions", lambda: ["26.10.0", "26.7.0"])
    node = get_package("node")
    targets = [t for t in ALL_TARGETS if node.missing_reason(t) is None]
    assert "linux-arm64-bionic" in targets  # servable, just never auto-updated
    current = {"linux-arm64-bionic": dict(TERMUX_ROW)}

    decision = resolve_package(node, targets, "26.7.0", artifacts=dict(current))
    assert decision.version == "26.10.0"
    assert "linux-arm64-bionic" not in decision.per_target

    hashed = []
    monkeypatch.setattr(cli, "hash_url", lambda url: hashed.append(url) or "1" * 64)
    pinned = cli._pin_artifacts(node, decision, current)
    # The termux row survives as-is; no resolve ever hashed a .deb URL.
    assert pinned["linux-arm64-bionic"] == TERMUX_ROW
    assert not [url for url in hashed if "packages.termux.dev" in url]
    assert any("nodejs.org/dist" in url for url in hashed)
    assert any("unofficial-builds.nodejs.org" in url for url in hashed)
