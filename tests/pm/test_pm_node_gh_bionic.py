"""Regression for #159592: node termux revision probe + gh bionic gap."""

from __future__ import annotations

import pytest

from pm.packages import Gh, Nodejs, _termux_nodejs_deb_url
from pm.registry import get_package


def test_gh_declares_bionic_gap():
    gh = get_package("gh")
    assert isinstance(gh, Gh)
    reason = gh.missing_reason("linux-arm64-bionic")
    assert reason is not None
    assert "Android" in reason or "bionic" in reason.lower() or "publish" in reason.lower()


def test_gh_fetch_url_still_rejects_three_segment_bionic_name():
    """Defense in depth: fetch_url itself still cannot unpack linux-arm64-bionic.
    The gap is what keeps pin from calling it (#159592)."""
    gh = get_package("gh")
    with pytest.raises(ValueError, match="too many values to unpack"):
        gh.fetch_url("2.105.0", "linux-arm64-bionic")


def test_termux_nodejs_prefers_bare_when_present(monkeypatch):
    seen = []

    def fake_exists(url: str) -> bool:
        seen.append(url)
        return url.endswith("nodejs_26.13.1_aarch64.deb")

    monkeypatch.setattr("pm.packages._http_url_exists", fake_exists)
    url = _termux_nodejs_deb_url("26.13.1")
    assert url.endswith("nodejs_26.13.1_aarch64.deb")
    assert not url.endswith("-1_aarch64.deb")
    assert any(u.endswith("nodejs_26.13.1_aarch64.deb") for u in seen)


def test_termux_nodejs_falls_back_to_rev1_when_bare_missing(monkeypatch):
    def fake_exists(url: str) -> bool:
        return url.endswith("nodejs_26.4.0-1_aarch64.deb")

    monkeypatch.setattr("pm.packages._http_url_exists", fake_exists)
    url = _termux_nodejs_deb_url("26.4.0")
    assert url.endswith("nodejs_26.4.0-1_aarch64.deb")


def test_termux_nodejs_offline_keeps_historic_rev1_default(monkeypatch):
    monkeypatch.setattr("pm.packages._http_url_exists", lambda url: False)
    url = _termux_nodejs_deb_url("26.4.0")
    assert url.endswith("nodejs_26.4.0-1_aarch64.deb")


def test_nodejs_fetch_url_bionic_uses_resolver(monkeypatch):
    monkeypatch.setattr(
        "pm.packages._termux_nodejs_deb_url",
        lambda version: f"https://example.test/nodejs_{version}_aarch64.deb",
    )
    node = get_package("node")
    assert isinstance(node, Nodejs)
    assert node.fetch_url("26.13.1", "linux-arm64-bionic").endswith(
        "nodejs_26.13.1_aarch64.deb"
    )


def test_node_bionic_lock_agreement_still_holds(monkeypatch):
    """Historic lock rows with ``-1`` must still round-trip offline."""
    import json
    from pathlib import Path

    monkeypatch.setattr("pm.packages._http_url_exists", lambda url: False)
    lock = json.loads(
        (Path(__file__).resolve().parents[2] / "pm" / "lock.json").read_text(
            encoding="utf-8"
        )
    )
    row = lock["packages"]["node"]["artifacts"]["linux-arm64-bionic"]
    # Current tip pin uses the -1 shape; offline fallback must reproduce it.
    import re

    m = re.search(r"nodejs_(?P<ver>[0-9.]+)-1_aarch64\.deb$", row["url"])
    assert m, row["url"]
    assert get_package("node").fetch_url(m.group("ver"), "linux-arm64-bionic") == row["url"]
