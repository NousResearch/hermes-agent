"""``hermes doctor`` probes survive a bracketed-IPv6 NO_PROXY environment (#118159).

Clash Verge / mihomo write ``[::1]`` into NO_PROXY; httpx 0.28.1 turns that into an
unparseable ``all://*[::1]`` mount, so every probe's bare ``httpx.get()``/``httpx.Client()``
raised ``InvalidURL: Invalid port`` and the probe table misreported broken connectivity for
every provider. ``run_probes`` (the single entry the doctor uses for the whole table) sanitizes
the environment once before any worker builds a client.
"""

from __future__ import annotations

import pytest

pytest.importorskip("httpx")

from hermes_cli import doctor_connectivity as dc

# The exact environment Clash Verge / mihomo exports (issue #118159).
_CLASH_NO_PROXY = "127.0.0.1,localhost,::1,[::1]"


@pytest.fixture
def clash_env(monkeypatch):
    for key in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", _CLASH_NO_PROXY)
    monkeypatch.setenv("no_proxy", _CLASH_NO_PROXY)


def test_run_probes_sanitizes_env_before_any_worker_builds_a_client(clash_env, monkeypatch):
    import httpx

    def probe_building_a_bare_client():
        # The probe body's exact defect: a bare trust_env httpx client. Red on base this raises
        # InvalidURL inside the worker; run_probes re-raises it from future.result().
        with httpx.Client() as client:
            assert client._mounts is not None
        return dc._row("stand-in", "ok")

    (result,) = dc.run_probes([("stand-in", probe_building_a_bare_client)])
    ((glyph, label, _detail),) = result.lines
    assert "✓" in glyph and label == "stand-in"
    # The seam did it: the in-place rewrite the rest of the process now sees.
    import os

    assert "[::1]" not in os.environ["NO_PROXY"]
    assert os.environ["no_proxy"] == "127.0.0.1,localhost,::1"


def test_run_probes_leaves_a_clean_environment_untouched(clash_env, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "example.com,10.0.0.0/8")
    monkeypatch.delenv("no_proxy", raising=False)
    (result,) = dc.run_probes([("stand-in", lambda: dc._row("stand-in", "ok"))])
    assert "✓" in result.lines[0][0]
    import os

    assert os.environ["NO_PROXY"] == "example.com,10.0.0.0/8"
