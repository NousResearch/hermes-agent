"""#104076 regression: the --no-overlay probe must not silently fail on a bare
command name under a thin PATH.

`_cua_driver_supports_no_overlay("cua-driver")` raised FileNotFoundError inside
the probing process (headless service / GUI launcher whose PATH lacks
~/.local/bin), the bare `except` returned False, and the flag was dropped even
with `computer_use.no_overlay: true` — captures then froze on the overlay's
stale frame. The probe must resolve a bare name through
`resolve_cua_driver_cmd()` before spawning, and a probe failure must be
visible in the log, not silent.
"""
import pytest
from unittest.mock import patch

import tools.computer_use.cua_backend_driver as cua_backend_driver

_LOGGER_NAME = "tools.computer_use.cua_backend"


class _FakeCompleted:
    def __init__(self, stdout: str = ""):
        self.stdout, self.stderr, self.returncode = stdout, "", 0


@pytest.fixture(autouse=True)
def _clean_probe_cache():
    """The probe is lru_cached module-wide; isolate every test from its
    neighbors (repo flake policy: state never leaks between tests)."""
    cua_backend_driver._cua_driver_supports_no_overlay.cache_clear()
    yield
    cua_backend_driver._cua_driver_supports_no_overlay.cache_clear()


def test_bare_name_probes_the_resolved_binary(monkeypatch):
    """A bare 'cua-driver' arg must probe resolve_cua_driver_cmd()'s answer,
    not spawn a name the probing process cannot find."""
    probed = {}

    def fake_run_driver(cmd, *a, **kw):
        probed["cmd"] = cmd
        return _FakeCompleted(stdout="  --no-overlay  disable the cursor overlay")

    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd",
                        lambda override=None: "/home/user/.local/bin/cua-driver")
    monkeypatch.setattr(cua_backend_driver._cb(), "_run_driver", fake_run_driver)

    assert cua_backend_driver._cua_driver_supports_no_overlay("cua-driver") is True
    assert probed["cmd"] == "/home/user/.local/bin/cua-driver"


def test_path_form_probes_that_exact_binary(monkeypatch):
    """An explicit path is probed as-is — never replaced by the resolver."""
    probed = {}

    def fake_run_driver(cmd, *a, **kw):
        probed["cmd"] = cmd
        return _FakeCompleted(stdout="--no-overlay")

    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd",
                        lambda override=None: "/somewhere/else/cua-driver")
    monkeypatch.setattr(cua_backend_driver._cb(), "_run_driver", fake_run_driver)

    assert cua_backend_driver._cua_driver_supports_no_overlay("/opt/bin/cua-driver") is True
    assert probed["cmd"] == "/opt/bin/cua-driver"


def test_overlay_flag_survives_a_thin_path(monkeypatch):
    """End-to-end contract: no_overlay=true + a bare command name + a driver
    that supports the flag => the flag is appended (#104076)."""
    monkeypatch.setattr(cua_backend_driver._cb(), "_cua_no_overlay", lambda: True)
    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd",
                        lambda override=None: "/home/user/.local/bin/cua-driver")
    monkeypatch.setattr(cua_backend_driver._cb(), "_run_driver",
                        lambda cmd, *a, **kw: _FakeCompleted(stdout="--no-overlay"))

    args = cua_backend_driver._mcp_args_with_overlay_flag(["serve", "--socket", "/x"])
    assert args[-1] == "--no-overlay"


def test_unresolvable_probe_fails_visible_not_silent(monkeypatch, caplog):
    """When nothing resolves, the flag is still dropped — but the probe says
    WHY in the debug log instead of failing silently."""
    def boom(cmd, *a, **kw):
        raise FileNotFoundError(cmd)

    monkeypatch.setattr(cua_backend_driver, "resolve_cua_driver_cmd", lambda override=None: None)
    monkeypatch.setattr(cua_backend_driver._cb(), "_run_driver", boom)

    with caplog.at_level("DEBUG", logger=_LOGGER_NAME):
        assert cua_backend_driver._cua_driver_supports_no_overlay("cua-driver") is False
    assert any("probe failed" in r.message for r in caplog.records)
