"""``lazy_ensure`` must not fail when a feature's SDK is already importable.

The helper's contract (docstring and callers) says the import at the call site
is the real gate: an install that cannot happen is only an error when the
package is genuinely absent. ``pm.ensure_import`` signals an unusable feature
with ``pm.InstallError`` — a ``RuntimeError``, not an ``ImportError`` — so a
naive ``except ImportError`` never caught it and the broad ``except Exception``
re-raised it as ``ImportError`` even when the package was present. On a host
with lazy installs off that made the provider unusable regardless of whether
the SDK was importable.

Ported from the per-provider fix (PR for "a benched lazy-install must not
disable an importable SDK") to the shared ``plugins/web/_common.lazy_ensure``
that succeeded it — same contract, every web vendor covered.
"""

from __future__ import annotations

import sys
import types

import pytest

from pm.package import InstallError
from plugins.web import _common


@pytest.fixture
def fake_parallel_sdk(monkeypatch):
    """Make ``import parallel`` succeed without installing anything."""
    module = types.ModuleType("parallel")

    class Parallel:
        def __init__(self, api_key):
            self.api_key = api_key

    class AsyncParallel:
        def __init__(self, api_key):
            self.api_key = api_key

    module.Parallel = Parallel
    module.AsyncParallel = AsyncParallel
    monkeypatch.setitem(sys.modules, "parallel", module)
    return module


@pytest.fixture
def fake_exa_sdk(monkeypatch):
    """Make ``import exa_py`` succeed without installing anything."""
    module = types.ModuleType("exa_py")

    class Exa:
        def __init__(self, api_key):
            self.api_key = api_key

    module.Exa = Exa
    monkeypatch.setitem(sys.modules, "exa_py", module)
    return module


def _deny_lazy_install(*_args, **_kwargs):
    raise InstallError(
        "venv",
        'package "parallel-web" missing and lazy installs are disabled',
        "lazy installs disabled (security.allow_lazy_installs=false)",
    )


def _deny_exa_lazy_install(*_args, **_kwargs):
    raise InstallError(
        "venv",
        "feature search.exa unavailable",
        "lazy installs disabled (security.allow_lazy_installs=false)",
    )


def test_importable_sdk_survives_disabled_lazy_installs(monkeypatch, fake_parallel_sdk):
    """Feature reported unavailable, but the package imports: not an error."""
    monkeypatch.setattr("pm.ensure_import", _deny_lazy_install)

    _common.lazy_ensure("search.parallel")  # must not raise

    from parallel import Parallel

    assert Parallel(api_key="k").api_key == "k"


def test_importable_exa_sdk_survives_disabled_lazy_installs(monkeypatch, fake_exa_sdk):
    """Same for a vendor whose package name is not its slug: ``exa-py`` imports as ``exa_py``."""
    monkeypatch.setattr("pm.ensure_import", _deny_exa_lazy_install)

    _common.lazy_ensure("search.exa")  # must not raise

    from exa_py import Exa

    assert Exa(api_key="k").api_key == "k"


def test_missing_sdk_still_reports_the_install_hint(monkeypatch):
    """Genuinely absent package keeps the actionable ImportError."""
    monkeypatch.delitem(sys.modules, "parallel", raising=False)
    monkeypatch.setattr("pm.ensure_import", _deny_lazy_install)

    with pytest.raises(ImportError) as excinfo:
        _common.lazy_ensure("search.parallel")

    assert "parallel-web" in str(excinfo.value)


def test_unrelated_failure_is_still_surfaced(monkeypatch, fake_parallel_sdk):
    """A non-availability error is a real fault and must not be swallowed."""

    def boom(*_args, **_kwargs):
        raise OSError("disk exploded")

    monkeypatch.setattr("pm.ensure_import", boom)

    with pytest.raises(ImportError, match="disk exploded"):
        _common.lazy_ensure("search.parallel")


def test_absent_helper_degrades_to_the_import(monkeypatch):
    """Without pm at all, the call-site import remains the only gate."""
    monkeypatch.setitem(sys.modules, "pm", None)  # import raises

    _common.lazy_ensure("search.parallel")  # must not raise, must not install
