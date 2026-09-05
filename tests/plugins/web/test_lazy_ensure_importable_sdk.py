"""``lazy_ensure`` must not fail when a feature's SDK is already importable.

The helper's contract (docstring and callers) says the import at the call site
is the real gate: an install that cannot happen is only an error when the
package is genuinely absent. ``tools.lazy_deps.ensure`` signals an unusable
feature with ``FeatureUnavailable`` — a ``RuntimeError``, not an
``ImportError`` — so the pre-split handler never caught it and the broad
``except Exception`` re-raised it as ``ImportError`` even when the package was
present. On a host with ``security.allow_lazy_installs=false`` that made the
provider unusable regardless of whether the SDK was installed.

Ported from the per-provider fix (PR for "a benched lazy-install must not
disable an importable SDK") to the shared ``plugins/web/_common.lazy_ensure``
that succeeded it — same contract, every web vendor covered.
"""

from __future__ import annotations

import sys
import types

import pytest

from plugins.web import _common
from tools.lazy_deps import FeatureUnavailable


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


def _deny_lazy_install(*_args, **_kwargs):
    raise FeatureUnavailable(
        "search.parallel",
        ("parallel-web==0.4.2",),
        "lazy installs disabled (security.allow_lazy_installs=false)",
    )


def test_importable_sdk_survives_disabled_lazy_installs(monkeypatch, fake_parallel_sdk):
    """Feature reported unavailable, but the package imports: not an error."""
    monkeypatch.setattr("tools.lazy_deps.ensure", _deny_lazy_install)

    _common.lazy_ensure("search.parallel")  # must not raise

    from parallel import Parallel

    assert Parallel(api_key="k").api_key == "k"


def test_missing_sdk_still_reports_the_install_hint(monkeypatch):
    """Genuinely absent package keeps the actionable ImportError."""
    monkeypatch.delitem(sys.modules, "parallel", raising=False)
    monkeypatch.setattr("tools.lazy_deps.ensure", _deny_lazy_install)

    with pytest.raises(ImportError) as excinfo:
        _common.lazy_ensure("search.parallel")

    assert "parallel-web" in str(excinfo.value)


def test_unrelated_failure_is_still_surfaced(monkeypatch, fake_parallel_sdk):
    """A non-availability error is a real fault and must not be swallowed."""

    def boom(*_args, **_kwargs):
        raise OSError("disk exploded")

    monkeypatch.setattr("tools.lazy_deps.ensure", boom)

    with pytest.raises(ImportError, match="disk exploded"):
        _common.lazy_ensure("search.parallel")


def test_absent_helper_degrades_to_the_import(monkeypatch):
    """Without lazy_deps at all, the call-site import remains the only gate."""
    monkeypatch.setitem(sys.modules, "tools", types.ModuleType("tools"))
    monkeypatch.setitem(sys.modules, "tools.lazy_deps", None)  # import raises

    _common.lazy_ensure("search.parallel")  # must not raise, must not install
