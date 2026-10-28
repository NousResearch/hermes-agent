"""The checkout exemption must not weaken the refusal of LIVE operator state.

A repo legitimately lives at ``~/.hermes/hermes-agent`` — the documented install location — so
the guard cannot treat the checkout's own files as the operator's live home. These pin that
boundary from BOTH sides: the checkout and its install metadata are exempt, while everything
else in the real home is still refused — including the secrets that sit in the very same
directory (``auth.json``), which is why the exemption matches exact filenames, never a prefix.
"""

import os
from pathlib import Path

import pytest

# This module probes the guard deliberately, so the autouse guard fixture must not also fire.
pytestmark = pytest.mark.allow_real_home_io

REAL_HOME = Path.home() / ".hermes"
CHECKOUT = Path(__file__).resolve().parent.parent


def _exists(path) -> bool:
    """Plain ``os.path`` rather than ``Path.exists``: the assertion under test is the guard, not
    this helper, so the setup must not depend on which I/O calls it wraps."""
    return os.path.exists(path)


def _guard(is_owned_path=None):
    from tests.home_io_guard import HomeIOGuard
    from tests.conftest import _REAL_HERMES_ROOT_CANDIDATES, _is_owned_by_checkout

    return HomeIOGuard(
        lambda: _REAL_HERMES_ROOT_CANDIDATES,
        lambda: (),
        is_owned_path=_is_owned_by_checkout if is_owned_path is None else is_owned_path,
    )


@pytest.mark.parametrize("name", ["auth.json", "gateway_state.json", "models_dev_cache.json"])
def test_live_state_in_the_real_home_is_still_refused(name):
    """The guard must REFUSE these. A failure here means the exemption leaked."""
    target = REAL_HOME / name
    if not _exists(target):
        pytest.skip(f"{name} not present on this machine")
    with pytest.raises(AssertionError) as excinfo:
        _guard().check(target)
    assert "REAL hermes home" in str(excinfo.value)


@pytest.mark.parametrize(
    "relative",
    ["hermes_cli/gateway.py", "hermes_cli/gateway_launchd.py", "tests/conftest.py", "pm/paths.py"],
)
def test_files_inside_the_checkout_are_exempt(relative):
    from tests.conftest import _is_owned_by_checkout

    assert _is_owned_by_checkout(CHECKOUT / relative)


@pytest.mark.parametrize("name", ["manifest.json", "install-stamp.json"])
def test_sibling_install_metadata_is_exempt(name):
    """The installer writes these beside the checkout; its own resolvers read them."""
    from tests.conftest import _is_owned_by_checkout

    assert _is_owned_by_checkout(REAL_HOME / name)


@pytest.mark.parametrize("name", ["auth.json", "state.db", "models_dev_cache.json"])
def test_unrelated_siblings_are_not_exempt(name):
    """Exact-filename only: the exemption must not spill onto the rest of the real home."""
    from tests.conftest import _is_owned_by_checkout

    assert not _is_owned_by_checkout(REAL_HOME / name)


def test_a_broken_predicate_exempts_nothing():
    """Fails SAFE: a predicate that raises must leave the refusal standing."""
    if not _exists(REAL_HOME / "auth.json"):
        pytest.skip("auth.json not present on this machine")

    def boom(_value):
        raise RuntimeError("predicate exploded")

    with pytest.raises(AssertionError, match="REAL hermes home"):
        _guard(is_owned_path=boom).check(REAL_HOME / "auth.json")


def test_a_falsely_predicate_exempts_nothing():
    """A predicate that always says False must leave every refusal standing.

    The exemption is driven entirely by the predicate, so an always-False one reproduces the
    pre-change behaviour: the guard still refuses. This is the real regression risk — a
    predicate wired up but ineffective would silently un-guard the home.
    """
    guarded = REAL_HOME / "auth.json"
    if not _exists(guarded):
        pytest.skip("auth.json not present on this machine")
    with pytest.raises(AssertionError, match="REAL hermes home"):
        _guard(is_owned_path=lambda _v: False).check(guarded)
