"""Shared fixtures for the AG-UI adapter test package."""
import pytest

from agui_adapter import approvals


@pytest.fixture(autouse=True)
def _clean_parked_registry():
    """Clear the process-global parked-approval registry around every test.

    The registry (``approvals._parked``) is keyed by ``thread_id`` and shared
    across the whole package. An entry leaked by a test that fails before its
    in-body ``take()``/``discard()`` would otherwise cascade into a later test
    (``register()`` refuses to overwrite, turning a real failure into a
    spurious one downstream). Living in a package ``conftest`` protects both
    ``test_approvals.py`` and ``test_e2e_aimock.py``.
    """
    approvals._parked.clear()
    yield
    approvals._parked.clear()
