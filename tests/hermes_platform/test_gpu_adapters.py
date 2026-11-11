"""DXGI's adapter list, read live on Windows."""

from __future__ import annotations

import pytest

from hermes_platform.host.gpu_adapters import windows_gpu_adapters


@pytest.mark.platforms("windows")
def test_dxgi_lists_this_machines_adapters() -> None:
    found = windows_gpu_adapters()
    assert found, "DXGI lists at least Microsoft's software adapter"
    assert any(a.software for a in found)
    assert all(a.shared_bytes > 0 and a.dedicated_bytes >= 0 for a in found)


@pytest.mark.platforms("not windows")
def test_dxgi_answers_empty_off_windows() -> None:
    assert windows_gpu_adapters() == ()
