from __future__ import annotations

import pytest

from hermes_platform.host import products
from hermes_platform.host.products import looks_like_nvidia_arm_soc

_MARKER = products._NVIDIA_SOC_MODEL_MARKERS[0]
_VENDOR = products._NVIDIA_SOC_VENDOR


@pytest.mark.parametrize(
    ("native_arch", "cpu_model", "cpu_vendor", "expected"),
    [
        ("arm64", f"{_VENDOR} {_MARKER} (18-core CPU)", _VENDOR, True),
        ("arm64", f"{_MARKER} (18-core CPU)", _VENDOR, True),
        ("arm64", f"{_VENDOR} {_MARKER}", "", True),
        ("amd64", f"{_VENDOR} {_MARKER}", _VENDOR, False),
        ("arm64", "Snapdragon X Elite", "Qualcomm", False),
        ("arm64", f"{_VENDOR} GPU only", _VENDOR, False),
    ],
)
def test_looks_like_nvidia_arm_soc(native_arch: str, cpu_model: str, cpu_vendor: str, expected: bool) -> None:
    assert looks_like_nvidia_arm_soc(native_arch=native_arch, cpu_model=cpu_model, cpu_vendor=cpu_vendor) is expected
