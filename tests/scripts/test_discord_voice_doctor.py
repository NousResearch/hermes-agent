"""Security-floor tests for the standalone Discord voice doctor."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def doctor_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "discord-voice-doctor.py"
    spec = spec_from_file_location("discord_voice_doctor", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.5.0", False),
        ("unknown", False),
        ("1.6.2", True),
        ("1.6.3", True),
    ],
)
def test_pynacl_security_floor(doctor_module, version, expected):
    assert doctor_module._pynacl_version_is_patched(version) is expected
