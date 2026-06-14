import importlib.util
from pathlib import Path


def _load_doctor():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "discord-voice-doctor.py"
    spec = importlib.util.spec_from_file_location("discord_voice_doctor", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pynacl_floor_rejects_vulnerable_versions():
    doctor = _load_doctor()

    assert not doctor._version_at_least("1.5.0", doctor.PYNACL_PATCHED_FLOOR)
    assert not doctor._version_at_least("unknown", doctor.PYNACL_PATCHED_FLOOR)
    assert doctor._version_at_least("1.6.2", doctor.PYNACL_PATCHED_FLOOR)
    assert doctor._version_at_least("1.6.3", doctor.PYNACL_PATCHED_FLOOR)
