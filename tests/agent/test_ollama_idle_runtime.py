"""Profile opt-in and hardware signals through the runtime's real config reader."""

from types import SimpleNamespace

import pytest


def test_profile_opt_in_is_not_inherited_between_profiles(tmp_path):
    from agent.ollama_idle_runtime import release_enabled

    first = tmp_path / "profiles" / "first"
    second = tmp_path / "profiles" / "second"
    for home in (first, second):
        home.mkdir(parents=True)
    (first / "config.yaml").write_text(
        "local_runtime:\n  ollama_idle_release: true\n", encoding="utf-8"
    )
    assert release_enabled(first)
    assert not release_enabled(second)
    assert release_enabled(first)


@pytest.mark.parametrize("raw", ["false", "'true'", "null", "[]"])
def test_invalid_or_disabled_setting_stays_off(tmp_path, raw):
    from agent.ollama_idle_runtime import release_enabled

    (tmp_path / "config.yaml").write_text(
        f"local_runtime:\n  ollama_idle_release: {raw}\n", encoding="utf-8"
    )
    assert not release_enabled(tmp_path)


@pytest.mark.parametrize(
    "output, expected",
    [
        ("10000, 100\n", True),
        ("10000, 9000\n", False),
        ("10000, 100\n10000, 9000\n", False),
        ("10000, 100\n10000, 200\n", True),
        ("N/A, N/A\n", None),
        ("0, 0\n", None),
        ("10000, -1\n", None),
    ],
)
def test_pressure_requires_valid_low_headroom_on_every_card(
    monkeypatch, output, expected
):
    from agent.ollama_idle_runtime import gpu_memory_pressure

    monkeypatch.setattr(
        "hermes_cli.local_runtime.hardware._nvidia_smi_path", lambda: "/fake/smi"
    )
    monkeypatch.setattr(
        "subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout=output)
    )
    assert gpu_memory_pressure() is expected


def test_unknown_pressure_does_not_start_an_unload(monkeypatch):
    from agent.ollama_idle_runtime import gpu_memory_pressure

    monkeypatch.setattr(
        "hermes_cli.local_runtime.hardware._nvidia_smi_path", lambda: None
    )
    monkeypatch.setattr("sys.platform", "linux")
    assert gpu_memory_pressure() is None
