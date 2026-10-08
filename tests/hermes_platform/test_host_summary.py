"""``hermes_platform.host.summary``: the machine and account facts the desktop questionnaire reads."""

from __future__ import annotations

import pytest

from hermes_platform.host import facts, summary

# The display name NVIDIA ships for the RTX Spark SoC (tests/hermes_cli/test_local_n1x_pci_identity.py).
_N1X = "NVIDIA RTX Spark N1X"


@pytest.fixture
def machine(monkeypatch):
    """Pin every hardware fact; returns a setter for the ones a case changes."""
    values = {"os_family": "linux", "native_arch": "amd64", "gpu_class": "unknown", "cpu_model": "",
              "cpu_vendor": "", "ram_total_bytes": 16 * 2**30}

    def pin(**changes):
        values.update(changes)
        for name, value in values.items():
            monkeypatch.setattr(facts, name, lambda value=value: value)
        summary.summary.cache_clear()
        return summary.summary()

    monkeypatch.setattr(summary, "_account", lambda: ("ada", "Ada Lovelace"))
    yield pin
    summary.summary.cache_clear()


def test_shape(machine):
    result = machine()

    assert set(result) == {"machine", "machine_kind", "has_nvidia_gpu", "is_spark", "locale", "full_name"}
    assert result["machine"]["os_family"] == "linux"
    assert result["machine"]["ram_gb"] == 16
    assert "gpu_class" not in result["machine"]  # unmeasured keys are left out, not sent as null
    assert result["full_name"] == "Ada Lovelace"
    assert result["locale"] is None or isinstance(result["locale"], str)


def test_n1x_cpu_string_is_a_spark(machine):
    result = machine(os_family="win32", native_arch="arm64", cpu_model=_N1X, cpu_vendor="NVIDIA")

    assert result["is_spark"] is True
    assert result["machine_kind"] == "Spark"


def test_dgx_spark_by_model_string(machine):
    assert machine(native_arch="arm64", cpu_model="DGX_Spark GB10")["is_spark"] is True


@pytest.mark.parametrize(("os_family", "arch", "gpu", "kind", "nvidia"), [
    ("win32", "amd64", "nvidia", "PC", True),
    ("darwin", "arm64", "unknown", "Mac", False),
    ("linux", "amd64", "amd", "computer", False),
])
def test_ordinary_machines_are_not_sparks(machine, os_family, arch, gpu, kind, nvidia):
    result = machine(os_family=os_family, native_arch=arch, gpu_class=gpu, cpu_model="Generic CPU")

    assert result["is_spark"] is False
    assert result["machine_kind"] == kind
    assert result["has_nvidia_gpu"] is nvidia


@pytest.mark.parametrize("full", ["ada", "Ada", "p14", "root", "x", "ada lovelace"])
def test_a_login_handle_is_never_offered_as_the_name(machine, monkeypatch, full):
    monkeypatch.setattr(summary, "_account", lambda: ("ada", full))

    assert machine()["full_name"] is None


@pytest.mark.parametrize(("os_family", "mac_ver", "release"), [
    ("darwin", "26.0.1", "26.0.1"),
    ("darwin", "", "25.0.0"),  # no macOS version reported: fall back to the kernel release
    ("linux", "26.0.1", "25.0.0"),
])
def test_os_release_is_the_macos_version_on_a_mac(machine, monkeypatch, os_family, mac_ver, release):
    monkeypatch.setattr(summary.platform, "mac_ver", lambda: (mac_ver, ("", "", ""), ""))
    monkeypatch.setattr(summary.platform, "release", lambda: "25.0.0")

    assert machine(os_family=os_family)["machine"]["os_release"] == release


def test_cached_per_process(machine, monkeypatch):
    first = machine()
    monkeypatch.setattr(facts, "os_family", lambda: "darwin")

    assert summary.summary() is first
