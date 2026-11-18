"""Startup latency and the install snapshot's version-lag, channel and hardware fields."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

import hermes_cli.observability.relay_shared_metrics as relay
import hermes_cli.observability.shared_metrics_install as install
import hermes_cli.observability.shared_metrics_startup as startup
from tui_gateway import server
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_platform.host import facts


@pytest.fixture
def marks(monkeypatch):
    recorded: list[tuple[str, dict]] = []
    monkeypatch.setattr(relay, "enabled", lambda: True)
    monkeypatch.setattr(relay, "record_process_mark", lambda mark, data: recorded.append((mark, data)))
    monkeypatch.setattr(startup, "_recorded", set())
    return recorded


@pytest.mark.parametrize(("elapsed_ms", "bucket"), [
    (0, "lt_500ms"), (499.9, "lt_500ms"), (500, "500ms_to_1s"), (1_999, "1s_to_2s"),
    (2_000, "2s_to_5s"), (9_999, "5s_to_10s"), (10_000, "gte_10s"), (3_600_000, "gte_10s"),
])
def test_startup_latency_buckets_are_valid_counter_dimensions(elapsed_ms, bucket):
    fields = startup.startup_latency_fields(surface="gateway_boot", elapsed_ms=elapsed_ms)

    assert fields == {"latency_bucket": bucket, "surface": "gateway_boot"}
    assert contract.counter_dimensions_are_valid(contract.STARTUP_LATENCY_METRIC, fields)


@pytest.mark.parametrize("raw", [{"surface": "web", "elapsed_ms": 10}, {"surface": "cli", "elapsed_ms": -1},
                                 {"surface": "cli", "elapsed_ms": "fast"}, {"surface": "cli", "elapsed_ms": True}])
def test_startup_latency_rejects_unknown_surfaces_and_bad_durations(raw):
    assert startup.startup_latency_fields(**raw) is None


def test_process_ready_records_process_start_to_now_once_per_surface(marks, monkeypatch):
    monkeypatch.setattr(startup, "process_started_at", lambda: time.time() - 1.5)

    startup.record_process_ready("cli")
    startup.record_process_ready("cli")
    startup.record_process_ready("gateway_boot")

    assert marks == [
        (contract.STARTUP_LATENCY_MARK, {"latency_bucket": "1s_to_2s", "surface": "cli"}),
        (contract.STARTUP_LATENCY_MARK, {"latency_bucket": "1s_to_2s", "surface": "gateway_boot"}),
    ]


def test_process_ready_records_nothing_when_shared_metrics_are_off(marks, monkeypatch):
    monkeypatch.setattr(relay, "enabled", lambda: False)

    startup.record_process_ready("serve_boot")

    assert marks == []


def test_background_ready_record_does_not_block_and_still_lands(marks, monkeypatch):
    monkeypatch.setattr(startup, "process_started_at", lambda: time.time() - 0.1)

    startup.record_process_ready("serve_boot", background=True)
    for _ in range(200):
        if marks:
            break
        time.sleep(0.01)

    assert marks == [(contract.STARTUP_LATENCY_MARK, {"latency_bucket": "lt_500ms", "surface": "serve_boot"})]


def test_kanban_worker_one_shot_is_not_a_cli_startup(marks, monkeypatch):
    monkeypatch.setattr(startup, "process_started_at", lambda: time.time())
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")

    startup.record_cli_one_shot_ready()

    assert marks == []


def _rpc(params: dict) -> dict:
    return server.handle_request(
        {"jsonrpc": "2.0", "id": "r1", "method": "shared_metrics.startup_latency", "params": params})


def test_rpc_takes_the_declared_surface_else_env_detection(marks, monkeypatch):
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    monkeypatch.delenv("HERMES_DESKTOP_TERMINAL", raising=False)

    # A Desktop attached to a URL/cloud backend: no HERMES_DESKTOP here, the client says who it is.
    assert _rpc({"elapsed_ms": 2500, "surface": "desktop_attach", "launch_id": "a"})["result"] == {"ok": True}
    assert _rpc({"elapsed_ms": 700, "launch_id": "b"})["result"] == {"ok": True}
    monkeypatch.setenv("HERMES_DESKTOP", "1")
    assert _rpc({"elapsed_ms": 12_000, "launch_id": "c"})["result"] == {"ok": True}
    assert _rpc({"elapsed_ms": -5, "surface": "tui", "launch_id": "d"})["result"] == {"ok": True}

    assert [data for _, data in marks] == [
        {"latency_bucket": "2s_to_5s", "surface": "desktop_attach"},
        {"latency_bucket": "500ms_to_1s", "surface": "tui"},
        {"latency_bucket": "gte_10s", "surface": "desktop_attach"},
    ]


def test_rpc_counts_each_client_launch_once_per_backend_process(marks):
    # A Desktop reconnecting to the same backend re-sends its launch; a later Desktop launch
    # against the same long-lived backend is a new launch and still counts.
    for _ in range(3):
        _rpc({"elapsed_ms": 1200, "surface": "desktop_attach", "launch_id": "launch-1"})
    _rpc({"elapsed_ms": 300, "surface": "desktop_attach", "launch_id": "launch-2"})
    # Older clients send no id and latch on their side; the backend still counts them once.
    for _ in range(2):
        _rpc({"elapsed_ms": 1200, "surface": "tui"})

    assert [data for _, data in marks] == [
        {"latency_bucket": "1s_to_2s", "surface": "desktop_attach"},
        {"latency_bucket": "lt_500ms", "surface": "desktop_attach"},
        {"latency_bucket": "1s_to_2s", "surface": "tui"},
    ]


def test_in_place_relaunch_is_not_a_startup_but_its_children_are(marks, monkeypatch):
    """``sessions browse`` -> resume execs in place: same PID, so process start covers the picker."""
    from hermes_cli import relaunch as relaunch_mod

    monkeypatch.setenv(startup.RELAUNCHED_PID_ENV, "")  # restored after relaunch() stamps it
    monkeypatch.setattr(relaunch_mod.sys, "platform", "linux")
    monkeypatch.setattr(startup, "process_started_at", lambda: time.time() - 30)
    # The exec'd program keeps this PID and environment and reaches its first prompt.
    monkeypatch.setattr(relaunch_mod.os, "execvp", lambda *_a: startup.record_process_ready("cli"))

    relaunch_mod.relaunch(["--resume", "x"], preserve_inherited=False)
    assert marks == []

    # A child it spawns later inherits the env but not the PID: a real start of its own.
    child_pid = os.getpid() + 1
    monkeypatch.setattr(startup.os, "getpid", lambda: child_pid)
    startup.record_process_ready("gateway_boot")
    assert [data["surface"] for _, data in marks] == ["gateway_boot"]


def test_process_ready_when_off_starts_no_thread_and_reads_no_process_time(monkeypatch):
    monkeypatch.setattr(relay, "enabled", lambda: False)
    monkeypatch.setattr(startup, "_recorded", set())
    touched: list[str] = []
    monkeypatch.setattr(startup, "process_started_at", lambda: touched.append("psutil"))
    monkeypatch.setattr(startup.threading, "Thread", lambda *a, **k: touched.append("thread"))

    startup.record_process_ready("serve_boot", background=True)
    startup.record_process_ready("cli")

    assert touched == []


def test_cli_first_rendered_prompt_records_once(marks, monkeypatch):
    monkeypatch.setattr(startup, "process_started_at", lambda: time.time() - 0.2)
    on_render = startup.cli_prompt_ready_handler()

    for _ in range(3):
        on_render(object())
    for _ in range(200):
        if marks:
            break
        time.sleep(0.01)
    time.sleep(0.05)

    assert marks == [(contract.STARTUP_LATENCY_MARK, {"latency_bucket": "lt_500ms", "surface": "cli"})]


# ---- install snapshot ----

_FULL_V4 = {
    "behind_bucket": "unknown", "gpu_class": "nvidia", "local_model_provider_used": "yes",
    "ram_bucket": "16g_to_32g", "release_channel": "stable", "version_age_bucket": "lt_7d",
}


def test_install_snapshot_accepts_v4_fields_and_rows_counted_before_them():
    legacy = {
        "cron_job_count_bucket": "0", "display_language": "en", "install_age_bucket": "lt_1h",
        "main_provider": "openrouter", "mcp_server_count_bucket": "0", "memory_provider": "builtin",
        "messaging_platform_count_bucket": "0", "plugin_count_bucket": "0", "profile_count_bucket": "1",
        "skill_count_bucket": "0", "terminal_backend": "local",
    }

    assert contract.counter_dimensions_are_valid(contract.INSTALL_SNAPSHOT_METRIC, legacy)
    assert contract.counter_dimensions_are_valid(contract.INSTALL_SNAPSHOT_METRIC, {**legacy, **_FULL_V4})
    assert not contract.counter_dimensions_are_valid(
        contract.INSTALL_SNAPSHOT_METRIC, {**legacy, **_FULL_V4, "gpu_class": "rtx-4090"})
    partial = {**legacy, "ram_bucket": "lt_8g"}
    assert not contract.counter_dimensions_are_valid(contract.INSTALL_SNAPSHOT_METRIC, partial)


@pytest.mark.parametrize(("gib", "bucket"), [
    (3.8, "lt_8g"), (7.6, "8g_to_16g"), (15.5, "16g_to_32g"), (31.2, "32g_to_64g"),
    (62.6, "64g_to_128g"), (125.7, "gte_128g"), (512, "gte_128g"),
])
def test_ram_buckets_land_on_the_nominal_size(monkeypatch, gib, bucket):
    monkeypatch.setattr(facts, "ram_total_bytes", lambda: int(gib * 1024 ** 3))

    assert install.ram_bucket() == bucket


def test_ram_unknown_when_the_os_does_not_say(monkeypatch):
    monkeypatch.setattr(facts, "ram_total_bytes", lambda: None)

    assert install.ram_bucket() == "unknown"


def test_meminfo_total_is_parsed_in_bytes():
    assert facts.parse_meminfo_total("MemFree: 12 kB\nMemTotal:       65681892 kB\n") == 65681892 * 1024
    assert facts.parse_meminfo_total("garbage") is None


@pytest.mark.parametrize(("vendors", "gpu"), [
    (["0x8086", "0x10de"], "nvidia"), (["0x8086", "0x1002"], "amd"), (["0x8086"], "intel"),
    (["0x1af4"], "none"), ([], "none"), (["10DE"], "nvidia"), ([""], "unknown"), (["", "0x10de"], "nvidia"),
])
def test_gpu_vendor_priority_prefers_the_discrete_card(vendors, gpu):
    assert facts.classify_gpu_vendors(vendors) == gpu


def _version(**kw):
    base = {"commit": "a" * 40, "commit_date": None, "branch": None}
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.parametrize(("days", "bucket"), [(0.5, "lt_7d"), (8, "7d_to_30d"), (45, "30d_to_90d"), (400, "gte_90d")])
def test_version_age_is_the_installed_commit_age(monkeypatch, days, bucket):
    now = 1_800_000_000
    monkeypatch.setattr(install, "_version_info", lambda: _version(commit_date=int(now - days * 86_400)))

    assert install.version_age_bucket(now=now) == bucket


def test_version_age_unknown_without_a_commit_date(monkeypatch):
    monkeypatch.setattr(install, "_version_info", lambda: _version(commit_date=None))

    assert install.version_age_bucket() == "unknown"


def _write_update_cache(home, root, *, behind, head, ts):
    from hermes_cli.update_channel import install_id

    path = home / "source-checks" / f"{install_id(root)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ts": ts, "identity": {"head": head}, "status": {"behind": behind}}))


def test_behind_comes_only_from_a_fresh_cached_check_of_this_revision(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    root.mkdir()
    monkeypatch.setattr(install, "_project_root", lambda: root)
    monkeypatch.setattr(install, "_version_info", lambda: _version())
    home = tmp_path / "home"
    now = time.time()

    assert install.behind_bucket(home=home, now=now) == "unknown"
    _write_update_cache(home, root, behind=4, head="a" * 40, ts=now - 3600)
    assert install.behind_bucket(home=home, now=now) == "3_to_5"
    _write_update_cache(home, root, behind=0, head="a" * 40, ts=now - 60)
    assert install.behind_bucket(home=home, now=now) == "0"
    # Checked before the install moved to another revision: that count is about a different commit.
    _write_update_cache(home, root, behind=4, head="b" * 40, ts=now - 60)
    assert install.behind_bucket(home=home, now=now) == "unknown"
    _write_update_cache(home, root, behind=4, head="a" * 40, ts=now - 8 * 86_400)
    assert install.behind_bucket(home=home, now=now) == "unknown"


def test_behind_never_touches_the_network(tmp_path, monkeypatch):
    import socket

    def _no_network(*_a, **_k):
        raise AssertionError("snapshot must not open a socket")

    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(install, "_project_root", lambda: tmp_path)

    assert install.behind_bucket(home=tmp_path) == "unknown"


@pytest.mark.parametrize(("branch", "channel"), [("main", "main"), ("master", "main"), ("feat/x", "dev"), (None, "unknown")])
def test_source_checkout_channel_follows_its_branch_never_its_remote(tmp_path, monkeypatch, branch, channel):
    monkeypatch.setattr(install, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(install, "_version_info", lambda: _version(branch=branch))

    assert install.release_channel({}) == channel


@pytest.mark.parametrize(("tag", "channel"), [("v0.13.0", "stable"), ("v0.13.0+canary.20260927T120000Z", "main")])
def test_packaged_build_channel_is_its_baked_channel_not_its_branch(tmp_path, monkeypatch, tag, channel):
    from hermes_cli.update_channel import is_canary_tag

    if channel == "main":
        assert is_canary_tag(tag)
    (tmp_path / "install-stamp.json").write_text(json.dumps({"updateMechanism": "electron-updater", "tag": tag}))
    monkeypatch.setattr(install, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(install, "_version_info", lambda: _version(branch="feat/x"))

    assert install.release_channel({}) == channel


@pytest.mark.parametrize(("config", "used"), [
    ({"model": {"provider": "openrouter", "default": "x"}}, "no"),
    ({"model": {"provider": "ollama"}}, "yes"),
    ({"model": {"provider": "custom", "base_url": "http://127.0.0.1:8080/v1"}}, "yes"),
    ({"model": {"provider": "custom", "base_url": "https://api.example.com/v1"}}, "no"),
    ({"model": {"provider": "openrouter"}, "auxiliary": {"vision": {"provider": "lmstudio"}}}, "yes"),
    ({"model": {"provider": "openrouter"}, "auxiliary": {"compression": {"provider": "auto", "base_url": "http://localhost:1"}}}, "no"),
    ({}, "no"),
])
def test_local_model_use_covers_main_and_auxiliary_slots(config, used):
    assert install.local_model_provider_used(config) == used


@pytest.mark.parametrize(("aux", "used"), [
    # The runtime routes a bare base_url + api_key to that endpoint even under provider auto.
    ({"provider": "auto", "base_url": "http://localhost:11434/v1", "api_key": "k", "model": "q"}, "yes"),
    ({"base_url": "http://127.0.0.1:8080/v1", "api_key": "k", "model": "q"}, "yes"),
    ({"base_url": "https://api.example.com/v1", "api_key": "k", "model": "q"}, "no"),
])
def test_local_model_use_sees_a_bare_auxiliary_endpoint(aux, used):
    assert install.local_model_provider_used({"model": {"provider": "openrouter"}, "auxiliary": {"vision": aux}}) == used


@pytest.mark.parametrize(("base_url", "used"), [("http://127.0.0.1:8080/v1", "yes"), ("https://llm.example.com/v1", "no")])
def test_local_model_use_resolves_a_named_provider_used_by_bare_name(monkeypatch, base_url, used):
    import hermes_cli.runtime_provider as rp

    config = {"model": {"provider": "home-llama", "default": "q"},
              "providers": {"home-llama": {"base_url": base_url, "api_key": "k"}}}
    monkeypatch.setattr(rp, "load_config", lambda: config)

    assert install.local_model_provider_used(config) == used


def test_snapshot_fields_degrade_to_unknown_when_a_reader_breaks(monkeypatch):
    def _boom(*_a, **_k):
        raise OSError("unreadable")

    for name in ("behind_bucket", "gpu_class", "ram_bucket", "release_channel", "version_age_bucket",
                 "local_model_provider_used"):
        monkeypatch.setattr(install, name, _boom)

    assert install.install_v4_snapshot_fields({}) == {
        "behind_bucket": "unknown", "gpu_class": "unknown", "local_model_provider_used": "no",
        "ram_bucket": "unknown", "release_channel": "unknown", "version_age_bucket": "unknown",
    }


@pytest.mark.skipif(not os.path.isdir("/sys/class/drm") and not os.path.exists("/proc/meminfo"),
                    reason="needs a Linux host")
def test_host_facts_are_cached_and_in_vocabulary():
    facts.clear_caches()

    assert facts.gpu_class() in contract.GPU_CLASSES
    assert facts.gpu_class.cache_info().currsize == 1
    assert (facts.ram_total_bytes() or 0) >= 0
