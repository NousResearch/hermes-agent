"""Web dashboard build resource caps (issue #63338).

``npm run build`` under ``web/`` runs Vite 8's Rust-native bundler (Rolldown) plus
the TypeScript solution builder; on small VPS hosts it saturates every CPU and can
OOM the machine. These tests cover the env capping helper and its propagation into
the install/update/dashboard-triggered source build path. A host without a cgroup
limit sizes the heap from ``MemAvailable`` (#125437, mechanism H).
"""

from __future__ import annotations

import pytest

from hermes_cli.local_runtime import hardware
from hermes_cli.web_build_limits import (
    _MAX_WEB_BUILD_HEAP_MB,
    _MIN_WEB_BUILD_HEAP_MB,
    apply_web_build_limits,
    web_build_limits,
)

# A 3.8 GB VPS with 2 GiB available and no swap.
_SMALL_VPS_AVAILABLE_MB = 2048
_SMALL_VPS_MEMINFO = (
    "MemTotal:        3891200 kB\n"
    "MemFree:          412000 kB\n"
    f"MemAvailable:    {_SMALL_VPS_AVAILABLE_MB * 1024} kB\n"
)


def _uncontained_host(monkeypatch, meminfo: str | None) -> None:
    monkeypatch.setattr(
        "hermes_cli.web_build_limits._cgroup_memory_limit_mb", lambda: None
    )
    monkeypatch.setattr(hardware, "_linux_meminfo_text", lambda: meminfo)


def _heap_mb(node_options: str) -> int:
    (flag,) = [t for t in node_options.split() if t.startswith("--max-old-space-size=")]
    return int(flag.partition("=")[2])


class TestWebBuildLimits:
    def test_heap_and_thread_caps_applied(self, monkeypatch):
        monkeypatch.delenv("NODE_OPTIONS", raising=False)
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        monkeypatch.delenv("HERMES_WEB_BUILD_MAX_OLD_SPACE_SIZE", raising=False)
        monkeypatch.delenv("HERMES_WEB_BUILD_THREADS", raising=False)
        env = web_build_limits({})
        assert "--max-old-space-size=" in env["NODE_OPTIONS"]
        assert int(env["RAYON_NUM_THREADS"]) >= 1

    def test_existing_user_node_options_preserved(self, monkeypatch):
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        env = web_build_limits({"NODE_OPTIONS": "--max-old-space-size=8192"})
        assert env["NODE_OPTIONS"].count("--max-old-space-size=") == 1
        assert "8192" in env["NODE_OPTIONS"]

    def test_existing_user_rayon_threads_preserved(self, monkeypatch):
        monkeypatch.delenv("NODE_OPTIONS", raising=False)
        env = web_build_limits({"RAYON_NUM_THREADS": "6"})
        assert env["RAYON_NUM_THREADS"] == "6"
        # heap cap still applied
        assert "--max-old-space-size=" in env["NODE_OPTIONS"]

    def test_explicit_env_overrides_win(self, monkeypatch):
        monkeypatch.delenv("NODE_OPTIONS", raising=False)
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        env = web_build_limits({
            "HERMES_WEB_BUILD_MAX_OLD_SPACE_SIZE": "3072",
            "HERMES_WEB_BUILD_THREADS": "1",
        })
        assert "--max-old-space-size=3072" in env["NODE_OPTIONS"]
        assert env["RAYON_NUM_THREADS"] == "1"

    def test_cgroup_limit_shrinks_heap_floor(self, monkeypatch):
        monkeypatch.delenv("NODE_OPTIONS", raising=False)
        monkeypatch.setattr(
            "hermes_cli.web_build_limits._cgroup_memory_limit_mb", lambda: 2048
        )
        env = web_build_limits({})
        # 75% of a 2GB container, above the GC-thrash floor but below the
        # unconstrained ceiling.
        assert "--max-old-space-size=1536" in env["NODE_OPTIONS"]

    def test_bounded_threads_half_cores(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.web_build_limits._available_cores", lambda: 16)
        env = web_build_limits({})
        assert env["RAYON_NUM_THREADS"] == "8"  # clamped ceiling

        monkeypatch.setattr("hermes_cli.web_build_limits._available_cores", lambda: 4)
        env = web_build_limits({})
        assert env["RAYON_NUM_THREADS"] == "2"  # half of 4 — the issue's host

        monkeypatch.setattr("hermes_cli.web_build_limits._available_cores", lambda: 1)
        env = web_build_limits({})
        assert env["RAYON_NUM_THREADS"] == "1"  # single-core host never gets 0

    def test_apply_updates_env_in_place(self, monkeypatch):
        monkeypatch.delenv("NODE_OPTIONS", raising=False)
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        env = {"PATH": "/bin"}
        apply_web_build_limits(env)
        assert "--max-old-space-size=" in env["NODE_OPTIONS"]
        assert env["RAYON_NUM_THREADS"].isdigit()
        assert env["PATH"] == "/bin"

    def test_light_mode_tightens_caps(self, monkeypatch):
        monkeypatch.delenv("NODE_OPTIONS", raising=False)
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)
        monkeypatch.delenv("HERMES_WEB_BUILD_MAX_OLD_SPACE_SIZE", raising=False)
        monkeypatch.delenv("HERMES_WEB_BUILD_THREADS", raising=False)
        monkeypatch.setattr("hermes_cli.web_build_limits._available_cores", lambda: 8)
        env = web_build_limits({"HERMES_WEB_BUILD_LIGHT": "1"})
        assert "--max-old-space-size=1024" in env["NODE_OPTIONS"]
        assert env["RAYON_NUM_THREADS"] == "1"


class TestBuildSourceWebPropagatesLimits:
    """The install/update/``hermes dashboard`` build path must apply the caps."""

    def test_build_source_web_applies_limits(self, tmp_path, monkeypatch):
        from hermes_cli import source_build

        captured: dict = {}

        def fake_run_source_script(project_root, script, *args, env, label):
            captured["env"] = env
            captured["label"] = label

        monkeypatch.setattr(source_build, "run_source_script", fake_run_source_script)
        monkeypatch.delenv("NODE_OPTIONS", raising=False)
        monkeypatch.delenv("RAYON_NUM_THREADS", raising=False)

        env = {"PATH": "/bin"}
        source_build.build_source_web(tmp_path, env=env)
        assert "--max-old-space-size=" in captured["env"]["NODE_OPTIONS"]
        assert captured["env"]["RAYON_NUM_THREADS"].isdigit()
        # original env keys untouched
        assert captured["env"]["PATH"] == "/bin"
        assert captured["label"] == "Building the web UI"

    def test_uncontained_small_host_heap_fits_available_memory(
        self, tmp_path, monkeypatch
    ):
        from hermes_cli import source_build

        captured: dict = {}
        monkeypatch.setattr(
            source_build,
            "run_source_script",
            lambda project_root, script, *args, env, label: captured.update(env=env),
        )
        _uncontained_host(monkeypatch, _SMALL_VPS_MEMINFO)

        source_build.build_source_web(tmp_path, env={"PATH": "/bin"})

        heap_mb = _heap_mb(captured["env"]["NODE_OPTIONS"])
        assert _MIN_WEB_BUILD_HEAP_MB <= heap_mb < _SMALL_VPS_AVAILABLE_MB


class TestUncontainedHostHeapBounds:
    @pytest.mark.parametrize(
        ("meminfo", "expected"),
        [
            pytest.param(
                "MemTotal: 8388608 kB\nMemAvailable: 0 kB\n",
                _MIN_WEB_BUILD_HEAP_MB,
                id="nothing-available-gets-the-floor",
            ),
            pytest.param(
                "MemTotal: 134217728 kB\nMemAvailable: 67108864 kB\n",
                _MAX_WEB_BUILD_HEAP_MB,
                id="large-host-keeps-the-ceiling",
            ),
            pytest.param(
                None, _MAX_WEB_BUILD_HEAP_MB, id="no-procfs-keeps-the-ceiling"
            ),
        ],
    )
    def test_heap_stays_within_bounds(self, monkeypatch, meminfo, expected):
        _uncontained_host(monkeypatch, meminfo)

        assert _heap_mb(web_build_limits({})["NODE_OPTIONS"]) == expected


class TestPackageJsonBuildScriptCapped:
    """``scripts/build/web.mjs`` must carry the caps for direct npm runs."""

    def test_build_script_has_heap_and_thread_caps(self):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        build_script = (repo_root / "scripts/build/web.mjs").read_text(encoding="utf-8")
        assert "--max-old-space-size=" in build_script
        assert "RAYON_NUM_THREADS" in build_script
        # caps apply on direct invocation before vite is imported
        assert "isMain(import.meta.url)" in build_script
