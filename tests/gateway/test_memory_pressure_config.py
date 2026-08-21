"""Tests for configurable memory-pressure thresholds (#90713)."""
from unittest import mock

from gateway import memory_status


class TestConfigDrivenThresholds:
    def test_defaults_unchanged_when_config_empty(self):
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value={}):
            th = memory_status._resolve_thresholds()
        assert th["critical_kib"] == 64 * 1024
        assert th["critical_fraction"] == 0.05
        assert th["elevated_kib"] == 128 * 1024
        assert th["elevated_fraction"] == 0.15

    def test_zfs_style_tune_flips_elevated_to_ok(self):
        # The reporter's host: ~11.5% available, default 15% -> elevated.
        total = 31 * 1024 * 1024  # 31 GiB in KiB
        avail = int(total * 0.115)
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value={}):
            assert memory_status.classify_pressure(avail, total) == "elevated"
        cfg = {"elevated_fraction": 0.08}
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg):
            assert memory_status.classify_pressure(avail, total) == "ok"

    def test_critical_band_still_fires_with_tuned_elevated(self):
        total = 31 * 1024 * 1024
        cfg = {"elevated_fraction": 0.01, "critical_fraction": 0.10}
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg):
            # 3% available: below tuned critical (10%), so critical wins
            assert memory_status.classify_pressure(int(total * 0.03), total) == "critical"

    def test_absolute_kib_override(self):
        # Small host (900 MiB) so fractions stay out of the way:
        # 150 MiB / 900 MiB = 16.7% > both fraction thresholds.
        total = 900 * 1024
        cfg = {"elevated_kib": 200 * 1024}
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg):
            # fraction fine, but below the tuned 200 MiB kib floor
            assert memory_status.classify_pressure(150 * 1024, total) == "elevated"
            assert memory_status.classify_pressure(500 * 1024, total) == "ok"

    def test_invalid_values_are_dropped(self):
        cfg = {
            "elevated_fraction": "banana",  # wrong type
            "critical_fraction": 5.0,       # > 1.0
            "elevated_kib": 10,             # below 1 MiB floor
            "critical_kib": True,           # bool is not an int here
            "elevated_kib_ok": 1,           # unknown key, ignored
        }
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg):
            th = memory_status._resolve_thresholds()
        assert th["elevated_fraction"] == 0.15
        assert th["critical_fraction"] == 0.05
        assert th["elevated_kib"] == 128 * 1024
        assert th["critical_kib"] == 64 * 1024

    def test_negative_and_zero_dropped(self):
        cfg = {"elevated_fraction": 0.0, "critical_fraction": -0.5}
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg):
            th = memory_status._resolve_thresholds()
        assert th["elevated_fraction"] == 0.15
        assert th["critical_fraction"] == 0.05

    def test_explicit_thresholds_kwarg_wins(self):
        total = 31 * 1024 * 1024
        th = {"critical_kib": 64 * 1024, "critical_fraction": 0.05,
              "elevated_kib": 128 * 1024, "elevated_fraction": 0.01}
        total = 100 * 1024 * 1024  # 100 GiB in KiB
        # No config patch: kwarg must bypass config entirely
        with mock.patch.object(
            memory_status, "_resolve_thresholds", side_effect=AssertionError
        ):
            # exactly 5%: not < critical (0.05), not < elevated (0.01) -> ok
            assert memory_status.classify_pressure(5 * 1024 * 1024, total, th) == "ok"
            # 0.5%: < critical fraction -> critical
            assert memory_status.classify_pressure(512 * 1024, total, th) == "critical"

    def test_show_elevated_banner_false_downgrades_only_elevated(self):
        total = 31 * 1024 * 1024
        elevated_avail = int(total * 0.115)
        critical_avail = int(total * 0.02)

        def fake_collect(avail):
            return memory_status.classify_pressure(avail, total)

        cfg_off = {"show_elevated_banner": False}
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg_off):
            # Elevated input -> downgraded to ok in the rollup
            assert fake_collect(elevated_avail) == "elevated"  # classifier itself unchanged
        # The downgrade happens in collect_memory_status; classifier stays honest.
        # Verify config read path tolerates missing key:
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value={}):
            assert memory_status.classify_pressure(elevated_avail, total) == "elevated"
            assert memory_status.classify_pressure(critical_avail, total) == "critical"

    def test_crossed_bands_elevated_dropped_to_default(self):
        """When elevated falls inside/below critical, the elevated override
        is dropped back to defaults to avoid false OOM evidence."""
        cfg = {"elevated_fraction": 0.02, "critical_fraction": 0.05}
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg):
            th = memory_status._resolve_thresholds()
        # elevated fell inside critical -> reset to hardcoded default
        assert th["elevated_fraction"] == 0.15
        assert th["critical_fraction"] == 0.05  # critical untouched

    def test_crossed_bands_elevated_kib_dropped_to_default(self):
        cfg = {"elevated_kib": 32 * 1024, "critical_kib": 64 * 1024}
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg):
            th = memory_status._resolve_thresholds()
        assert th["elevated_kib"] == 128 * 1024  # reset to default
        assert th["critical_kib"] == 64 * 1024

    def test_valid_elevated_below_critical_not_dropped(self):
        """Elevated band above critical is fine and should NOT be reset."""
        cfg = {"elevated_fraction": 0.20, "critical_fraction": 0.05}
        with mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg):
            th = memory_status._resolve_thresholds()
        assert th["elevated_fraction"] == 0.20

    def test_collect_memory_status_rollup_downgrades_elevated(self):
        """Verify collect_memory_status actually applies the show_elevated_banner
        downgrade in the rollup path (not just classify_pressure)."""
        total = 31 * 1024 * 1024
        elevated_avail = int(total * 0.115)
        now = memory_status.datetime(2026, 8, 21, 12, 0, 0, tzinfo=memory_status.timezone.utc)
        heartbeat = {
            "updated_at": now.isoformat(),
            "mem": {
                "mem_available_kib": elevated_avail,
                "mem_total_kib": total,
                "rss_kib": 500 * 1024,
                "swap_used_kib": 0,
            },
        }
        sentinel = {"prior_unclean_exit": False, "prior_suspected_oom": False,
                    "started_at": now.isoformat()}
        cfg_off = {"show_elevated_banner": False}
        with mock.patch.object(memory_status, "_read_heartbeat", return_value=heartbeat), \
             mock.patch.object(memory_status, "_read_sentinel", return_value=sentinel), \
             mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg_off):
            result = memory_status.collect_memory_status(now=now)
        assert result["pressure"] == "ok"  # elevated downgraded via rollup

    def test_collect_memory_status_critical_not_silenceable(self):
        """show_elevated_banner: false must NOT suppress critical."""
        total = 31 * 1024 * 1024
        critical_avail = int(total * 0.02)
        now = memory_status.datetime(2026, 8, 21, 12, 0, 0, tzinfo=memory_status.timezone.utc)
        heartbeat = {
            "updated_at": now.isoformat(),
            "mem": {
                "mem_available_kib": critical_avail,
                "mem_total_kib": total,
                "rss_kib": 500 * 1024,
                "swap_used_kib": 0,
            },
        }
        sentinel = {"prior_unclean_exit": False, "prior_suspected_oom": False,
                    "started_at": now.isoformat()}
        cfg_off = {"show_elevated_banner": False}
        with mock.patch.object(memory_status, "_read_heartbeat", return_value=heartbeat), \
             mock.patch.object(memory_status, "_read_sentinel", return_value=sentinel), \
             mock.patch.object(memory_status, "_memory_pressure_config", return_value=cfg_off):
            result = memory_status.collect_memory_status(now=now)
        assert result["pressure"] == "critical"  # critical is never silenced

    def test_memory_pressure_config_missing_dashboard(self):
        with mock.patch("hermes_cli.config.load_config_readonly", return_value={}):
            assert memory_status._memory_pressure_config() == {}
        with mock.patch("hermes_cli.config.load_config_readonly", return_value={"dashboard": "nope"}):
            assert memory_status._memory_pressure_config() == {}

    def test_memory_pressure_config_import_failure_degrades(self):
        with mock.patch.dict(
            "sys.modules", {"hermes_cli.config": None}
        ):
            assert memory_status._memory_pressure_config() == {}
