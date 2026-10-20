#!/usr/bin/env python3
"""TP2 — read-path E2E against the dead-localhost mocks.

Offline by construction: dead-proxy trap + loopback mocks only. Covers
per-service reads, cross-service subs (library/ops/dashboard), and the
PII gate on default human output (no emails even though fixtures carry
one). Jellyfin-specific rows live in test_jellyfin.py.
"""

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402
from harness import make_env, run_cli  # noqa: E402


def setUpModule():
    harness.setup_sandbox()
    harness.start_mock()


def tearDownModule():
    harness.stop_mock()
    harness.teardown_sandbox()


class TestReads(unittest.TestCase):
    def _ok(self, proc, *needles, json_mode=False):
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        haystack = proc.stdout if json_mode else proc.stdout + proc.stderr
        for needle in needles:
            self.assertIn(needle, haystack,
                          "missing %r in output:\n%s" % (needle,
                                                         proc.stdout[:400]))
        return proc

    # ---- per-service reads -------------------------------------------
    def test_sonarr_queue_and_calendar(self):
        self._ok(run_cli(["sonarr", "queue"]), "Mock Series Alpha")
        self._ok(run_cli(["sonarr", "calendar"]), "Mock Episode")

    def test_sonarr_wanted(self):
        self._ok(run_cli(["sonarr", "wanted"]), "Mock Episode Two",
                 "Total missing: 2")

    def test_radarr_lookup(self):
        self._ok(run_cli(["radarr", "lookup", "mock movie"]),
                 "Mock Movie Alpha", "100001")

    def test_lidarr_lookup(self):
        self._ok(run_cli(["lidarr", "lookup", "mock artist"]),
                 "Mock Artist", "Mock Album")

    def test_lidarr_queue(self):
        self._ok(run_cli(["lidarr", "queue"]), "Mock Album")

    def test_prowlarr_indexers_stats_search(self):
        self._ok(run_cli(["prowlarr", "indexers"]), "Mock Indexer One",
                 "3 total", "2 enabled")
        self._ok(run_cli(["prowlarr", "stats"]), "Mock Indexer One")
        self._ok(run_cli(["prowlarr", "search", "mock",
                          "--indexers", "1,2"]), "Mock Result Alpha")
        self._ok(run_cli(["prowlarr", "status"]), "1.0.0-mock")
        self._ok(run_cli(["prowlarr", "logs"]), "Mock log line")
        self._ok(run_cli(["prowlarr", "apps"]), "Sonarr", "fullSync")

    def test_bazarr_wanted_languages(self):
        proc = self._ok(run_cli(["bazarr", "wanted"]), "Mock Series Alpha")
        self.assertIn("English", proc.stdout)   # list-of-dicts -> names
        self.assertNotIn("{", proc.stdout)      # never the raw dict repr
        self._ok(run_cli(["bazarr", "languages"]), "en", "de")
        self._ok(run_cli(["bazarr", "history"]), "mockprovider")

    def test_overseerr_list_info(self):
        self._ok(run_cli(["overseerr", "list"]), "Mock Movie Gamma",
                 "Mock User")

    def test_overseerr_list_pending_filter(self):
        self._ok(run_cli(["overseerr", "list", "pending"]),
                 "Mock Series Delta")

    def test_overseerr_info(self):
        self._ok(run_cli(["overseerr", "info", "101"]), "Title:",
                 "Requested by: Mock User")

    def test_overseerr_stats(self):
        self._ok(run_cli(["overseerr", "stats"]), "Total: 2",
                 "Top requesters")

    def test_pii_gate_no_email_in_human_output(self):
        proc = self._ok(run_cli(["overseerr", "list"]), "Mock Movie Gamma")
        self.assertNotIn("mock@users.test", proc.stdout + proc.stderr)

    # ---- cross-service: library --------------------------------------
    def test_library_stats_all(self):
        self._ok(run_cli(["library", "stats"]), "radarr:", "sonarr:",
                 "lidarr:", "movies: 3", "series: 3", "artists: 1")

    def test_library_stats_radarr_only(self):
        proc = self._ok(run_cli(["library", "stats", "radarr"]), "movies: 3")
        self.assertNotIn("sonarr:", proc.stdout)

    def test_library_stats_sonarr_statistics(self):
        # 17/10 live bug: the sonarr leg called bare GET /api/v3/episode,
        # which real Sonarr rejects with 400 "seriesId or episodeIds must
        # be provided". It must aggregate /series statistics instead, and
        # the expected numbers below derive from sonarr__series.json
        # (12/10 + 6/6 + 8/0 episodes/files), NOT sonarr__episodes.json.
        before = len(harness.mock_records())
        proc = self._ok(run_cli(["library", "stats", "sonarr"]),
                        "series: 3", "monitored_series: 2",
                        "episodes: 26", "downloaded_episodes: 16",
                        "missing_episodes: 2", "disk_gb: 65.19")
        self.assertNotIn("error", proc.stdout)
        bare = [r for r in harness.mock_records()[before:]
                if r.get("path") == "/api/v3/episode" and not r.get("query")]
        self.assertEqual(
            bare, [],
            "library stats must never issue a bare GET /api/v3/episode "
            "(real Sonarr 400s it; mock contract now enforces that too)")

    def test_library_missing(self):
        self._ok(run_cli(["library", "missing"]), "Radarr missing movies: 1",
                 "Sonarr missing episodes: 2",
                 "Mock Series Alpha: 2 missing")

    def test_library_disk(self):
        proc = self._ok(run_cli(["library", "disk"]), "/movies-mock",
                        "/tv-mock", "/music-mock")
        self.assertIn("total unknown", proc.stdout)  # totalSpace=0 -> unknown
        self.assertNotIn(" 0 GB total", proc.stdout)  # never a bogus zero

    def test_library_quality(self):
        self._ok(run_cli(["library", "quality"]), "Mock Any", "3")

    def test_library_recent(self):
        self._ok(run_cli(["library", "recent", "radarr", "--days", "3650"]),
                 "Mock Movie")

    def test_library_facets(self):
        self._ok(run_cli(["library", "genres", "radarr"]),
                 "Science Fiction")
        self._ok(run_cli(["library", "years", "radarr"]), "2021")
        self._ok(run_cli(["library", "studios"]), "Mock Studio One")
        self._ok(run_cli(["library", "unmonitored", "radarr"]),
                 "Mock Movie Gamma")
        self._ok(run_cli(["library", "nofiles", "radarr"]),
                 "Mock Movie Beta")

    # ---- cross-service: ops ------------------------------------------
    def test_status_seven_rows(self):
        proc = self._ok(run_cli(["status"]), "Service status")
        for name in ("sonarr", "radarr", "lidarr", "prowlarr", "bazarr",
                     "overseerr", "jellyfin"):
            self.assertIn(name, proc.stdout)
        self.assertEqual(proc.stdout.count("\n") >= 9, True)

    def test_discover_finds_all_seven(self):
        proc = self._ok(run_cli(["discover", "127.0.0.1"]),
                        "Found 7 service(s)", "jellyfin",
                        "Jellyfin Server")
        for name in ("sonarr", "radarr", "lidarr", "prowlarr", "bazarr",
                     "overseerr"):
            self.assertIn(name, proc.stdout)

    def test_setup_prints_export_block(self):
        proc = self._ok(run_cli(["setup", "127.0.0.1"]),
                        "export TALARIA_HOST=127.0.0.1",
                        "Found 7 service(s)")
        self.assertIn("export SONARR_KEY=mock-sonarr-key-from-initialize",
                      proc.stdout)
        self.assertIn("JELLYFIN_KEY", proc.stdout)   # instructions or verify

    def test_queue_snapshot(self):
        proc = self._ok(run_cli(["queue"]), "radarr queue: 1 item(s)",
                        "sonarr queue: 3 item(s)", "lidarr queue: 1 item(s)")
        self.assertIn("Mock Movie Alpha", proc.stdout)

    def test_diagnose_report(self):
        self._ok(run_cli(["diagnose"]), "Diagnostics:", "queue",
                 "/movies-mock", "/tv-mock")

    # ---- artifact: dashboard -----------------------------------------
    def test_dashboard_artifact(self):
        out = harness.SANDBOX / "dash.html"
        proc = run_cli(["dashboard", str(out)])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(out.is_file(), "dashboard file must exist")
        html_text = out.read_text()
        self.assertIn("<svg", html_text)
        self.assertIn("12", html_text)            # movieCount fixture value
        self.assertIn("Mock Movies", html_text)   # VirtualFolders fixture
        self.assertIn("talaria", html_text)
        # latest-adds panel must render rows, not a muted 400: live 12.2.0
        # requires userId on /Items/Latest (17/10 contract pin). The name
        # alone is too weak (sessions panel carries it too) — assert no
        # degraded panel text anywhere in the artifact.
        self.assertNotIn("HTTP 400", html_text)

    def test_json_flag_shapes(self):
        proc = self._ok(run_cli(["--json", "sonarr", "queue"]), "records",
                        json_mode=True)
        proc.stdout.strip()                       # payload is valid JSON
        import json
        payload = json.loads(proc.stdout)
        self.assertIn("records", payload)


class TestMockSonarrContract(unittest.TestCase):
    """The mock must model real-service request contracts, not just serve
    fixtures. Learned from the 17/10 live bug: the offline suite was green
    on `GET /api/v3/episode` with no params because the mock happily
    served the fixture, while real Sonarr 400s — "seriesId or episodeIds
    must be provided". These pins keep the mock honest about that."""

    def _get(self, path):
        req = urllib.request.Request(
            "http://127.0.0.1:8989" + path,
            headers={"X-Api-Key": "test-sonarr-key"})
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}))    # never proxy loopback
        try:
            with opener.open(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_episode_without_params_400s(self):
        status, body = self._get("/api/v3/episode")
        self.assertEqual(status, 400)
        self.assertEqual(body.get("error"), "BadRequest")
        self.assertIn("seriesId or episodeIds", body.get("message", ""))

    def test_episode_with_seriesid_serves_fixture(self):
        status, body = self._get("/api/v3/episode?seriesId=1")
        self.assertEqual(status, 200)
        self.assertIsInstance(body, list)


if __name__ == "__main__":
    unittest.main()
