#!/usr/bin/env python3
"""TP2w — write-path E2E against the dead-localhost mocks.

Every destructive sub is invoked EXPLICITLY (the mock is
the sole victim) and its exact request shape is asserted from the JSONL
request log — bodies are normative from the service API shapes (design
). Also asserts secrets never appear in argv-visible output.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402
from harness import make_env, post_bodies, requests_to, run_cli  # noqa: E402

PORT = {"sonarr": 8989, "radarr": 7878, "lidarr": 8686, "prowlarr": 9696,
        "bazarr": 6767, "overseerr": 5055, "jellyfin": 8096}


def setUpModule():
    harness.setup_sandbox()
    harness.start_mock()


def tearDownModule():
    harness.stop_mock()
    harness.teardown_sandbox()


class TestArrWrites(unittest.TestCase):
    def test_sonarr_add_posts_v1_body(self):
        proc = run_cli(["sonarr", "add", "mock series"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        bodies = post_bodies(PORT["sonarr"], "/api/v3/series")
        self.assertTrue(bodies, "POST /series must reach the mock")
        body = bodies[-1]
        self.assertEqual(body["title"], "Mock Series Alpha")
        self.assertEqual(body["tvdbId"], 123456)
        self.assertEqual(body["qualityProfileId"], 1)
        self.assertEqual(body["rootFolderPath"], "/tv-mock")
        self.assertTrue(body["seasonFolder"])
        self.assertTrue(body["monitored"])
        self.assertEqual(body["addOptions"],
                         {"searchForMissingEpisodes": True})

    def test_sonarr_add_honours_flags(self):
        proc = run_cli(["sonarr", "add", "mock series", "--quality", "7",
                        "--folder", "/tv-other"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["sonarr"], "/api/v3/series")[-1]
        self.assertEqual(body["qualityProfileId"], 7)
        self.assertEqual(body["rootFolderPath"], "/tv-other")

    def test_radarr_add_posts_v1_body(self):
        proc = run_cli(["radarr", "add", "mock movie"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["radarr"], "/api/v3/movie")[-1]
        self.assertEqual(body["title"], "Mock Movie Alpha")
        self.assertEqual(body["tmdbId"], 100001)
        self.assertEqual(body["year"], 2021)
        self.assertEqual(body["qualityProfileId"], 1)
        self.assertEqual(body["rootFolderPath"], "/movies-mock")
        self.assertTrue(body["monitored"])
        self.assertEqual(body["minimumAvailability"], "released")
        self.assertEqual(body["addOptions"], {"searchForMovie": True})

    def test_remove_deletes_only_when_invoked(self):
        before = requests_to(PORT["sonarr"], "/api/v3/series", "DELETE")
        proc = run_cli(["sonarr", "remove", "1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        after = requests_to(PORT["sonarr"], "/api/v3/series", "DELETE")
        self.assertEqual(len(after), len(before) + 1,
                         "DELETE fires exactly when the sub is invoked")
        rec = after[-1]
        self.assertEqual(rec["path"], "/api/v3/series/1")
        self.assertIn("deleteFiles=false", rec["query"])

    def test_radarr_remove(self):
        proc = run_cli(["radarr", "remove", "1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        recs = requests_to(PORT["radarr"], "/api/v3/movie", "DELETE")
        self.assertEqual(recs[-1]["path"], "/api/v3/movie/1")

    def test_rename_posts_command(self):
        proc = run_cli(["sonarr", "rename", "1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["sonarr"], "/api/v3/command")[-1]
        self.assertEqual(body, {"name": "RenameSeries", "seriesIds": [1]})
        proc = run_cli(["radarr", "rename", "2"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["radarr"], "/api/v3/command")[-1]
        self.assertEqual(body, {"name": "RenameMovie", "movieIds": [2]})

    def test_refresh_posts_command(self):
        proc = run_cli(["sonarr", "refresh", "5"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["sonarr"], "/api/v3/command")[-1]
        self.assertEqual(body, {"name": "RefreshSeries", "seriesId": 5})
        proc = run_cli(["radarr", "refresh"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["radarr"], "/api/v3/command")[-1]
        self.assertEqual(body, {"name": "RefreshMovie"})


class TestProwlarrWrites(unittest.TestCase):
    def test_add_app_posts_v1_shape_from_env(self):
        proc = run_cli(["prowlarr", "add-app", "--type", "sonarr"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("test-sonarr-key", proc.stdout + proc.stderr)
        body = post_bodies(PORT["prowlarr"], "/api/v1/applications")[-1]
        self.assertEqual(body["name"], "Sonarr")
        self.assertEqual(body["implementation"], "Sonarr")
        self.assertEqual(body["implementationName"], "Sonarr")
        self.assertEqual(body["configContract"], "SonarrSettings")
        self.assertEqual(body["syncLevel"], "fullSync")
        fields = {f["name"]: f["value"] for f in body["fields"]}
        self.assertEqual(fields["baseUrl"], "http://127.0.0.1:8989")
        self.assertEqual(fields["prowlarrUrl"], "http://127.0.0.1:9696")
        self.assertEqual(fields["apiKey"], "test-sonarr-key")
        self.assertEqual(fields["syncCategories"], [])

    def test_add_app_accepts_exactly_three_impls(self):
        for impl in ("sonarr", "radarr", "lidarr"):
            proc = run_cli(["prowlarr", "add-app", "--type", impl])
            self.assertEqual(proc.returncode, 0, proc.stderr)
        proc = run_cli(["prowlarr", "add-app", "--type", "readarr"])
        self.assertEqual(proc.returncode, 2)     # argparse rejects: dropped

    def test_sync_posts_indexer_sync_command(self):
        proc = run_cli(["prowlarr", "sync"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["prowlarr"], "/api/v1/command")[-1]
        self.assertEqual(body, {"name": "ApplicationsIndexerSync"})

    def test_test_all_posts_testall(self):
        proc = run_cli(["prowlarr", "test"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        recs = requests_to(PORT["prowlarr"], "/api/v1/indexer/testall",
                           "POST")
        self.assertTrue(recs, "POST /indexer/testall must reach the mock")

    def test_test_single_fetches_then_posts(self):
        proc = run_cli(["prowlarr", "test", "1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        got = requests_to(PORT["prowlarr"], "/api/v1/indexer/1", "GET")
        posted = requests_to(PORT["prowlarr"], "/api/v1/indexer/test",
                             "POST")
        self.assertTrue(got and posted)
        body = post_bodies(PORT["prowlarr"], "/api/v1/indexer/test")[-1]
        self.assertIn("implementation", body)


class TestBazarrOverseerrWrites(unittest.TestCase):
    def test_bazarr_search_episode(self):
        proc = run_cli(["bazarr", "search", "--episode", "4"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["bazarr"], "/api/episodes/search")[-1]
        self.assertEqual(body, {"seriesId": 4})

    def test_bazarr_search_movie(self):
        proc = run_cli(["bazarr", "search", "--movie", "9"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["bazarr"], "/api/movies/search")[-1]
        self.assertEqual(body, {"radarrId": 9})

    def test_bazarr_search_requires_a_target(self):
        proc = run_cli(["bazarr", "search"])
        self.assertEqual(proc.returncode, 1)
        self.assertIn("--episode", proc.stderr)

    def test_overseerr_approve_deny(self):
        proc = run_cli(["overseerr", "approve", "101"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        recs = requests_to(PORT["overseerr"], "/api/v1/request/101/approve",
                           "POST")
        self.assertTrue(recs)
        proc = run_cli(["overseerr", "deny", "102", "--reason",
                        "not available"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        body = post_bodies(PORT["overseerr"], "/api/v1/request/102/decline"
                           )[-1]
        self.assertEqual(body, {"message": "not available"})

    def test_keys_never_in_output(self):
        for argv in (["prowlarr", "add-app", "--type", "radarr"],
                     ["sonarr", "add", "mock series"],
                     ["overseerr", "approve", "101"]):
            proc = run_cli(argv)
            combined = proc.stdout + proc.stderr
            for key in harness.DUMMY_CREDS.values():
                self.assertNotIn(key, combined,
                                 "key leaked by %r" % (argv,))


if __name__ == "__main__":
    unittest.main()
