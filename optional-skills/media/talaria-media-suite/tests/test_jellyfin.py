#!/usr/bin/env python3
"""TP2j — Jellyfin auth + analytics contract.

(i)   every jellyfin request in the log carries the MediaBrowser
      Authorization header — zero api_key/ApiKey query params or
      X-Emby-Token headers anywhere (the negative traps would have 400'd);
(ii)  wrong key -> 401 -> stderr reads "jellyfin rejected the API key",
      key value absent from stdout AND stderr;
(iii) /System/Info/Public is fetched without auth by discover;
(iv)  the 5 ⚠ subs print the approximation banner (P8) and --json
      carries "approx": true;
(v)   the /emby/… route prefix 404s on the mock (the skill never emits it).
"""

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402
from harness import make_env, mock_records, requests_to, run_cli  # noqa: E402

JF = 8096
APPROX_SUBS = ["history", "most-watched", "user-stats", "play-totals",
               "popular-genres"]


def setUpModule():
    harness.setup_sandbox()
    harness.start_mock()


def tearDownModule():
    harness.stop_mock()
    harness.teardown_sandbox()


def _no_opener_proxies():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


class TestAuthContract(unittest.TestCase):
    def test_every_jellyfin_request_uses_mediabrowser(self):
        run_cli(["jellyfin", "sessions"])
        run_cli(["jellyfin", "counts"])
        run_cli(["jellyfin", "history"])
        # auth-required routes: every request must carry MediaBrowser auth
        authed = [r for r in requests_to(port=JF)
                  if r.get("path") not in ("/System/Info/Public",
                                           "/System/Ping")]
        self.assertTrue(authed, "jellyfin requests must exist")
        for rec in authed:
            self.assertEqual(rec.get("auth"), "mediabrowser",
                             "non-MediaBrowser auth on %s"
                             % rec.get("path"))
            self.assertTrue(rec.get("key_ok"))
        # and the public routes must be the ONLY unauthenticated ones
        for rec in requests_to(port=JF):
            if rec.get("path") in ("/System/Info/Public", "/System/Ping"):
                self.assertNotEqual(rec.get("auth"), "mediabrowser")

    def test_no_legacy_channels_anywhere(self):
        for rec in mock_records():
            self.assertNotIn("violation", rec,
                             "legacy channel emitted: %r" % rec)
            self.assertNotIn("x-emby-token", rec.get("headers", []))
            self.assertNotIn("api_key", (rec.get("query") or ""))
            self.assertNotIn("apikey=", (rec.get("query") or "").lower())

    def test_wrong_key_auth_error_without_leak(self):
        bad = "zzfake-jf-key-value-zz"
        proc = run_cli(["jellyfin", "sessions"],
                       env=make_env(JELLYFIN_KEY=bad))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("jellyfin rejected the API key", proc.stderr)
        self.assertNotIn(bad, proc.stdout)
        self.assertNotIn(bad, proc.stderr)

    def test_discover_uses_public_info_without_auth(self):
        proc = run_cli(["discover", "127.0.0.1"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        public = [r for r in requests_to(port=JF, path="/System/Info/Public")]
        self.assertTrue(public, "discover must probe /System/Info/Public")
        for rec in public:
            self.assertNotEqual(rec.get("auth"), "mediabrowser")

    def test_emby_prefix_404s(self):
        opener = _no_opener_proxies()
        req = urllib.request.Request("http://127.0.0.1:8096/emby/System/Info",
                                     headers={"Authorization":
                                              'MediaBrowser Token="x"'})
        try:
            with opener.open(req, timeout=5) as resp:
                status = resp.status
        except urllib.error.HTTPError as exc:
            status = exc.code
        self.assertEqual(status, 404)


class TestReadSubs(unittest.TestCase):
    def test_sessions_renders_active_and_idle(self):
        proc = run_cli(["jellyfin", "sessions"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Mock Movie Alpha", proc.stdout)
        self.assertIn("MockUserOne", proc.stdout)
        self.assertIn("idle", proc.stdout)
        self.assertIn("Transcode", proc.stdout)
        self.assertIn("25%", proc.stdout)

    def test_libraries_counts_latest(self):
        proc = run_cli(["jellyfin", "libraries"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # live-shaped 5-library server (17/10): every library must render
        for name in ("Mock Movies", "Mock Shows", "Mock Music",
                     "Mock Collections", "Mock Unorganized Shows"):
            self.assertIn(name, proc.stdout)
        proc = run_cli(["jellyfin", "counts"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # The fixture serves the live 12.2.0 shape: schema-native
        # PascalCase keys (MovieCount…). A camelCase-only reader prints
        # "No items" here — the 17/10 live bug this test guards against.
        self.assertNotIn("No items", proc.stdout)
        for kind, value in (("movieCount", "12"), ("seriesCount", "3"),
                            ("episodeCount", "40"), ("artistCount", "2"),
                            ("albumCount", "8"), ("songCount", "90"),
                            ("boxSetCount", "1"), ("itemCount", "156")):
            self.assertIn(kind, proc.stdout)
            self.assertIn(value, proc.stdout)
        for zero_kind in ("musicVideoCount", "bookCount"):
            self.assertNotIn(zero_kind, proc.stdout)
        proc = run_cli(["jellyfin", "latest"])
        self.assertIn("Mock Movie Alpha", proc.stdout)

    def test_resume_nextup_users_activity(self):
        proc = run_cli(["jellyfin", "resume"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Mock Movie Alpha", proc.stdout)
        self.assertIn("30:40", proc.stdout)          # 18.4e9 ticks
        proc = run_cli(["jellyfin", "nextup"])
        self.assertIn("Mock Episode Four", proc.stdout)
        proc = run_cli(["jellyfin", "users"])
        self.assertIn("MockUserOne", proc.stdout)
        self.assertIn("MockUserTwo", proc.stdout)
        proc = run_cli(["jellyfin", "activity"])
        self.assertIn("has played", proc.stdout)

    def test_default_user_resolution(self):
        proc = run_cli(["jellyfin", "resume"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        users_calls = requests_to(port=JF, path="/Users")
        self.assertTrue(users_calls, "default user resolves via /Users")


class TestLiveUserContextContract(unittest.TestCase):
    """Live 12.2.0 (17/10): user-scoped /Items queries require userId.

    The sweep found latest/history/most-watched/play-totals HTTP 400 on
    live while the offline suite was green: the mock served fixtures for
    userId-less queries the real server rejects. /Items/Latest requires
    userId always; bare /Items requires it when user-scoped params ride
    the query (sortBy=DatePlayed/PlayCount, recursive+filters=IsPlayed).
    The CONTRACT_400S pins keep the mock honest; the CLI assertions keep
    talaria emitting user context on every user-scoped query.
    """

    UID_ONE = "11aa22bb33cc44dd11aa22bb33cc44dd"   # first /Users fixture id
    UID_TWO = "22bb33cc44dd11aa22bb33cc44dd11aa"

    def _last_query(self, path):
        recs = requests_to(port=JF, path=path, method="GET")
        self.assertTrue(recs, "no request logged to %s" % path)
        return recs[-1].get("query") or ""

    def test_user_scoped_subs_carry_user_context(self):
        for sub, path in (("latest", "/Items/Latest"),
                          ("history", "/Items"),
                          ("most-watched", "/Items"),
                          ("play-totals", "/Items"),
                          ("popular-genres", "/Items")):
            proc = run_cli(["jellyfin", sub])
            self.assertEqual(proc.returncode, 0,
                             "%s: %s" % (sub, proc.stderr))
            self.assertIn("userId=", self._last_query(path),
                          "%s must send userId (live 12.2.0 400s without)"
                          % sub)

    def test_latest_resolves_default_first_user(self):
        proc = run_cli(["jellyfin", "latest"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Mock Movie Alpha", proc.stdout)
        self.assertIn("userId=%s" % self.UID_ONE,
                      self._last_query("/Items/Latest"))

    def test_latest_accepts_explicit_user(self):
        proc = run_cli(["jellyfin", "latest", "--user", "MockUserTwo"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("userId=%s" % self.UID_TWO,
                      self._last_query("/Items/Latest"))

    def test_history_and_most_watched_render(self):
        for sub in ("history", "most-watched"):
            proc = run_cli(["jellyfin", sub])
            self.assertEqual(proc.returncode, 0,
                             "%s: %s" % (sub, proc.stderr))
            self.assertIn("Mock Movie Alpha", proc.stdout)

    def test_play_totals_renders_per_library(self):
        proc = run_cli(["jellyfin", "play-totals"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Mock Movies", proc.stdout)

    def test_user_stats_keeps_per_user_context(self):
        before = len(mock_records())   # only inspect THIS run's requests
        proc = run_cli(["jellyfin", "user-stats"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # per-user aggregation sends each /Users id, never a bare query
        # (exact path match — /Items/Counts is a different, param-less
        # endpoint; the boundary probes in this class also hit bare
        # /Items on purpose and must not be swept in)
        legs = [r for r in mock_records()[before:]
                if r.get("port") == JF and r.get("method") == "GET"
                and r.get("path") == "/Items"]
        self.assertTrue(legs, "user-stats must query /Items")
        for rec in legs:
            self.assertIn("userId=", rec.get("query") or "",
                          "user-stats leg without userId: %r" % rec)

    def _probe(self, path):
        opener = _no_opener_proxies()
        req = urllib.request.Request(
            "http://127.0.0.1:8096" + path,
            headers={"Authorization":
                     'MediaBrowser Token="test-jellyfin-key"'})
        try:
            with opener.open(req, timeout=5) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            return exc.code

    def test_mock_pins_latest_requires_userid(self):
        self.assertEqual(self._probe("/Items/Latest?limit=10"), 400)
        self.assertEqual(
            self._probe("/Items/Latest?limit=10&userId=%s" % self.UID_ONE),
            200)

    def test_mock_pins_userdata_sorts_require_userid(self):
        for sort_by in ("DatePlayed", "PlayCount"):
            self.assertEqual(
                self._probe("/Items?sortBy=%s&sortOrder=Descending"
                            % sort_by), 400)
            self.assertEqual(
                self._probe("/Items?sortBy=%s&userId=%s"
                            % (sort_by, self.UID_ONE)), 200)

    def test_mock_pins_recursive_isplayed_requires_userid(self):
        self.assertEqual(
            self._probe("/Items?recursive=true&filters=IsPlayed&limit=1000"),
            400)
        self.assertEqual(
            self._probe("/Items?recursive=true&filters=IsPlayed&limit=1000"
                        "&userId=%s" % self.UID_ONE), 200)

    def test_mock_keeps_non_user_queries_open(self):
        # the boundary is user-scoped only — these stayed 200 live and
        # must NOT be vetoed by the contract table
        self.assertEqual(self._probe("/Items"), 200)
        self.assertEqual(self._probe("/Items?sortBy=SortName"), 200)
        self.assertEqual(self._probe("/Items?recursive=true"), 200)
        self.assertEqual(self._probe("/Items?filters=IsPlayed"), 200)
        self.assertEqual(self._probe("/Items?limit=20"), 200)
        self.assertEqual(self._probe("/Items?fields=UserData"), 200)


class TestApproxContract(unittest.TestCase):
    def test_approx_subs_print_banner(self):
        for sub in APPROX_SUBS:
            argv = ["jellyfin", sub]
            if sub in ("history", "most-watched"):
                argv += ["--limit", "5"]
            proc = run_cli(argv)
            self.assertEqual(proc.returncode, 0,
                             "%s: %s" % (sub, proc.stderr))
            first = proc.stdout.splitlines()[0]
            self.assertIn("approximate", first,
                          "%s must lead with the banner" % sub)
            self.assertIn("play-state", first)

    def test_approx_json_carries_marker(self):
        for sub in APPROX_SUBS:
            proc = run_cli(["--json", "jellyfin", sub])
            self.assertEqual(proc.returncode, 0,
                             "%s: %s" % (sub, proc.stderr))
            payload = json.loads(proc.stdout)
            self.assertIs(payload.get("approx"), True,
                          "%s --json must carry approx: true" % sub)

    def test_exact_subs_do_not_carry_banner(self):
        for sub in ("sessions", "counts", "latest", "users"):
            proc = run_cli(["jellyfin", sub])
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertNotIn("approximate", proc.stdout.splitlines()[0])


class TestWrites(unittest.TestCase):
    def test_refresh_all_posts_library_refresh(self):
        proc = run_cli(["jellyfin", "refresh"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        recs = requests_to(JF, "/Library/Refresh", "POST")
        self.assertTrue(recs)

    def test_refresh_item_posts_items_refresh(self):
        proc = run_cli(["jellyfin", "refresh", "--item", "88cd"])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        recs = requests_to(JF, "/Items/88cd/Refresh", "POST")
        self.assertTrue(recs)
        self.assertIn("metadataRefreshMode=Default", recs[-1]["query"])
        self.assertIn("imageRefreshMode=Default", recs[-1]["query"])


if __name__ == "__main__":
    unittest.main()
