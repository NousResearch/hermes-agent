"""Unit tests for the Hermes Hub dashboard server (stdlib only).

Run:  python3 -m unittest discover -s apps/dashboard/tests -v
"""

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assistant  # noqa: E402
import ics  # noqa: E402
import indicators as ind  # noqa: E402
import router as router_mod  # noqa: E402
import evolve as evolve_mod  # noqa: E402
import server  # noqa: E402
import telemetry as telemetry_mod  # noqa: E402


RSS_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel>
  <title>Test Feed</title>
  <item>
    <title>First &amp; foremost story</title>
    <link>https://example.org/a</link>
    <description><![CDATA[<p>Some <b>bold</b> text&nbsp;here.</p>]]></description>
    <pubDate>Mon, 30 Jun 2025 10:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Second story</title>
    <link>https://example.org/b</link>
    <description>Plain text</description>
    <pubDate>Mon, 30 Jun 2025 12:00:00 GMT</pubDate>
  </item>
</channel></rss>"""

ATOM_SAMPLE = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry>
    <title>Atom entry one</title>
    <link rel="alternate" href="https://example.org/atom1"/>
    <summary>An atom summary</summary>
    <published>2025-06-30T09:30:00Z</published>
  </entry>
</feed>"""

HTML_SAMPLE = b"""<!doctype html>
<html><head><title>Article Title | Site</title>
<script>var junk = "should not appear";</script></head>
<body>
<nav><p>Navigation junk paragraph that should be skipped entirely by parser.</p></nav>
<article>
<h1>Article Title</h1>
<p>This is the first real paragraph of the article body, long enough to pass the filter.</p>
<p>Second paragraph with more useful content that should also survive extraction here.</p>
</article>
</body></html>"""


class ParseFeedTests(unittest.TestCase):
    def test_rss_parsing(self):
        items = server.parse_feed(RSS_SAMPLE, "Test Feed")
        self.assertEqual(len(items), 2)
        first = items[0]
        self.assertEqual(first["title"], "First & foremost story")
        self.assertEqual(first["url"], "https://example.org/a")
        self.assertEqual(first["source"], "Test Feed")
        self.assertNotIn("<", first["summary"])  # HTML stripped
        self.assertIn("bold", first["summary"])
        self.assertTrue(first["published"].startswith("2025-06-30T10:00:00"))

    def test_atom_parsing(self):
        items = server.parse_feed(ATOM_SAMPLE, "Atom")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["url"], "https://example.org/atom1")
        self.assertEqual(items[0]["summary"], "An atom summary")
        self.assertTrue(items[0]["published"].startswith("2025-06-30T09:30:00"))

    def test_unknown_root_raises(self):
        with self.assertRaises(ValueError):
            server.parse_feed(b"<html></html>", "x")

    def test_merge_dedupes_and_sorts(self):
        items = server.parse_feed(RSS_SAMPLE, "A") + server.parse_feed(RSS_SAMPLE, "B")
        merged = server.merge_items(items, limit=10)
        self.assertEqual(len(merged), 2)  # duplicates removed by URL
        self.assertEqual(merged[0]["url"], "https://example.org/b")  # newest first

    def test_strip_html_truncates(self):
        text = server.strip_html("word " * 100, limit=50)
        self.assertLessEqual(len(text), 51)
        self.assertTrue(text.endswith("…"))


class ArticleExtractorTests(unittest.TestCase):
    def test_extracts_title_and_paragraphs(self):
        ex = server._ArticleExtractor()
        ex.feed(HTML_SAMPLE.decode())
        self.assertIn("Article Title", ex.title)
        texts = [t for _, t in ex.blocks]
        self.assertTrue(any("first real paragraph" in t for t in texts))
        self.assertFalse(any("junk" in t for t in texts))  # nav + script skipped

    def test_extracts_meta_image_author(self):
        html = (b"<html><head><title>T</title>"
                b'<meta property="og:image" content="https://cdn.example.com/hero.jpg">'
                b'<meta name="author" content="Jane Reporter">'
                b'<meta property="article:published_time" content="2026-07-16T10:00:00Z">'
                b"</head><body><p>" + b"word " * 250 + b"</p></body></html>")
        ex = server._ArticleExtractor()
        ex.feed(html.decode())
        self.assertEqual(ex.image, "https://cdn.example.com/hero.jpg")
        self.assertEqual(ex.author, "Jane Reporter")
        self.assertTrue(ex.published.startswith("2026-07-16"))

    def test_live_reader_enriches_from_fixture(self):
        html = (b"<html><head><title>Big Story</title>"
                b'<meta property="og:image" content="https://cdn.example.com/x.jpg">'
                b'<meta name="author" content="Reporter">'
                b"</head><body>" + (b"<p>" + b"word " * 60 + b"</p>") * 4 + b"</body></html>")
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=html):
            doc = server.live_reader("https://news.example.com/story")
        self.assertEqual(doc["image"], "https://cdn.example.com/x.jpg")
        self.assertEqual(doc["author"], "Reporter")
        self.assertGreaterEqual(doc["readingMinutes"], 1)


class SampleFallbackTests(unittest.TestCase):
    def setUp(self):
        server.CACHE.clear()
        self.api = server.Api(offline=True)

    def test_news_offline_serves_samples(self):
        data = self.api.news({"topic": ["tech"], "limit": ["5"]})
        self.assertEqual(data["source"], "sample")
        self.assertLessEqual(len(data["items"]), 5)
        for item in data["items"]:
            self.assertTrue(item["title"] and item["url"] and item["published"])

    def test_air_offline_shape(self):
        d = self.api.air({"lat": ["40.71"], "lon": ["-74.01"]})
        self.assertEqual(d["source"], "sample")
        self.assertIsInstance(d["aqi"], int)
        self.assertIn("label", d["band"])
        self.assertTrue(d["pollutants"])
        for p in d["pollutants"]:
            self.assertIn("label", p)
            self.assertIsNotNone(p["value"])
        self.assertTrue(d["pollen"])

    def test_marine_offline_shape(self):
        d = self.api.marine({"lat": ["-33.92"], "lon": ["18.42"], "name": ["Cape Town"]})
        self.assertEqual(d["source"], "sample")
        for k in ("waveHeight", "swellHeight", "seaTemp", "seaState"):
            self.assertIn(k, d)
        self.assertIsInstance(d["waveHeight"], (int, float))
        self.assertTrue(d["waveDirText"])

    def test_marine_normalizer_and_sea_state(self):
        self.assertEqual(server._sea_state(0.3), "Calm")
        self.assertEqual(server._sea_state(3.0), "Rough")
        self.assertEqual(server._compass(0), "N")
        self.assertEqual(server._compass(180), "S")
        import unittest.mock as mock
        payload = json.dumps({
            "current": {"wave_height": 1.4, "wave_period": 9.0, "wave_direction": 225,
                        "sea_surface_temperature": 16.2, "swell_wave_height": 1.1,
                        "swell_wave_period": 11.0, "wind_wave_height": 0.6},
            "daily": {"wave_height_max": [2.0]},
        }).encode()
        with mock.patch.object(server, "fetch_url", return_value=payload):
            out = server.live_marine(-33.92, 18.42, "Cape Town")
        self.assertEqual(out["waveDirText"], "SW")
        self.assertEqual(out["seaState"], "Moderate")
        self.assertEqual(out["waveMax"], 2.0)

    def test_flights_offline_shape(self):
        d = self.api.flights({"lat": ["40.71"], "lon": ["-74.01"]})
        self.assertEqual(d["source"], "sample")
        self.assertTrue(d["flights"])
        for f in d["flights"]:
            self.assertTrue(f["callsign"])
            self.assertFalse(f["onGround"])

    def test_flights_normalizer_sorts_by_altitude(self):
        import unittest.mock as mock
        # state vector layout: icao,callsign,country,_,_,lon,lat,alt,onground,vel,track,vrate
        raw = {"states": [
            ["abc", "HIGH123 ", "US", 0, 0, -74.0, 40.7, 11000, False, 240, 90, 0],
            ["def", "LOW456 ", "US", 0, 0, -74.1, 40.6, 2000, False, 150, 180, 5.0],
            ["ghi", "GND789 ", "US", 0, 0, -74.2, 40.5, None, True, 0, 0, 0],
        ]}
        with mock.patch.object(server, "fetch_url", return_value=json.dumps(raw)):
            out = server.live_flights(40.7, -74.0, "Test")
        self.assertEqual(out["count"], 2)                    # grounded one excluded
        self.assertEqual(out["flights"][0]["callsign"], "LOW456")  # lowest first
        self.assertEqual(out["flights"][0]["dir"], "S")      # track 180 → S

    def test_flights_rejects_bad_coords(self):
        with self.assertRaises(server.ApiError):
            self.api.flights({"lat": ["nope"], "lon": ["1"]})

    def test_alerts_offline_shape(self):
        d = self.api.alerts({"lat": ["40.71"], "lon": ["-74.01"]})
        self.assertEqual(d["source"], "sample")
        self.assertTrue(d["alerts"])
        for a in d["alerts"]:
            self.assertTrue(a["event"])
            self.assertIn(a["tone"], ("down", "warn", "neutral"))

    def test_alerts_normalizer_sorts_by_severity(self):
        import unittest.mock as mock
        raw = {"features": [
            {"properties": {"event": "Flood Advisory", "severity": "Minor",
                            "areaDesc": "County A", "expires": "2026-07-17T20:00:00Z"}},
            {"properties": {"event": "Tornado Warning", "severity": "Extreme",
                            "areaDesc": "County B", "expires": "2026-07-17T19:00:00Z"}},
        ]}
        with mock.patch.object(server, "fetch_url", return_value=json.dumps(raw)):
            out = server.live_alerts(40.0, -75.0, "Test")
        self.assertEqual(out["alerts"][0]["event"], "Tornado Warning")  # Extreme first
        self.assertEqual(out["alerts"][0]["tone"], "down")

    def test_alerts_rejects_bad_coords(self):
        with self.assertRaises(server.ApiError):
            self.api.alerts({"lat": ["x"], "lon": ["1"]})

    def test_spaceweather_offline_shape(self):
        d = self.api.spaceweather({})
        self.assertEqual(d["source"], "sample")
        self.assertIsInstance(d["kp"], (int, float))
        self.assertIn("label", d["band"])
        self.assertGreaterEqual(d["peak24h"], d["kp"])
        self.assertTrue(d["series"])
        for s in d["series"]:
            self.assertIn("t", s)
            self.assertIsInstance(s["kp"], (int, float))

    def test_kp_band_thresholds(self):
        self.assertEqual(server.kp_band(2)["label"], "Quiet")
        self.assertEqual(server.kp_band(4.5)["label"], "Unsettled")
        self.assertEqual(server.kp_band(5)["label"], "G1 minor storm")
        self.assertEqual(server.kp_band(9)["label"], "G5 extreme storm")
        self.assertEqual(server.kp_band(None)["label"], "—")

    def test_spaceweather_normalizer_from_fixture(self):
        import unittest.mock as mock
        raw = [["time_tag", "Kp", "a_running", "station_count"],
               ["2026-07-17 00:00:00", "3.00", "5", "8"],
               ["2026-07-17 03:00:00", "5.67", "20", "8"]]
        with mock.patch.object(server, "fetch_url", return_value=json.dumps(raw)):
            out = server.live_spaceweather()
        self.assertEqual(out["kp"], 5.67)
        self.assertEqual(out["peak24h"], 5.67)
        self.assertIn("Aurora", out["aurora"])

    def test_air_band_thresholds(self):
        self.assertEqual(server.aqi_band(20)["label"], "Good")
        self.assertEqual(server.aqi_band(75)["label"], "Moderate")
        self.assertEqual(server.aqi_band(175)["label"], "Unhealthy")
        self.assertEqual(server.aqi_band(500)["label"], "Hazardous")
        self.assertEqual(server.aqi_band(None)["label"], "—")

    def test_air_rejects_bad_coords(self):
        with self.assertRaises(server.ApiError):
            self.api.air({"lat": ["abc"], "lon": ["1"]})

    def test_latlon_rejects_nonfinite_and_out_of_range(self):
        for endpoint in (self.api.air, self.api.alerts, self.api.flights, self.api.weather):
            with self.assertRaises(server.ApiError):
                endpoint({"lat": ["inf"], "lon": ["0"]})
            with self.assertRaises(server.ApiError):
                endpoint({"lat": ["nan"], "lon": ["0"]})
            with self.assertRaises(server.ApiError):
                endpoint({"lat": ["120"], "lon": ["0"]})    # lat > 90
            with self.assertRaises(server.ApiError):
                endpoint({"lat": ["0"], "lon": ["999"]})    # lon > 180

    def test_news_all_aggregates_topics(self):
        data = self.api.news({"all": ["1"], "limit": ["40"]})
        self.assertEqual(data["topic"], "all")
        self.assertTrue(data["items"])
        topics = {item.get("topic") for item in data["items"]}
        self.assertGreaterEqual(len(topics), 2)      # spans multiple topics
        urls = [i["url"] for i in data["items"]]
        self.assertEqual(len(urls), len(set(urls)))  # deduped across topics

    def test_news_rejects_unknown_topic(self):
        with self.assertRaises(server.ApiError):
            self.api.news({"topic": ["nonsense"]})

    def test_weather_offline_shape(self):
        data = self.api.weather({"lat": ["51.5"], "lon": ["-0.1"], "name": ["London"]})
        self.assertEqual(data["source"], "sample")
        self.assertEqual(data["location"]["name"], "London")
        self.assertEqual(len(data["hourly"]), 24)
        self.assertGreaterEqual(len(data["daily"]), 7)
        for key in ("temp", "feels", "humidity", "wind", "code"):
            self.assertIn(key, data["current"])

    def test_markets_offline_shape(self):
        data = self.api.markets({})
        self.assertEqual(data["source"], "sample")
        self.assertTrue(data["assets"])
        for asset in data["assets"]:
            self.assertGreater(len(asset["spark"]), 10)

    def test_worldstate_offline(self):
        data = self.api.worldstate({})
        self.assertEqual(data["source"], "sample")
        self.assertEqual(len(data["domains"]), len(server.WORLD_DOMAINS))
        for domain in data["domains"]:
            self.assertTrue(0 <= domain["score"] <= 100)
            self.assertIn(domain["level"], ("stable", "watch", "elevated", "critical"))
            self.assertTrue(domain["explanation"])
        self.assertIn(data["overall"]["level"], ("stable", "watch", "elevated", "critical"))

    def test_reader_blocks_private_hosts(self):
        for url in ("http://localhost/x", "http://127.0.0.1/x", "http://10.1.2.3/",
                    "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data/",
                    "http://[::1]/", "http://foo.internal/", "ftp://example.org/"):
            with self.assertRaises(server.ApiError, msg=url):
                self.api.reader({"url": [url]})

    def test_host_is_blocked_helper(self):
        # loopback / private / link-local / metadata / reserved → blocked
        for host in ("localhost", "127.0.0.1", "10.0.0.5", "192.168.0.1",
                     "172.16.0.1", "169.254.169.254", "::1", "0.0.0.0",
                     "box.local", "svc.internal", ""):
            self.assertTrue(server.host_is_blocked(host), host)
        # ordinary public IP literals are allowed
        for host in ("8.8.8.8", "1.1.1.1"):
            self.assertFalse(server.host_is_blocked(host), host)

    def test_guarded_redirect_rejects_private_target(self):
        handler = server._GuardedRedirectHandler()
        with self.assertRaises(urllib.error.URLError):
            handler.redirect_request(
                None, None, 302, "Found", {}, "http://169.254.169.254/")

    def test_reader_offline_note(self):
        data = self.api.reader({"url": ["https://example.org/story"]})
        self.assertEqual(data["source"], "sample")
        self.assertIn("note", data)

    def test_gaming_is_a_default_topic(self):
        self.assertIn("gaming", self.api.feeds.topics())
        data = self.api.news({"topic": ["gaming"], "limit": ["10"]})
        self.assertTrue(data["items"])

    def test_regional_and_ai_source_packs(self):
        topics = self.api.feeds.topics()
        for pack in ("southafrica", "africa", "ai", "finance"):
            self.assertIn(pack, topics)
            data = self.api.news({"topic": [pack], "limit": ["10"]})
            self.assertTrue(data["items"])

    def test_gaming_free_and_deals_sample(self):
        f = self.api.gaming_free({})
        self.assertTrue(f["current"])
        self.assertIn("title", f["current"][0])
        d = self.api.gaming_deals({})
        self.assertTrue(d["deals"])
        self.assertIn("discount", d["deals"][0])

    def test_steam_deals_normalizer_drops_undiscounted(self):
        raw = {"specials": {"items": [
            {"id": 1, "name": "On Sale", "discounted": True, "discount_percent": 50,
             "final_price": 999, "header_image": "x"},
            {"id": 2, "name": "Full Price", "discounted": False},
        ]}}
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=json.dumps(raw).encode()):
            out = server.live_steam_deals()
        self.assertEqual(len(out["deals"]), 1)
        self.assertEqual(out["deals"][0]["price"], 9.99)

    def test_epic_free_games_normalizer(self):
        raw = {"data": {"Catalog": {"searchStore": {"elements": [
            {"title": "Freebie", "productSlug": "freebie", "keyImages": [{"type": "Thumbnail", "url": "u"}],
             "promotions": {"promotionalOffers": [{"promotionalOffers": [{"endDate": "2026-08-01T00:00:00Z"}]}],
                            "upcomingPromotionalOffers": []}},
        ]}}}}
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=json.dumps(raw).encode()):
            out = server.live_free_games()
        self.assertEqual(out["current"][0]["title"], "Freebie")
        self.assertTrue(out["current"][0]["url"].endswith("/freebie"))

    def test_cache_hit_returns_same_object(self):
        first = self.api.news({"topic": ["top"], "limit": ["10"]})
        second = self.api.news({"topic": ["top"], "limit": ["10"]})
        self.assertIs(first, second)


class AssistantLocalTests(unittest.TestCase):
    CONTEXT = {
        "tasks": [{"name": "Today", "items": [
            {"text": "pay rent", "done": False},
            {"text": "walk the dog", "done": True},
        ]}],
        "events": [{"date": "2099-01-02", "title": "review"}],
        "headlines": ["Something happened"],
        "apps": [{"name": "GitHub", "url": "https://github.com"}],
    }

    def setUp(self):
        self.assistant = assistant.Assistant()

    def test_mode_is_local_without_credentials(self):
        # The sandbox has no anthropic SDK installed; status must degrade cleanly.
        status = self.assistant.status()
        self.assertIn(status["mode"], ("local", "claude"))
        if not assistant._HAVE_SDK:
            self.assertEqual(status["mode"], "local")
            self.assertIn("pip install anthropic", status["hint"])

    def _chat(self, text):
        return self.assistant._chat_local([{"role": "user", "content": text}], self.CONTEXT)

    def _actions(self, response):
        return [(b["name"], b["input"]) for b in response["content"] if b["type"] == "tool_use"]

    def test_add_task_command(self):
        actions = self._actions(self._chat("add task buy stamps to errands"))
        self.assertEqual(actions, [("add_task", {"text": "buy stamps", "list": "Errands",
                                                 "due": None, "priority": None})])

    def test_add_task_defaults_to_today(self):
        actions = self._actions(self._chat("task: water plants"))
        self.assertEqual(actions[0][1]["list"], "Today")

    def test_add_task_parses_due_and_priority(self):
        actions = self._actions(self._chat("add task pay rent !high @2026-07-20 to bills"))
        name, args = actions[0]
        self.assertEqual(name, "add_task")
        self.assertEqual(args["text"], "pay rent")
        self.assertEqual(args["list"], "Bills")
        self.assertEqual(args["due"], "2026-07-20")
        self.assertEqual(args["priority"], "high")

    def test_parse_task_tokens_relative_dates(self):
        from datetime import date, timedelta
        _, due, prio = assistant.parse_task_tokens("ship it !low @tomorrow")
        self.assertEqual(due, (date.today() + timedelta(days=1)).isoformat())
        self.assertEqual(prio, "low")

    def test_complete_command(self):
        actions = self._actions(self._chat("complete pay rent"))
        self.assertEqual(actions, [("complete_task", {"text": "pay rent"})])

    def test_event_command_tomorrow(self):
        actions = self._actions(self._chat("add event tomorrow: dentist"))
        self.assertEqual(actions[0][0], "add_event")
        self.assertRegex(actions[0][1]["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(actions[0][1]["title"], "dentist")

    def test_open_known_app(self):
        actions = self._actions(self._chat("open github"))
        self.assertEqual(actions, [("open_url", {"url": "https://github.com", "title": "GitHub"})])

    def test_news_topic_command(self):
        actions = self._actions(self._chat("show tech news"))
        self.assertEqual(actions, [("switch_news_topic", {"topic": "tech"})])

    def test_unknown_command_is_helpful_not_action(self):
        response = self._chat("please compose a symphony")
        self.assertEqual(self._actions(response), [])
        self.assertIn("add task", response["content"][0]["text"])

    def test_briefing_mentions_tasks_and_automation(self):
        briefing = assistant.local_briefing(self.CONTEXT)
        self.assertIn("pay rent", briefing)
        self.assertIn("FOCUS", briefing)
        self.assertIn("recurring", briefing)  # 'pay' triggers automation rule

    def test_extractive_summary_prefers_lead(self):
        text = ("Alpha beta gamma delta epsilon zeta eta theta first sentence here. "
                "Filler words entirely unrelated content again okay fine. " * 3)
        result = assistant.extractive_summary(text)
        self.assertTrue(result.startswith("Alpha"))

    def test_extractive_summary_lists(self):
        text = "\n".join(f"headline number {i}" for i in range(8))
        result = assistant.extractive_summary(text)
        self.assertTrue(result.startswith("• headline number 0"))
        self.assertIn("more", result)

    def test_summarize_requires_content(self):
        with self.assertRaises(ValueError):
            self.assistant.summarize({"content": ""})


class AutomationsTests(unittest.TestCase):
    def setUp(self):
        server.CACHE.clear()
        self.api = server.Api(offline=True, data_dir=Path(tempfile.mkdtemp()))
        self.autos = self.api.automations

    def test_validation(self):
        import automations as autos_mod
        bad = [
            {"name": "x", "trigger": {"type": "hourly"}, "action": {"type": "notify", "message": "m"}},
            {"name": "x", "trigger": {"type": "daily", "time": "25:00"}, "action": {"type": "notify", "message": "m"}},
            {"name": "x", "trigger": {"type": "market", "symbol": "BTC", "percent": 0}, "action": {"type": "notify", "message": "m"}},
            {"name": "x", "trigger": {"type": "worldstate", "level": "calm"}, "action": {"type": "notify", "message": "m"}},
            {"name": "x", "trigger": {"type": "daily", "time": "08:00"}, "action": {"type": "notify", "message": " "}},
            {"name": "", "trigger": {"type": "daily", "time": "08:00"}, "action": {"type": "briefing"}},
        ]
        for rule in bad:
            self.assertIsNotNone(autos_mod.validate_rule(rule), rule)
        self.assertIsNone(autos_mod.validate_rule(
            {"name": "ok", "trigger": {"type": "daily", "time": "08:00"}, "action": {"type": "briefing"}}))

    def test_crud_and_persistence(self):
        rule = self.autos.create_rule(
            {"name": "R1", "trigger": {"type": "daily", "time": "08:00"}, "action": {"type": "briefing"}})
        self.assertEqual(rule["id"], 1)
        # a new engine instance sees the persisted rule
        reloaded = server.Automations(self.autos.path, self.api)
        self.assertEqual(len(reloaded.list_rules()), 1)
        self.assertTrue(self.autos.delete_rule(1))
        self.assertFalse(self.autos.delete_rule(1))

    def test_market_rule_edge_triggered(self):
        # sample BTC 24h change is +2.41%
        self.autos.create_rule({
            "name": "BTC", "trigger": {"type": "market", "symbol": "BTC", "percent": 2},
            "action": {"type": "notify", "message": "moved"}})
        self.assertEqual(self.autos.tick(), 1)   # crossing fires
        self.assertEqual(self.autos.tick(), 0)   # still beyond → no refire
        notes = self.autos.notifications_after(0)["notifications"]
        self.assertEqual(len(notes), 1)
        self.assertIn("2.41%", notes[0]["body"])

    def test_market_rule_below_threshold(self):
        self.autos.create_rule({
            "name": "BTC", "trigger": {"type": "market", "symbol": "BTC", "percent": 50},
            "action": {"type": "notify", "message": "moved"}})
        self.assertEqual(self.autos.tick(), 0)

    def test_market_price_cross_edge_triggered(self):
        # sample BTC spot is 112,840
        self.autos.create_rule({
            "name": "BTC 100k", "trigger": {
                "type": "market", "symbol": "BTC", "price": 100000, "direction": "above"},
            "action": {"type": "notify", "message": "crossed"}})
        self.assertEqual(self.autos.tick(), 1)   # 112840 >= 100000 → fires
        self.assertEqual(self.autos.tick(), 0)   # still above → no refire
        body = self.autos.notifications_after(0)["notifications"][0]["body"]
        self.assertIn("above", body)

    def test_market_price_cross_not_reached(self):
        self.autos.create_rule({
            "name": "BTC 200k", "trigger": {
                "type": "market", "symbol": "BTC", "price": 200000, "direction": "above"},
            "action": {"type": "notify", "message": "crossed"}})
        self.assertEqual(self.autos.tick(), 0)   # 112840 < 200000 → silent

    def test_market_rsi_threshold(self):
        # sample BTC RSI(14) ≈ 44.7
        self.autos.create_rule({
            "name": "BTC oversold-ish", "trigger": {
                "type": "market", "symbol": "BTC", "rsi": 50, "direction": "below"},
            "action": {"type": "notify", "message": "rsi"}})
        self.assertEqual(self.autos.tick(), 1)   # 44.7 <= 50 → fires
        self.assertEqual(self.autos.tick(), 0)   # still below → no refire

    def test_market_rsi_not_reached(self):
        self.autos.create_rule({
            "name": "BTC deep oversold", "trigger": {
                "type": "market", "symbol": "BTC", "rsi": 30, "direction": "below"},
            "action": {"type": "notify", "message": "rsi"}})
        self.assertEqual(self.autos.tick(), 0)   # 44.7 > 30 → silent

    def test_market_trigger_validation_modes(self):
        import automations as autos_mod
        # exactly one of percent/price/rsi
        self.assertIsNotNone(autos_mod.validate_rule({
            "name": "x", "trigger": {"type": "market", "symbol": "BTC"},
            "action": {"type": "notify", "message": "m"}}))
        self.assertIsNotNone(autos_mod.validate_rule({
            "name": "x", "trigger": {"type": "market", "symbol": "BTC", "price": 1, "percent": 2},
            "action": {"type": "notify", "message": "m"}}))
        # price needs a direction
        self.assertIsNotNone(autos_mod.validate_rule({
            "name": "x", "trigger": {"type": "market", "symbol": "BTC", "price": 100},
            "action": {"type": "notify", "message": "m"}}))
        # well-formed price + rsi
        self.assertIsNone(autos_mod.validate_rule({
            "name": "x", "trigger": {"type": "market", "symbol": "BTC", "price": 100, "direction": "above"},
            "action": {"type": "notify", "message": "m"}}))
        self.assertIsNone(autos_mod.validate_rule({
            "name": "x", "trigger": {"type": "market", "symbol": "ETH", "rsi": 30, "direction": "below"},
            "action": {"type": "notify", "message": "m"}}))

    def test_daily_rule_fires_once_per_day(self):
        from datetime import datetime
        rule = self.autos.create_rule({
            "name": "Brief", "trigger": {"type": "daily", "time": "08:00"},
            "action": {"type": "briefing"}})
        # simulate: created before 08:00 (clear the retro-guard), then 09:00 arrives
        with self.autos._lock:
            self.autos._data["rules"][0]["state"] = {}
        nine = datetime(2030, 1, 2, 9, 0)
        self.assertEqual(self.autos.tick(now=nine), 1)
        self.assertEqual(self.autos.tick(now=nine), 0)  # same day: once only
        next_day = datetime(2030, 1, 3, 9, 0)
        self.assertEqual(self.autos.tick(now=next_day), 1)
        body = self.autos.notifications_after(0)["notifications"][0]["body"]
        self.assertIn("FOCUS", body)  # briefing action produced a briefing

    def test_worldstate_rule(self):
        # sample data compiles to a "stable"-ish index; watch threshold with
        # rank >= watch should depend on computed level — use "watch" and
        # verify edge-triggering semantics rather than a fixed outcome.
        self.autos.create_rule({
            "name": "World", "trigger": {"type": "worldstate", "level": "watch"},
            "action": {"type": "notify", "message": "world moved"}})
        first = self.autos.tick()
        second = self.autos.tick()
        self.assertIn(first, (0, 1))
        self.assertEqual(second, 0)  # never refires while condition persists

    def test_notifications_after_filtering(self):
        self.autos._notify("A", "a")
        self.autos._notify("B", "b")
        all_notes = self.autos.notifications_after(0)
        self.assertEqual([n["title"] for n in all_notes["notifications"]], ["A", "B"])
        newer = self.autos.notifications_after(all_notes["notifications"][0]["id"])
        self.assertEqual([n["title"] for n in newer["notifications"]], ["B"])


class FeedConfigTests(unittest.TestCase):
    def setUp(self):
        server.CACHE.clear()
        self.api = server.Api(offline=True, data_dir=Path(tempfile.mkdtemp()))

    def test_defaults_present(self):
        snap = self.api.feeds_config({})
        self.assertIn("top", snap["topics"])
        self.assertIn("tech", snap["sources"])
        self.assertTrue(snap["sources"]["tech"])

    def test_add_topic_and_source_roundtrip(self):
        self.api.feeds_op({"op": "add_topic", "name": "Gaming News!"})
        snap = self.api.feeds_op({
            "op": "add_source", "topic": "gaming-news",
            "name": "RPS", "url": "https://example.org/feed.xml"})
        self.assertEqual(snap["sources"]["gaming-news"][0]["name"], "RPS")
        # persisted for a fresh instance
        reloaded = server.FeedConfig(self.api.feeds.path)
        self.assertIn("gaming-news", reloaded.topics())
        # custom topic serves sample data offline instead of erroring
        news = self.api.news({"topic": ["gaming-news"], "limit": ["4"]})
        self.assertEqual(news["source"], "sample")

    def test_follow_a_search_creates_google_news_topic(self):
        self.api.feeds_op({"op": "add_search", "name": "Fed rate", "query": "Fed rate decision"})
        self.assertIn("fed-rate", self.api.feeds.topics())
        src = self.api.feeds.snapshot()["sources"]["fed-rate"][0]
        self.assertIn("news.google.com/rss/search", src["url"])
        self.assertIn("Fed%20rate%20decision", src["url"])
        with self.assertRaises(server.ApiError):     # empty query rejected
            self.api.feeds_op({"op": "add_search", "name": "x", "query": ""})

    def test_validation_errors(self):
        with self.assertRaises(server.ApiError):
            self.api.feeds_op({"op": "add_topic", "name": "!!!"})
        with self.assertRaises(server.ApiError):
            self.api.feeds_op({"op": "add_source", "topic": "tech",
                               "name": "x", "url": "ftp://nope"})
        with self.assertRaises(server.ApiError):
            self.api.feeds_op({"op": "remove_topic", "name": "not-there"})
        with self.assertRaises(server.ApiError):
            self.api.feeds_op({"op": "explode"})

    def test_duplicate_source_rejected(self):
        self.api.feeds_op({"op": "add_source", "topic": "tech",
                           "name": "X", "url": "https://example.org/f"})
        with self.assertRaises(server.ApiError):
            self.api.feeds_op({"op": "add_source", "topic": "tech",
                               "name": "Y", "url": "https://example.org/f"})

    def test_remove_topic_and_worldstate_resilience(self):
        self.api.feeds_op({"op": "remove_topic", "name": "world"})
        with self.assertRaises(server.ApiError):
            self.api.news({"topic": ["world"]})
        world = self.api.worldstate({})  # must not 500 with a topic missing
        self.assertIn("overall", world)

    def test_reset_restores_defaults(self):
        self.api.feeds_op({"op": "remove_topic", "name": "world"})
        self.api.feeds_op({"op": "reset"})
        self.assertIn("world", self.api.feeds.topics())


class MarketsWatchlistTests(unittest.TestCase):
    def setUp(self):
        server.CACHE.clear()
        self.api = server.Api(offline=True, data_dir=Path(tempfile.mkdtemp()))

    def test_ids_filter_sample(self):
        data = self.api.markets({"ids": ["bitcoin,solana"]})
        self.assertEqual({a["symbol"] for a in data["assets"]}, {"BTC", "SOL"})

    def test_unknown_ids_fall_back_to_full_sample(self):
        data = self.api.markets({"ids": ["floopcoin"]})
        self.assertGreaterEqual(len(data["assets"]), 4)

    def test_ids_sanitized_and_capped(self):
        raw = ",".join([f"coin{i}" for i in range(30)]) + ",<script>"
        data = self.api.markets({"ids": [raw]})  # must not blow up
        self.assertIn("assets", data)

    def test_sample_market_assets_carry_id(self):
        data = self.api.markets({"ids": ["bitcoin"]})
        self.assertEqual(data["assets"][0]["id"], "bitcoin")

    def test_coin_detail_and_chart_sample(self):
        d = self.api.crypto_coin({"id": ["bitcoin"]})
        self.assertEqual(d["symbol"], "BITC")
        self.assertGreater(d["price"], 0)
        c = self.api.crypto_chart({"id": ["bitcoin"], "days": ["30"]})
        self.assertGreaterEqual(len(c["candles"]), 2)
        self.assertIn("sma20", c["overlays"])
        self.assertTrue(all(k in c["candles"][0] for k in ("t", "o", "h", "l", "c")))

    def test_chart_days_clamped(self):
        c = self.api.crypto_chart({"id": ["bitcoin"], "days": ["999"]})
        self.assertEqual(c["days"], 30)  # unknown range → default 30

    def test_crypto_global_and_trending_sample(self):
        g = self.api.crypto_global({})
        self.assertGreater(g["btcDominance"], 0)
        self.assertIn("value", g["fearGreed"])
        t = self.api.crypto_trending({})
        self.assertTrue(t["coins"])
        self.assertIn("symbol", t["coins"][0])

    def test_stocks_sample_and_history(self):
        d = self.api.stocks({})
        self.assertTrue(d["assets"])
        a = d["assets"][0]
        for k in ("symbol", "name", "price", "changePct"):
            self.assertIn(k, a)
        hist = self.api.stocks_history({"symbol": ["^spx"]})
        self.assertGreaterEqual(len(hist["candles"]), 2)
        self.assertIn("sma20", hist["overlays"])

    def test_stocks_csv_normalizer_skips_nd(self):
        csv_text = ("Symbol,Date,Time,Open,High,Low,Close,Volume\n"
                    "AAPL.US,2026-07-16,22:00:00,200,205,199,204,1000\n"
                    "BADX.US,N/D,N/D,N/D,N/D,N/D,N/D,N/D\n")
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=csv_text.encode()):
            out = server.live_stocks(["aapl.us", "badx.us"])
        self.assertEqual(len(out["assets"]), 1)          # N/D row skipped
        a = out["assets"][0]
        self.assertEqual(a["symbol"], "AAPL")
        self.assertAlmostEqual(a["changePct"], (204 - 200) / 200 * 100)

    def test_scores_sample_shape(self):
        for league in ("nba", "nfl", "epl"):
            d = self.api.scores({"league": [league]})
            self.assertEqual(d["league"], league)
            self.assertTrue(d["games"])
            g = d["games"][0]
            self.assertIn(g["state"], ("pre", "in", "post"))
            for side in ("home", "away"):
                self.assertIn("abbr", g[side])

    def test_scores_rugby_cricket_leagues(self):
        for league in ("urc", "rugbyc", "cricket"):
            d = self.api.scores({"league": [league]})
            self.assertEqual(d["league"], league)
            self.assertTrue(d["games"])
        boks = self.api.scores({"league": ["rugbyc"]})["games"]
        self.assertTrue(any(g["home"]["abbr"] == "RSA" or g["away"]["abbr"] == "RSA"
                            for g in boks))

    def test_scores_soccer_combat_racket_leagues(self):
        for league in ("laliga", "seriea", "bundesliga", "ligue1", "ucl", "psl",
                       "mma", "atp", "wta"):
            d = self.api.scores({"league": [league]})
            self.assertEqual(d["league"], league)
            self.assertTrue(d["games"])
            g = d["games"][0]
            self.assertIn("abbr", g["home"])
            self.assertIn("abbr", g["away"])
        psl = self.api.scores({"league": ["psl"]})["games"]
        self.assertTrue(any("Sundowns" in g["home"]["name"] or "Sundowns" in g["away"]["name"]
                            for g in psl))

    def test_racing_sample_and_normalizer(self):
        for series in ("f1", "motogp", "nascar", "indycar"):
            d = self.api.racing({"series": [series]})
            self.assertEqual(d["series"], series)
            self.assertTrue(d["races"])
        f1 = self.api.racing({"series": ["f1"]})
        finished = [r for r in f1["races"] if r["state"] == "post"]
        self.assertTrue(finished and finished[0]["winner"])
        self.assertTrue(len(finished[0]["top"]) >= 1)
        with self.assertRaises(server.ApiError):
            self.api.racing({"series": ["motocross"]})

    def test_racing_live_normalizer_from_fixture(self):
        import unittest.mock as mock
        payload = json.dumps({"events": [{
            "id": "1", "shortName": "Kyalami GP",
            "status": {"type": {"state": "post", "shortDetail": "Final"}},
            "competitions": [{
                "venue": {"fullName": "Kyalami Circuit"},
                "competitors": [
                    {"order": 2, "athlete": {"displayName": "L. Norris"}},
                    {"order": 1, "athlete": {"displayName": "M. Verstappen"}},
                ]}]}]}).encode()
        with mock.patch.object(server, "fetch_url", return_value=payload):
            out = server.live_racing("f1")
        self.assertEqual(out["races"][0]["winner"], "M. Verstappen")
        self.assertEqual(out["races"][0]["top"][0], "M. Verstappen")

    def test_scores_unknown_league_rejected(self):
        with self.assertRaises(server.ApiError):
            self.api.scores({"league": ["quidditch"]})

    def test_team_schedule_sample_shape(self):
        d = self.api.team_schedule({"league": ["nba"], "team": ["BOS"]})
        self.assertEqual(d["team"], "BOS")
        self.assertTrue(d["games"])
        states = {g["state"] for g in d["games"]}
        self.assertTrue({"pre", "post"} & states)      # has recent and/or upcoming
        for g in d["games"]:
            self.assertIn("start", g)
            for side in ("home", "away"):
                self.assertIn("abbr", g[side])

    def test_team_schedule_requires_team(self):
        with self.assertRaises(server.ApiError):
            self.api.team_schedule({"league": ["nba"], "team": [""]})

    def test_team_schedule_unknown_league_rejected(self):
        with self.assertRaises(server.ApiError):
            self.api.team_schedule({"league": ["quidditch"], "team": ["BOS"]})

    def test_team_news_sample_shape(self):
        d = self.api.team_news({"team": ["Arsenal"]})
        self.assertEqual(d["team"], "Arsenal")
        self.assertTrue(d["items"])
        for item in d["items"]:
            self.assertTrue(item["title"] and item["url"])

    def test_standings_sample_shape(self):
        for lg in ("nba", "epl"):
            d = self.api.standings({"league": [lg]})
            self.assertTrue(d["columns"])
            self.assertTrue(d["groups"][0]["teams"])
            t = d["groups"][0]["teams"][0]
            self.assertIn("abbr", t)
            self.assertIn(d["columns"][0]["key"], t["stats"])

    def test_standings_normalizer_from_fixture(self):
        raw = {"children": [{"name": "East", "standings": {"entries": [
            {"team": {"abbreviation": "BOS", "displayName": "Celtics"},
             "stats": [{"name": "wins", "displayValue": "48"},
                       {"name": "losses", "displayValue": "12"},
                       {"name": "randomstat", "displayValue": "x"}]},
        ]}}]}
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=json.dumps(raw).encode()):
            out = server.live_standings("nba")
        self.assertEqual(out["groups"][0]["teams"][0]["abbr"], "BOS")
        keys = [c["key"] for c in out["columns"]]
        self.assertIn("wins", keys)
        self.assertNotIn("randomstat", keys)  # unknown stat filtered out

    def test_pubmed_sample_and_normalizer(self):
        d = self.api.pubmed({"q": ["tuberculosis south africa"]})
        self.assertTrue(d["articles"])
        self.assertIn("journal", d["articles"][0])
        # normalizer maps NCBI esearch/esummary JSON
        import unittest.mock as mock
        esearch = json.dumps({"esearchresult": {"idlist": ["111", "222"]}}).encode()
        esummary = json.dumps({"result": {"uids": ["111"], "111": {
            "title": "A study", "source": "SAMJ", "pubdate": "2026 Jul",
            "authors": [{"name": "A B"}, {"name": "C D"}, {"name": "E F"}, {"name": "G H"}]}}}).encode()
        with mock.patch.object(server, "fetch_url", side_effect=[esearch, esummary]):
            out = server.live_pubmed("x")
        self.assertEqual(out["articles"][0]["pmid"], "111")
        self.assertIn("et al.", out["articles"][0]["authors"])
        self.assertTrue(out["articles"][0]["url"].endswith("/111/"))

    def test_trials_sample_and_normalizer(self):
        d = self.api.trials({"q": ["HIV"]})
        self.assertTrue(d["trials"])
        self.assertIn("status", d["trials"][0])
        import unittest.mock as mock
        raw = json.dumps({"studies": [{"protocolSection": {
            "identificationModule": {"nctId": "NCT01", "briefTitle": "Trial X"},
            "statusModule": {"overallStatus": "RECRUITING", "lastUpdatePostDateStruct": {"date": "2026-07-01"}},
            "conditionsModule": {"conditions": ["TB", "HIV"]}}}]}).encode()
        with mock.patch.object(server, "fetch_url", return_value=raw):
            out = server.live_trials("x")
        self.assertEqual(out["trials"][0]["nct"], "NCT01")
        self.assertEqual(out["trials"][0]["status"], "RECRUITING")
        self.assertTrue(out["trials"][0]["url"].endswith("NCT01"))

    def test_drug_sample_and_normalizer(self):
        d = self.api.drug({"q": ["metformin"]})
        self.assertIsNotNone(d["drug"])
        self.assertTrue(d["drug"]["sections"])
        for s in d["drug"]["sections"]:
            self.assertTrue(s["label"] and s["text"])
        import unittest.mock as mock
        raw = json.dumps({"results": [{
            "openfda": {"brand_name": ["Lipitor"], "generic_name": ["Atorvastatin"],
                        "manufacturer_name": ["Acme"], "route": ["ORAL"]},
            "indications_and_usage": ["Reduces LDL cholesterol."],
            "contraindications": ["Active liver disease."],
            "drug_interactions": ["Avoid strong CYP3A4 inhibitors."],
            "some_unmapped_field": ["ignored"]}]}).encode()
        with mock.patch.object(server, "fetch_url", return_value=raw):
            out = server.live_drug("atorvastatin")
        self.assertEqual(out["drug"]["generic"], "Atorvastatin")
        self.assertEqual(out["drug"]["brand"], "Lipitor")
        labels = [s["label"] for s in out["drug"]["sections"]]
        self.assertEqual(labels, ["Indications", "Contraindications", "Interactions"])  # order preserved, unmapped dropped

    def test_drug_no_results(self):
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=b'{"results": []}'):
            out = server.live_drug("zzznotadrug")
        self.assertIsNone(out["drug"])

    def test_drug_requires_query(self):
        with self.assertRaises(server.ApiError):
            self.api.drug({"q": ["  "]})

    def test_repos_sample_and_normalizer(self):
        d = self.api.repos({"window": ["week"]})
        self.assertTrue(d["repos"])
        for r in d["repos"]:
            self.assertTrue(r["name"] and "url" in r)
        import unittest.mock as mock
        raw = json.dumps({"items": [{"full_name": "a/b", "description": "x", "stargazers_count": 12,
                                     "language": "Python", "html_url": "https://github.com/a/b",
                                     "topics": ["ai", "x"]}]}).encode()
        with mock.patch.object(server, "fetch_url", return_value=raw):
            out = server.live_repos("day")
        self.assertEqual(out["repos"][0]["name"], "a/b")
        self.assertEqual(out["repos"][0]["stars"], 12)

    def test_repos_window_validated(self):
        d = self.api.repos({"window": ["bogus"]})
        self.assertEqual(d["window"], "week")

    def test_papers_sample_and_normalizer(self):
        d = self.api.papers({"cat": ["cs.CL"]})
        self.assertTrue(d["papers"])
        for p in d["papers"]:
            self.assertTrue(p["title"] and "url" in p)
        import unittest.mock as mock
        atom = (b'<feed><entry><title>A Paper</title><summary>Some abstract.</summary>'
                b'<published>2026-07-20T00:00:00Z</published>'
                b'<author><name>Jane Doe</name></author>'
                b'<link rel="alternate" href="https://arxiv.org/abs/1234"/></entry></feed>')
        with mock.patch.object(server, "fetch_url", return_value=atom):
            out = server.live_papers("cs.AI")
        self.assertEqual(out["papers"][0]["title"], "A Paper")
        self.assertTrue(out["papers"][0]["url"].endswith("1234"))

    def test_papers_category_validated(self):
        d = self.api.papers({"cat": ["cs.ZZ"]})
        self.assertEqual(d["category"], "cs.AI")

    def test_ai_news_sample_and_validation(self):
        d = self.api.ai_news({"topic": ["agents"]})
        self.assertEqual(d["topic"], "agents")
        self.assertTrue(d["items"])
        for it in d["items"]:
            self.assertTrue(it["title"] and it["url"])
        self.assertEqual(self.api.ai_news({"topic": ["bogus"]})["topic"], "claude")

    def test_commodities_sample_and_normalizer(self):
        d = self.api.commodities({})
        self.assertTrue(d["assets"])
        groups = {a["group"] for a in d["assets"]}
        self.assertTrue({"Metals", "Energy", "Rates"} <= groups)
        for a in d["assets"]:
            self.assertTrue(a["symbol"] and a["unit"])
            self.assertIsInstance(a["price"], (int, float))
        import unittest.mock as mock
        csv = (b"Symbol,Date,Time,Open,High,Low,Close,Volume\n"
               b"XAUUSD,2026-07-20,20:00:00,2380,2400,2370,2400,0\n")
        with mock.patch.object(server, "fetch_url", return_value=csv):
            out = server.live_commodities()
        gold = next(a for a in out["assets"] if a["id"] == "xauusd")
        self.assertEqual(gold["price"], 2400)
        self.assertEqual(gold["group"], "Metals")

    def test_changelog_sample_and_normalizer(self):
        d = self.api.changelog({})
        self.assertTrue(d["releases"])
        for r in d["releases"]:
            self.assertTrue(r["product"] and r["tag"] and r["url"])
        import unittest.mock as mock
        rel = json.dumps([
            {"tag_name": "v2.4.0", "name": "Claude Code v2.4.0",
             "body": "Subagents and background tasks.",
             "published_at": "2026-07-18T00:00:00Z",
             "html_url": "https://github.com/anthropics/claude-code/releases/tag/v2.4.0"},
        ]).encode()
        with mock.patch.object(server, "fetch_url", return_value=rel):
            out = server.live_changelog()
        self.assertEqual(out["source"], "live")
        self.assertTrue(any(r["tag"] == "v2.4.0" for r in out["releases"]))
        # newest first
        pubs = [r["published"] for r in out["releases"]]
        self.assertEqual(pubs, sorted(pubs, reverse=True))

    def test_pubmed_grounding_normalizer(self):
        import unittest.mock as mock
        esearch = json.dumps({"esearchresult": {"idlist": ["111"]}}).encode()
        esummary = json.dumps({"result": {"uids": ["111"], "111": {
            "title": "Grounding study", "source": "SAMJ", "pubdate": "2026",
            "authors": [{"name": "A B"}]}}}).encode()
        efetch = b"1. SAMJ. 2026.\nGrounding study.\nAbstract: important findings ..."
        with mock.patch.object(server, "fetch_url", side_effect=[esearch, esummary, efetch]):
            out = server.pubmed_grounding("tb")
        self.assertEqual(out["articles"][0]["pmid"], "111")
        self.assertIn("important findings", out["text"])

    def test_medchat_grounding_injected_and_sources_returned(self):
        import unittest.mock as mock
        api = server.Api(offline=True, data_dir=Path(tempfile.mkdtemp()))
        arts = [{"pmid": "999", "title": "Key SA trial", "journal": "Lancet",
                 "date": "2026", "authors": "X", "url": "u"}]

        class _Stream:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            text_stream = ["Answer "]
            def get_final_message(self):
                return type("R", (), {"content": [type("B", (), {"type": "text", "text": "Answer [999]"})()],
                                      "stop_reason": "end_turn"})()

        client = mock.Mock()
        client.messages.stream.return_value = _Stream()
        with mock.patch.object(type(api.assistant), "mode", new_callable=mock.PropertyMock, return_value="claude"), \
             mock.patch.object(api.assistant, "_get_client", return_value=client), \
             mock.patch.object(api, "pubmed_grounding_cached", return_value={"articles": arts, "text": "abstract text"}):
            events = list(api.assistant.med_chat_stream({"messages": [{"role": "user", "content": "SA TB treatment?"}]}))
        done = next(p for k, p in events if k == "done")
        self.assertEqual(done["sources"], arts)          # citations surfaced to the UI
        system = client.messages.stream.call_args.kwargs["system"]
        joined = " ".join(b["text"] for b in system)
        self.assertIn("999", joined)                     # grounding injected into context
        self.assertIn("Key SA trial", joined)

    def test_medchat_local_fallback(self):
        events = list(self.api.assistant.med_chat_stream({"messages": [{"role": "user", "content": "hi"}]}))
        kinds = [e[0] for e in events]
        self.assertIn("done", kinds)
        done = next(p for k, p in events if k == "done")
        self.assertEqual(done["mode"], "local")
        self.assertIn("South African", done["content"][0]["text"])

    def test_medicine_is_a_default_topic(self):
        self.assertIn("medicine", self.api.feeds.topics())
        self.assertTrue(self.api.news({"topic": ["medicine"], "limit": ["5"]})["items"])

    def test_podcast_sample_and_normalizer(self):
        d = self.api.podcast({"url": ["https://example.com/feed.xml"]})
        self.assertTrue(d["episodes"])
        rss = (b'<?xml version="1.0"?><rss><channel><title>My Show</title>'
               b'<item><title>Ep 1</title><enclosure url="https://cdn/ep1.mp3" type="audio/mpeg"/>'
               b'<itunes:duration xmlns:itunes="x">42:00</itunes:duration></item>'
               b'<item><title>No audio</title></item></channel></rss>')
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=rss):
            out = server.live_podcast("https://example.com/feed.xml")
        self.assertEqual(out["show"], "My Show")
        self.assertEqual(len(out["episodes"]), 1)     # audio-less item skipped
        self.assertEqual(out["episodes"][0]["audio"], "https://cdn/ep1.mp3")

    def test_podcast_bad_url_rejected(self):
        with self.assertRaises(server.ApiError):
            self.api.podcast({"url": ["not-a-url"]})

    def test_quakes_sample_shape(self):
        d = self.api.quakes({})
        self.assertTrue(d["quakes"])
        q = d["quakes"][0]
        for k in ("mag", "place", "time"):
            self.assertIn(k, q)

    def test_quakes_normalizer_from_fixture(self):
        raw = {"features": [
            {"properties": {"mag": 5.4, "place": "Somewhere", "time": 1000, "url": "u", "tsunami": 1}},
            {"properties": {"mag": None, "place": "bad"}},
        ]}
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=json.dumps(raw).encode()):
            out = server.live_quakes()
        self.assertEqual(len(out["quakes"]), 1)     # mag-less dropped
        self.assertTrue(out["quakes"][0]["tsunami"])

    def test_fx_sample_and_base(self):
        d = self.api.fx({"base": ["EUR"], "symbols": ["USD,GBP"]})
        self.assertEqual(d["base"], "EUR")
        self.assertIn("USD", d["rates"])
        self.assertNotIn("EUR", d["rates"])          # base excluded
        self.assertGreater(d["rates"]["USD"], 0)

    def test_convert_rate_table(self):
        d = self.api.convert({})
        self.assertIn("USD", d["fiat"])
        self.assertEqual(d["fiat"]["USD"], 1.0)
        self.assertTrue(d["coins"])                   # at least one coin priced
        sym, info = next(iter(d["coins"].items()))
        self.assertIn("name", info)
        self.assertGreater(info["usd"], 0)
        # a coin↔fiat cross-rate is computable from the table
        self.assertGreater(info["usd"] / d["fiat"]["USD"], 0)

    def test_social_sample_shapes(self):
        for net in ("hn", "lobsters", "reddit"):
            d = self.api.social({"network": [net]})
            self.assertEqual(d["network"], net)
            self.assertTrue(d["items"])
            i = d["items"][0]
            for key in ("title", "url", "source", "score", "comments"):
                self.assertIn(key, i)

    def test_social_reddit_sub_sanitized(self):
        d = self.api.social({"network": ["reddit"], "sub": ["home lab/../etc"]})
        self.assertTrue(d["items"])  # sanitized, no crash

    def test_social_unknown_network_rejected(self):
        with self.assertRaises(server.ApiError):
            self.api.social({"network": ["myspace"]})

    def test_social_reddit_normalizer_from_fixture(self):
        raw = {"data": {"children": [
            {"data": {"title": "Cool build", "author": "bob", "subreddit": "diy",
                      "score": 42, "num_comments": 7, "permalink": "/r/diy/x",
                      "url_overridden_by_dest": "https://example.com/x"}},
            {"data": {"title": "pinned", "stickied": True, "permalink": "/r/diy/y"}},
        ]}}
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=json.dumps(raw).encode()):
            out = server.live_social_reddit("diy")
        self.assertEqual(len(out["items"]), 1)  # stickied dropped
        self.assertEqual(out["items"][0]["author"], "bob")
        self.assertEqual(out["items"][0]["score"], 42)

    def test_scores_normalizer_from_espn_fixture(self):
        raw = {
            "events": [{
                "id": "1", "date": "2026-07-16T00:00:00Z",
                "status": {"type": {"state": "in", "shortDetail": "Q2"}, "displayClock": "5:00", "period": 2},
                "competitions": [{"competitors": [
                    {"homeAway": "home", "team": {"abbreviation": "KC", "displayName": "Chiefs"}, "score": "14"},
                    {"homeAway": "away", "team": {"abbreviation": "BUF", "displayName": "Bills"}, "score": "10"},
                ]}],
            }],
        }
        import unittest.mock as mock
        with mock.patch.object(server, "fetch_url", return_value=json.dumps(raw).encode()):
            out = server.live_scores("nfl")
        g = out["games"][0]
        self.assertEqual(g["state"], "in")
        self.assertEqual(g["home"]["abbr"], "KC")
        self.assertEqual(g["away"]["abbr"], "BUF")
        self.assertEqual(g["home"]["score"], "14")


class MemoryAndToolsTests(unittest.TestCase):
    def setUp(self):
        server.CACHE.clear()
        self.api = server.Api(offline=True, data_dir=Path(tempfile.mkdtemp()))

    def test_memory_roundtrip(self):
        self.api.memory_append("Likes espresso")
        self.api.memory_append("Runs on Tuesdays")
        memory = self.api.memory_read()
        self.assertIn("Likes espresso", memory)
        self.assertIn("Runs on Tuesdays", memory)

    def test_memory_rejects_empty(self):
        with self.assertRaises(server.ApiError):
            self.api.memory_append("   ")

    def test_rank_facts_relevance(self):
        facts = ["likes espresso in the morning", "allergic to penicillin",
                 "works as a cardiologist", "prefers dark mode"]
        top = server.rank_facts(facts, "which drugs am I allergic to?", 2)
        self.assertEqual(top[0], "allergic to penicillin")

    def test_rank_facts_empty_query_is_recency(self):
        facts = ["oldest fact", "middle fact", "newest fact"]
        self.assertEqual(server.rank_facts(facts, "", 2), ["newest fact", "middle fact"])

    def test_rank_facts_no_match_falls_back(self):
        facts = ["likes espresso", "runs on tuesdays"]
        out = server.rank_facts(facts, "xylophone zeppelin", 1)
        self.assertEqual(out, ["runs on tuesdays"])   # newest, not empty

    def test_memory_recall_integration(self):
        for f in ("enjoys hiking", "allergic to shellfish", "uses vim"):
            self.api.memory_append(f)
        out = self.api.memory_recall("shellfish reaction", 2)
        self.assertTrue(any("allergic to shellfish" in f for f in out))
        self.assertEqual("allergic to shellfish" in out[0], True)  # ranked first

    def test_server_tools(self):
        run = self.api.assistant.run_server_tool
        self.assertIn("New York", run("get_weather", {}))
        self.assertIn("BTC", run("get_markets", {}))
        self.assertIn("Global index", run("get_worldstate", {}))
        self.assertIn("1.", run("get_news", {"topic": "science"}))
        self.assertEqual(run("remember", {"fact": "test fact"}), "Saved to long-term memory.")
        self.assertIn("test fact", self.api.memory_read())

    def test_unknown_tool_rejected(self):
        with self.assertRaises(ValueError):
            self.api.assistant.run_server_tool("format_disk", {})

    def test_local_automation_intents(self):
        a = self.api.assistant
        r = a._chat_local([{"role": "user", "content": "alert me if BTC moves 5%"}], {})
        tools = [b for b in r["content"] if b["type"] == "tool_use"]
        self.assertEqual(tools[0]["name"], "create_automation")
        self.assertEqual(tools[0]["input"]["symbol"], "BTC")
        self.assertEqual(tools[0]["input"]["percent"], 5.0)

        r = a._chat_local([{"role": "user", "content": "alert me when BTC goes above $100,000"}], {})
        inp = [b for b in r["content"] if b["type"] == "tool_use"][0]["input"]
        self.assertEqual(inp["symbol"], "BTC")
        self.assertEqual(inp["price"], 100000.0)
        self.assertEqual(inp["direction"], "above")

        r = a._chat_local([{"role": "user", "content": "alert me if ETH RSI below 30"}], {})
        inp = [b for b in r["content"] if b["type"] == "tool_use"][0]["input"]
        self.assertEqual(inp["symbol"], "ETH")
        self.assertEqual(inp["rsi"], 30.0)
        self.assertEqual(inp["direction"], "below")

        r = a._chat_local([{"role": "user", "content": "every morning at 7 brief me"}], {})
        tools = [b for b in r["content"] if b["type"] == "tool_use"]
        self.assertEqual(tools[0]["input"]["time"], "07:00")

        r = a._chat_local([{"role": "user", "content": "brief me daily at 8pm"}], {})
        tools = [b for b in r["content"] if b["type"] == "tool_use"]
        self.assertEqual(tools[0]["input"]["time"], "20:00")

        r = a._chat_local([{"role": "user", "content": "what's the weather like?"}], {})
        self.assertIn("New York", r["content"][0]["text"])


class SourceRegistryTests(unittest.TestCase):
    def setUp(self):
        server.CACHE.clear()
        self.api = server.Api(offline=True, data_dir=Path(tempfile.mkdtemp()))

    def test_registry_entries_well_formed(self):
        self.assertTrue(server.SOURCES)
        for name, spec in server.SOURCES.items():
            self.assertIn("ttl", spec, name)
            self.assertGreater(spec["ttl"], 0, name)
            self.assertTrue(callable(spec["live"]), name)
            self.assertTrue(callable(spec["sample"]), name)

    def test_fetch_source_no_args_caches(self):
        first = self.api.fetch_source("quakes")
        self.assertEqual(first["source"], "sample")   # offline → sample path
        self.assertIsNotNone(server.CACHE.get("quakes"))
        self.assertIs(self.api.fetch_source("quakes"), first)  # served from cache

    def test_fetch_source_folds_args_into_key(self):
        self.api.fetch_source("standings", "nba")
        self.assertIsNotNone(server.CACHE.get("standings:nba"))
        self.api.fetch_source("teamsched", "nba", "BOS")
        self.assertIsNotNone(server.CACHE.get("teamsched:nba:bos"))  # lowercased

    def test_registered_methods_still_route(self):
        # the public methods now delegate to fetch_source; shapes unchanged
        self.assertTrue(self.api.quakes({})["quakes"] if "quakes" in self.api.quakes({}) else True)
        self.assertEqual(self.api.standings({"league": ["nba"]})["league"], "nba")
        self.assertTrue(self.api.gaming_free({}))
        self.assertTrue(self.api.crypto_trending({}))


class ScoreLevelTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(server.score_level(75), "stable")
        self.assertEqual(server.score_level(74.9), "watch")
        self.assertEqual(server.score_level(60), "watch")
        self.assertEqual(server.score_level(59.9), "elevated")
        self.assertEqual(server.score_level(40), "elevated")
        self.assertEqual(server.score_level(39.9), "critical")


class StateStoreTests(unittest.TestCase):
    def setUp(self):
        self.store = server.StateStore(Path(tempfile.mkdtemp()) / "hub.db")

    def test_empty(self):
        self.assertEqual(self.store.get(), {"rev": 0, "updated": None, "state": None})
        self.assertEqual(self.store.rev(), 0)

    def test_put_and_get_roundtrip(self):
        ok, rev = self.store.put({"tasks": [1, 2]}, None)
        self.assertTrue(ok)
        self.assertEqual(rev, 1)
        data = self.store.get()
        self.assertEqual(data["state"], {"tasks": [1, 2]})
        self.assertIsNotNone(data["updated"])

    def test_optimistic_concurrency(self):
        self.store.put({"v": 1}, None)
        ok, rev = self.store.put({"v": 2}, 1)
        self.assertTrue(ok)
        self.assertEqual(rev, 2)
        ok, rev = self.store.put({"v": 99}, 1)  # stale base rev
        self.assertFalse(ok)
        self.assertEqual(rev, 2)
        self.assertEqual(self.store.get()["state"], {"v": 2})

    def test_force_put_without_base(self):
        self.store.put({"v": 1}, None)
        ok, rev = self.store.put({"v": 2}, None)
        self.assertTrue(ok)
        self.assertEqual(rev, 2)


class HttpTests(unittest.TestCase):
    """End-to-end over a real socket, offline mode."""

    @classmethod
    def setUpClass(cls):
        server.CACHE.clear()
        cls.httpd = server.make_server(
            "127.0.0.1", 0, offline=True, data_dir=Path(tempfile.mkdtemp()))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def get(self, path):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers, err.read()

    def get_json(self, path):
        status, _, body = self.get(path)
        return status, json.loads(body)

    def test_index_served(self):
        status, headers, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"Hermes Hub", body)

    def test_static_js_mime(self):
        status, headers, _ = self.get("/js/main.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["Content-Type"])

    def test_missing_file_404(self):
        status, _, _ = self.get("/nope.txt")
        self.assertEqual(status, 404)

    def test_traversal_blocked(self):
        status, _, body = self.get("/..%2f..%2fserver.py")
        self.assertEqual(status, 404)
        self.assertNotIn(b"TTLCache", body)

    def test_api_health(self):
        status, data = self.get_json("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(data["ok"])
        self.assertTrue(data["offline"])

    def test_api_news_and_error(self):
        status, data = self.get_json("/api/news?topic=science&limit=3")
        self.assertEqual(status, 200)
        self.assertEqual(data["topic"], "science")
        self.assertLessEqual(len(data["items"]), 3)

        status, _, body = self.get("/api/news?topic=bogus")
        self.assertEqual(status, 400)
        self.assertIn("unknown topic", json.loads(body)["error"])

    def test_api_worldstate(self):
        status, data = self.get_json("/api/worldstate")
        self.assertEqual(status, 200)
        self.assertTrue(data["domains"])

    def post(self, path, payload):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    def test_assistant_status_endpoint(self):
        status, data = self.get_json("/api/assistant/status")
        self.assertEqual(status, 200)
        self.assertIn(data["mode"], ("local", "claude"))

    def test_assistant_chat_endpoint(self):
        status, data = self.post("/api/assistant/chat", {
            "messages": [{"role": "user", "content": "add task test the endpoint"}],
            "context": {},
        })
        self.assertEqual(status, 200)
        tools = [b for b in data["content"] if b["type"] == "tool_use"]
        self.assertEqual(tools[0]["name"], "add_task")

    def test_assistant_chat_rejects_bad_body(self):
        status, data = self.post("/api/assistant/chat", {"messages": []})
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    def test_assistant_summarize_endpoint(self):
        status, data = self.post("/api/assistant/summarize", {
            "kind": "note", "title": "t",
            "content": "One important sentence about the project deadline. Another sentence.",
        })
        self.assertEqual(status, 200)
        self.assertTrue(data["summary"])

    def test_assistant_chat_stream_sse(self):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/api/assistant/chat-stream",
            data=json.dumps({
                "messages": [{"role": "user", "content": "add task stream test"}],
                "context": {},
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertIn("text/event-stream", resp.headers["Content-Type"])
            raw = resp.read().decode()
        events = {}
        for frame in raw.strip().split("\n\n"):
            lines = frame.split("\n")
            name = next(l[7:] for l in lines if l.startswith("event: "))
            data = next(l[6:] for l in lines if l.startswith("data: "))
            events[name] = json.loads(data)
        self.assertIn("delta", events)
        self.assertIn("done", events)
        # the done payload matches the non-streaming chat shape
        done = events["done"]
        self.assertIn(done["stop_reason"], ("end_turn", "tool_use"))
        tools = [b for b in done["content"] if b["type"] == "tool_use"]
        self.assertEqual(tools[0]["name"], "add_task")

    def test_assistant_chat_stream_rejects_bad_body(self):
        status, data = self.post("/api/assistant/chat-stream", {"messages": []})
        self.assertEqual(status, 400)
        self.assertIn("error", data)

    def test_assistant_briefing_endpoint(self):
        status, data = self.post("/api/assistant/briefing", {"context": {
            "tasks": [{"name": "Today", "items": [{"text": "renew passport", "done": False}]}],
        }})
        self.assertEqual(status, 200)
        self.assertIn("renew passport", data["briefing"])

    def test_post_unknown_route_404(self):
        status, _ = self.post("/api/nope", {})
        self.assertEqual(status, 404)

    def test_state_sync_roundtrip(self):
        status, data = self.get_json("/api/state")
        self.assertEqual(status, 200)
        base = data["rev"]

        status, data = self.post("/api/state", {"state": {"version": 1, "layout": []}, "baseRev": base})
        self.assertEqual(status, 200)
        new_rev = data["rev"]
        self.assertEqual(new_rev, base + 1)

        status, data = self.get_json("/api/state/rev")
        self.assertEqual(data["rev"], new_rev)

        status, data = self.get_json("/api/state")
        self.assertEqual(data["state"]["version"], 1)

        # stale write is refused
        status, data = self.post("/api/state", {"state": {"version": 2}, "baseRev": base})
        self.assertEqual(status, 409)

        # bad payloads are rejected
        status, _ = self.post("/api/state", {"state": "not-an-object"})
        self.assertEqual(status, 400)


class AuthHttpTests(unittest.TestCase):
    """Token-locked server: API requires the bearer token, static does not."""

    TOKEN = "s3cret-code"

    @classmethod
    def setUpClass(cls):
        server.CACHE.clear()
        cls.httpd = server.make_server(
            "127.0.0.1", 0, offline=True, token=cls.TOKEN, data_dir=Path(tempfile.mkdtemp()))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, path, token=None, payload=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.read()

    def test_api_requires_token(self):
        status, _ = self.request("/api/news?topic=top")
        self.assertEqual(status, 401)
        status, _ = self.request("/api/state")
        self.assertEqual(status, 401)
        status, _ = self.request("/api/assistant/chat", payload={"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 401)

    def test_wrong_token_rejected(self):
        status, _ = self.request("/api/news?topic=top", token="wrong")
        self.assertEqual(status, 401)

    def test_correct_token_accepted(self):
        status, body = self.request("/api/news?topic=top", token=self.TOKEN)
        self.assertEqual(status, 200)
        self.assertIn(b"items", body)
        status, _ = self.request("/api/state", token=self.TOKEN)
        self.assertEqual(status, 200)

    def test_health_stays_open_for_lock_screen(self):
        status, _ = self.request("/api/health")
        self.assertEqual(status, 200)

    def test_static_shell_served_without_token(self):
        status, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn(b"Hermes Hub", body)

    def test_body_token_accepted_for_beacon_posts(self):
        # sendBeacon cannot set headers; the token may ride in the JSON body.
        status, _ = self.request("/api/state", payload={
            "state": {"version": 1, "layout": []}, "token": self.TOKEN})
        self.assertEqual(status, 200)

    def test_wrong_body_token_rejected(self):
        status, _ = self.request("/api/state", payload={
            "state": {"version": 1, "layout": []}, "token": "nope"})
        self.assertEqual(status, 401)


ICS_SAMPLE = "\r\n".join([
    "BEGIN:VCALENDAR",
    "VERSION:2.0",
    "BEGIN:VEVENT",
    "SUMMARY:One-off meeting with a very long",
    " ly folded line",
    "DTSTART:20260720T093000Z",
    "END:VEVENT",
    "BEGIN:VEVENT",
    "SUMMARY:All-day thing",
    "DTSTART;VALUE=DATE:20260722",
    "END:VEVENT",
    "END:VCALENDAR",
])


class IcsParserTests(unittest.TestCase):
    WINDOW = (date(2026, 7, 14), date(2026, 8, 14))

    def parse(self, body_lines, window=None):
        text = "\n".join(["BEGIN:VCALENDAR", "BEGIN:VEVENT", *body_lines,
                          "END:VEVENT", "END:VCALENDAR"])
        start, end = window or self.WINDOW
        return ics.parse_ics(text, "Test", start, end)

    def test_unfolding_and_timed_plus_allday(self):
        events = ics.parse_ics(ICS_SAMPLE, "Work", *self.WINDOW)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["title"], "One-off meeting with a very longly folded line")
        self.assertEqual(events[0]["date"], "2026-07-20")
        self.assertEqual(events[0]["time"], "09:30")
        self.assertEqual(events[1]["time"], None)
        self.assertEqual(events[1]["calendar"], "Work")

    def test_single_event_outside_window_dropped(self):
        events = self.parse(["SUMMARY:Old", "DTSTART:20250101T100000"])
        self.assertEqual(events, [])

    def test_daily_rrule_fast_forwards_old_start(self):
        events = self.parse([
            "SUMMARY:Run", "DTSTART:20240101T070000", "RRULE:FREQ=DAILY"])
        self.assertEqual(len(events), 32)  # every day of the window
        self.assertEqual(events[0]["date"], "2026-07-14")
        self.assertTrue(all(e["time"] == "07:00" for e in events))

    def test_count_semantics_count_from_dtstart(self):
        # 5 occurrences starting 2 days before the window → only 3 land inside.
        start = self.WINDOW[0] - timedelta(days=2)
        events = self.parse([
            "SUMMARY:Sprint", f"DTSTART:{start.strftime('%Y%m%d')}T090000",
            "RRULE:FREQ=DAILY;COUNT=5"])
        self.assertEqual(len(events), 3)

    def test_until_is_inclusive(self):
        events = self.parse([
            "SUMMARY:Ends", "DTSTART;VALUE=DATE:20260714",
            "RRULE:FREQ=DAILY;UNTIL=20260716"])
        self.assertEqual([e["date"] for e in events],
                         ["2026-07-14", "2026-07-15", "2026-07-16"])

    def test_exdate_removes_occurrence(self):
        events = self.parse([
            "SUMMARY:Standup", "DTSTART;VALUE=DATE:20260714",
            "RRULE:FREQ=DAILY;UNTIL=20260717", "EXDATE;VALUE=DATE:20260715"])
        self.assertEqual([e["date"] for e in events],
                         ["2026-07-14", "2026-07-16", "2026-07-17"])

    def test_weekly_byday(self):
        events = self.parse([
            "SUMMARY:Sync", "DTSTART:20260713T100000",  # a Monday
            "RRULE:FREQ=WEEKLY;BYDAY=MO,FR",
        ], window=(date(2026, 7, 13), date(2026, 7, 26)))
        self.assertEqual([e["date"] for e in events],
                         ["2026-07-13", "2026-07-17", "2026-07-20", "2026-07-24"])

    def test_weekly_interval(self):
        events = self.parse([
            "SUMMARY:Payday", "DTSTART;VALUE=DATE:20260701",
            "RRULE:FREQ=WEEKLY;INTERVAL=2",
        ], window=(date(2026, 7, 1), date(2026, 8, 1)))
        self.assertEqual([e["date"] for e in events],
                         ["2026-07-01", "2026-07-15", "2026-07-29"])

    def test_monthly_and_yearly(self):
        monthly = self.parse([
            "SUMMARY:Rent", "DTSTART;VALUE=DATE:20260115", "RRULE:FREQ=MONTHLY",
        ], window=(date(2026, 7, 1), date(2026, 9, 30)))
        self.assertEqual([e["date"] for e in monthly],
                         ["2026-07-15", "2026-08-15", "2026-09-15"])
        yearly = self.parse([
            "SUMMARY:Birthday", "DTSTART;VALUE=DATE:19900722", "RRULE:FREQ=YEARLY",
        ])
        self.assertEqual([e["date"] for e in yearly], ["2026-07-22"])

    def test_summary_unescaping_and_missing_fields(self):
        events = self.parse([
            "SUMMARY:Dinner\\, wine \\; cheese", "DTSTART;VALUE=DATE:20260720"])
        self.assertEqual(events[0]["title"], "Dinner, wine ; cheese")
        self.assertEqual(self.parse(["DTSTART;VALUE=DATE:20260720"]), [])
        self.assertEqual(self.parse(["SUMMARY:No date"]), [])


class CalendarConfigTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "calendars.json"
        self.config = server.CalendarConfig(self.path)

    def test_add_list_remove_roundtrip(self):
        self.config.add("Personal", "https://example.org/basic.ics")
        self.assertEqual(self.config.list(), [
            {"name": "Personal", "url": "https://example.org/basic.ics"}])
        self.config.remove("https://example.org/basic.ics")
        self.assertEqual(self.config.list(), [])

    def test_webcal_rewritten_to_https(self):
        self.config.add("Apple", "webcal://p01-cal.icloud.com/pub.ics")
        self.assertTrue(self.config.list()[0]["url"].startswith("https://"))

    def test_validation(self):
        with self.assertRaises(server.ApiError):
            self.config.add("Bad", "ftp://example.org/cal.ics")
        with self.assertRaises(server.ApiError):
            self.config.add("", "https://example.org/cal.ics")
        with self.assertRaises(server.ApiError):
            self.config.remove("https://example.org/never-added.ics")
        self.config.add("Dup", "https://example.org/cal.ics")
        with self.assertRaises(server.ApiError):
            self.config.add("Dup again", "https://example.org/cal.ics")

    def test_limit_enforced(self):
        for i in range(server.CalendarConfig.MAX_CALENDARS):
            self.config.add(f"c{i}", f"https://example.org/{i}.ics")
        with self.assertRaises(server.ApiError):
            self.config.add("extra", "https://example.org/extra.ics")

    def test_persistence_survives_reload(self):
        self.config.add("Keep", "https://example.org/keep.ics")
        reloaded = server.CalendarConfig(self.path)
        self.assertEqual(reloaded.list(), self.config.list())


class CalendarHttpTests(unittest.TestCase):
    """Subscribe to the demo.ics the server itself hosts, then read /api/events."""

    @classmethod
    def setUpClass(cls):
        server.CACHE.clear()
        cls.httpd = server.make_server(
            "127.0.0.1", 0, offline=True, data_dir=Path(tempfile.mkdtemp()))
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, path, payload=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    def test_demo_ics_served_with_calendar_mime(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/demo.ics") as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/calendar", resp.headers["Content-Type"])

    def test_subscribe_events_unsubscribe_flow(self):
        status, data = self.request("/api/calendars")
        self.assertEqual(status, 200)
        self.assertEqual(data["calendars"], [])

        demo_url = f"http://127.0.0.1:{self.port}/demo.ics"
        status, data = self.request("/api/calendars", {"op": "add", "name": "Demo", "url": demo_url})
        self.assertEqual(status, 200)
        self.assertEqual(data["calendars"][0]["name"], "Demo")

        status, data = self.request("/api/events?days=7")
        self.assertEqual(status, 200)
        self.assertEqual(data["failures"], [])
        titles = {e["title"] for e in data["events"]}
        self.assertIn("Demo: morning run", titles)  # FREQ=DAILY reaches any window
        run = next(e for e in data["events"] if e["title"] == "Demo: morning run")
        self.assertEqual(run["time"], "07:00")
        self.assertEqual(run["calendar"], "Demo")

        # cache is epoch-keyed: removing the calendar empties the next read
        status, data = self.request("/api/calendars", {"op": "remove", "url": demo_url})
        self.assertEqual(status, 200)
        status, data = self.request("/api/events?days=7")
        self.assertEqual(data["events"], [])

    def test_calendars_op_validation(self):
        status, data = self.request("/api/calendars", {"op": "add", "name": "Bad", "url": "not-a-url"})
        self.assertEqual(status, 400)
        status, data = self.request("/api/calendars", {"op": "bogus"})
        self.assertEqual(status, 400)
        status, data = self.request("/api/events?days=zero")
        self.assertEqual(status, 400)

    def test_weather_sample_includes_aqi_and_sun(self):
        status, data = self.request("/api/weather?lat=40.7&lon=-74")
        self.assertEqual(status, 200)
        self.assertEqual(data["source"], "sample")
        self.assertIsInstance(data["current"]["aqi"], int)
        self.assertRegex(data["sun"]["sunrise"], r"^\d\d:\d\d$")
        self.assertRegex(data["sun"]["sunset"], r"^\d\d:\d\d$")


class BackupTests(unittest.TestCase):
    def setUp(self):
        server.CACHE.clear()
        data_dir = Path(tempfile.mkdtemp())
        store = server.StateStore(data_dir / "hub.db")
        self.api = server.Api(offline=True, state_store=store, data_dir=data_dir)

    def test_backup_creates_snapshot_and_lists(self):
        self.api.state_store.put({"version": 1, "layout": [], "note": "hi"}, None)
        info = self.api.backup_now({})
        self.assertRegex(info["name"], r"^hub-[0-9-]+\.json$")
        self.assertGreater(info["size"], 0)
        listed = self.api.backups_list({})["backups"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["name"], info["name"])

    def test_backup_prunes_to_keep_limit(self):
        for _ in range(server.BACKUP_KEEP + 5):
            self.api.backup_now({})
        self.assertEqual(len(self.api.backups_list({})["backups"]), server.BACKUP_KEEP)

    def test_backup_get_returns_snapshot_and_validates_name(self):
        self.api.memory_append("marker fact")
        name = self.api.backup_now({})["name"]
        snap = self.api.backup_get({"name": [name]})
        self.assertEqual(snap["kind"], "hermes-hub-backup")
        self.assertIn("marker fact", snap["memory"])
        with self.assertRaises(server.ApiError):
            self.api.backup_get({"name": ["../etc/passwd"]})
        with self.assertRaises(server.ApiError):
            self.api.backup_get({"name": ["hub-nope.json"]})

    def test_backup_import_roundtrips_via_get(self):
        self.api.memory_append("offbox fact")
        name = self.api.backup_now({})["name"]
        snap = self.api.backup_get({"name": [name]})
        # simulate wiping the box, then re-importing the downloaded snapshot
        for f in self.api.backups_dir.glob("hub-*.json"):
            f.unlink()
        info = self.api.backup_import({"snapshot": snap})
        self.assertRegex(info["name"], r"^hub-[0-9-]+\.json$")
        self.assertEqual(len(self.api.backups_list({})["backups"]), 1)
        # and it restores
        result = self.api.backup_restore({"name": info["name"]})
        self.assertIn("memory", result["restored"])

    def test_backup_import_rejects_non_backup(self):
        with self.assertRaises(server.ApiError):
            self.api.backup_import({"snapshot": {"kind": "something-else"}})
        with self.assertRaises(server.ApiError):
            self.api.backup_import({"snapshot": "not-a-dict"})

    def test_restore_roundtrips_state_and_config(self):
        self.api.calendars.add("Work", "https://example.org/work.ics")
        self.api.feeds.add_topic("gadgets")
        self.api.feeds.add_source("gadgets", "Verge", "https://example.org/verge.xml")
        self.api.memory_append("the dog is named Rex")
        self.api.state_store.put({"version": 1, "layout": [], "marker": "A"}, None)
        name = self.api.backup_now({})["name"]

        # mutate everything, then restore
        self.api.calendars.remove("https://example.org/work.ics")
        self.api.feeds.remove_topic("gadgets")
        self.api.state_store.put({"version": 1, "layout": [], "marker": "B"}, None)

        result = self.api.backup_restore({"name": name})
        self.assertEqual(set(result["restored"]),
                         {"state", "feeds", "calendars", "automations", "memory", "agent_notes"})
        self.assertEqual(self.api.state_store.get()["state"]["marker"], "A")
        self.assertTrue(any(c["url"] == "https://example.org/work.ics"
                            for c in self.api.calendars.list()))
        self.assertIn("gadgets", self.api.feeds.topics())
        self.assertIn("Rex", self.api.memory_read())

    def test_restore_rejects_bad_name_and_missing(self):
        with self.assertRaises(server.ApiError):
            self.api.backup_restore({"name": "../etc/passwd"})
        with self.assertRaises(server.ApiError):
            self.api.backup_restore({"name": "hub-does-not-exist.json"})

    def test_backup_automation_action_fires(self):
        rule = self.api.automations.create_rule({
            "name": "Nightly backup",
            "trigger": {"type": "daily", "time": "00:00"},
            "action": {"type": "backup"},
        })
        self.api.automations._fire(rule)
        self.assertEqual(len(self.api.backups_list({})["backups"]), 1)
        notifs = self.api.automations.notifications_after(0)["notifications"]
        self.assertTrue(any("Snapshot" in n["body"] for n in notifs))


class RouterTests(unittest.TestCase):
    def test_cheap_by_default(self):
        r = router_mod.Router()
        self.assertEqual(r.route("summarize")["tier"], "fast")
        self.assertEqual(r.route("chat", "what's the weather today")["tier"], "core")
        self.assertEqual(r.route("briefing")["tier"], "core")

    def test_unknown_task_falls_back_to_core(self):
        self.assertEqual(router_mod.Router().route("mystery")["tier"], "core")

    def test_escalates_on_hard_patterns(self):
        r = router_mod.Router()
        for text in [
            "design the architecture for a distributed system",
            "is there a security vulnerability in this code",
            "review my investment portfolio",
            "change your own routing config",
        ]:
            self.assertEqual(r.route("chat", text)["tier"], "deep", text)

    def test_deep_cap_downgrades_to_core(self):
        r = router_mod.Router(max_deep_per_hour=2)
        tiers = [r.route("chat", "security exploit")["tier"] for _ in range(4)]
        self.assertEqual(tiers, ["deep", "deep", "core", "core"])
        self.assertEqual(r.deep_calls_last_hour(), 2)

    def test_pin_forces_one_model_but_records_tier(self):
        r = router_mod.Router(pin="claude-opus-4-8")
        deep = r.route("chat", "design the architecture")
        fast = r.route("summarize")
        self.assertEqual(deep["model"], "claude-opus-4-8")
        self.assertEqual(fast["model"], "claude-opus-4-8")
        self.assertTrue(deep["pinned"])
        self.assertEqual(deep["requested_tier"], "deep")  # tier still tracked
        self.assertEqual(fast["tier"], "fast")

    def test_snapshot_shape(self):
        r = router_mod.Router()
        r.route("chat", "hi")
        snap = r.snapshot()
        self.assertEqual(set(snap["tiers"]), {"fast", "core", "deep"})
        self.assertIn("deep_calls_last_hour", snap)
        self.assertEqual(len(snap["recent"]), 1)

    def test_assistant_status_exposes_tiers_when_claude(self):
        from unittest import mock
        a = assistant.Assistant()
        # report claude-mode without real credentials/SDK
        with mock.patch.object(type(a), "mode", new_callable=mock.PropertyMock,
                               return_value="claude"):
            status = a.status()
        self.assertEqual(set(status["routing"]["tiers"]), {"fast", "core", "deep"})
        self.assertIsNotNone(status["model"])

    def test_assistant_status_local_mode_hides_routing(self):
        status = assistant.Assistant().status()
        self.assertEqual(status["mode"], "local")
        self.assertIsNone(status["routing"])

    def test_file_override_applies_when_no_env(self):
        r = router_mod.Router(overrides={"core": "claude-custom-core"})
        self.assertEqual(r.tiers["core"], "claude-custom-core")
        self.assertEqual(r.route("chat", "hi")["model"], "claude-custom-core")
        # untouched tiers keep their defaults
        self.assertEqual(r.tiers["fast"], router_mod._TIER_DEFAULTS["fast"])

    def test_env_beats_file_override(self):
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"HERMES_HUB_MODEL_DEEP": "claude-from-env"}):
            r = router_mod.Router(overrides={"deep": "claude-from-file"})
            self.assertEqual(r.tiers["deep"], "claude-from-env")
            self.assertTrue(r.env_locked()["deep"])

    def test_set_overrides_updates_live(self):
        r = router_mod.Router()
        r.set_overrides({"fast": "claude-x"})
        self.assertEqual(r.tiers["fast"], "claude-x")
        r.set_overrides({})  # cleared → default returns
        self.assertEqual(r.tiers["fast"], router_mod._TIER_DEFAULTS["fast"])

    def test_routing_set_persists_and_validates(self):
        api = server.Api(offline=True, data_dir=Path(tempfile.mkdtemp()))
        snap = api.routing_set({"overrides": {"core": "claude-sonnet-x"}})
        self.assertEqual(snap["tiers"]["core"], "claude-sonnet-x")
        self.assertEqual(snap["overrides"]["core"], "claude-sonnet-x")
        # persisted: a fresh Api on the same dir picks it up
        api2 = server.Api(offline=True, data_dir=api.data_dir)
        self.assertEqual(api2.assistant.router.tiers["core"], "claude-sonnet-x")
        # clearing removes it
        snap = api.routing_set({"overrides": {"core": ""}})
        self.assertEqual(snap["tiers"]["core"], router_mod._TIER_DEFAULTS["core"])
        # bad id rejected
        with self.assertRaises(server.ApiError):
            api.routing_set({"overrides": {"deep": "bad id!"}})


class IndicatorTests(unittest.TestCase):
    def test_sma_exact(self):
        out = ind.sma([1, 2, 3, 4, 5], 3)
        self.assertEqual(out, [None, None, 2.0, 3.0, 4.0])

    def test_ema_seed_and_length(self):
        out = ind.ema([1, 2, 3, 4, 5], 3)
        self.assertIsNone(out[0]); self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], 2.0)          # seeded with SMA of first 3
        # k = 0.5 → next = 4*0.5 + 2*0.5 = 3.0, then 5*0.5 + 3*0.5 = 4.0
        self.assertAlmostEqual(out[3], 3.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_rsi_all_gains_is_100_all_losses_is_0(self):
        up = list(range(1, 30))
        down = list(range(30, 1, -1))
        self.assertEqual(ind.rsi(up)[-1], 100.0)
        self.assertEqual(ind.rsi(down)[-1], 0.0)

    def test_rsi_warmup_is_none(self):
        vals = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1]  # len 15, period 14
        r = ind.rsi(vals, 14)
        self.assertTrue(all(v is None for v in r[:14]))
        self.assertIsNotNone(r[14])

    def test_macd_line_is_ema_diff(self):
        vals = [float(i) for i in range(1, 60)]
        m = ind.macd(vals)
        ef = ind.ema(vals, 12)[-1]
        es = ind.ema(vals, 26)[-1]
        self.assertAlmostEqual(m["macd"][-1], ef - es, places=6)

    def test_bollinger_symmetry(self):
        vals = [float(i % 5) for i in range(40)]
        b = ind.bollinger(vals, 20, 2)
        i = 30
        self.assertAlmostEqual(b["upper"][i] - b["mid"][i], b["mid"][i] - b["lower"][i], places=9)

    def test_read_signals_shape(self):
        vals = [float(i) for i in range(1, 80)]
        sigs = ind.read_signals(vals)
        labels = {s["label"] for s in sigs}
        self.assertIn("RSI(14)", labels)
        self.assertIn("MACD", labels)
        for s in sigs:
            self.assertIn(s["tone"], ("up", "down", "neutral"))


class EvolveTests(unittest.TestCase):
    def _api(self):
        return server.Api(offline=True, data_dir=Path(tempfile.mkdtemp()))

    def test_dedupe_memory(self):
        text = "# mem\n- (2026-01-01) cat is Milo\n- (2026-02-02) cat is Milo\n- dog is Rex\n"
        out, removed = evolve_mod.dedupe_memory(text)
        self.assertEqual(len(removed), 1)
        self.assertEqual(out.count("cat is Milo"), 1)
        self.assertIn("dog is Rex", out)

    def test_memory_prune_auto_applies(self):
        api = self._api()
        api.memory_append("cat is Milo")
        api.memory_append("cat is Milo")
        created = api.evolve.reflect()
        prune = [p for p in created if p["kind"] == "memory_prune"]
        self.assertEqual(len(prune), 1)
        self.assertEqual(prune[0]["status"], "auto-applied")     # auto policy
        self.assertEqual(api.memory_read().count("cat is Milo"), 1)
        self.assertTrue(prune[0]["snapshot"].startswith("hub-"))  # reversible

    def test_denied_tool_becomes_pending_addendum(self):
        api = self._api()
        for _ in range(2):
            api.telemetry.record({"kind": "tool", "name": "add_app", "ok": False, "approved": False})
        api.evolve.reflect()
        pending = [p for p in api.evolve.list_proposals() if p["status"] == "pending"]
        self.assertTrue(any(p["kind"] == "prompt_addendum" and "add_app" in p["title"] for p in pending))

    def test_apply_addendum_writes_agent_notes_and_injects(self):
        api = self._api()
        for _ in range(2):
            api.telemetry.record({"kind": "tool", "name": "open_url", "ok": False, "approved": False})
        api.evolve.reflect()
        pid = next(p["id"] for p in api.evolve.list_proposals() if p["status"] == "pending")
        result = api.evolve.apply(pid)
        self.assertEqual(result["status"], "applied")
        self.assertIn("open_url", api.agent_notes_read())
        # the guideline is injected into the system prompt
        system, _ = api.assistant._prepare_claude_request([{"role": "user", "content": "hi"}], {})
        self.assertTrue(any("open_url" in b["text"] for b in system))

    def test_dismiss_and_no_duplicate_stacking(self):
        api = self._api()
        for _ in range(2):
            api.telemetry.record({"kind": "tool", "name": "add_app", "ok": False, "approved": False})
        api.evolve.reflect()
        first = [p for p in api.evolve.list_proposals() if p["status"] == "pending"]
        api.evolve.reflect()  # same finding still open → must not stack
        second = [p for p in api.evolve.list_proposals() if p["status"] == "pending"]
        self.assertEqual(len(first), len(second))
        api.evolve.dismiss(first[0]["id"])
        self.assertEqual(
            next(p["status"] for p in api.evolve.list_proposals() if p["id"] == first[0]["id"]),
            "dismissed")

    def test_apply_twice_rejected(self):
        api = self._api()
        for _ in range(2):
            api.telemetry.record({"kind": "tool", "name": "add_app", "ok": False, "approved": False})
        api.evolve.reflect()
        pid = next(p["id"] for p in api.evolve.list_proposals() if p["status"] == "pending")
        api.evolve.apply(pid)
        with self.assertRaises(ValueError):
            api.evolve.apply(pid)

    def test_rollback_reverts_applied_addendum(self):
        api = self._api()
        for _ in range(2):
            api.telemetry.record({"kind": "tool", "name": "open_url", "ok": False, "approved": False})
        api.evolve.reflect()
        pid = next(p["id"] for p in api.evolve.list_proposals() if p["status"] == "pending")
        api.evolve.apply(pid)
        self.assertIn("open_url", api.agent_notes_read())          # guideline written
        rolled = api.evolve.rollback(pid)
        self.assertEqual(rolled["status"], "rolled-back")
        self.assertNotIn("open_url", api.agent_notes_read())        # snapshot reverted it

    def test_rollback_requires_applied_with_snapshot(self):
        api = self._api()
        for _ in range(2):
            api.telemetry.record({"kind": "tool", "name": "add_app", "ok": False, "approved": False})
        api.evolve.reflect()
        pid = next(p["id"] for p in api.evolve.list_proposals() if p["status"] == "pending")
        with self.assertRaises(ValueError):        # pending, not applied
            api.evolve.rollback(pid)

    def test_history_lists_completed_newest_first(self):
        api = self._api()
        api.memory_append("x"); api.memory_append("x")           # → auto-applied prune
        for _ in range(2):
            api.telemetry.record({"kind": "tool", "name": "add_app", "ok": False, "approved": False})
        api.evolve.reflect()
        pid = next(p["id"] for p in api.evolve.list_proposals() if p["status"] == "pending")
        api.evolve.dismiss(pid)
        history = api.evolve.history()
        self.assertTrue(history)
        self.assertTrue(all(p["status"] != "pending" for p in history))
        # newest-first: the dismissed addendum was completed after the auto prune
        self.assertEqual(history[0]["id"], pid)

    def test_model_reflection_disabled_in_local_mode(self):
        api = self._api()
        self.assertEqual(api.assistant.reflect_candidates({}), [])

    def test_model_reflection_parses_and_validates(self):
        from types import SimpleNamespace
        from unittest import mock
        api = self._api()
        payload = json.dumps([
            {"title": "Batch reads", "rationale": "seen sequential reads",
             "text": "Batch independent read tools into one turn."},
            {"title": "", "text": "dropped — no title"},          # invalid → skipped
            {"nonsense": True},                                    # invalid → skipped
        ])
        fake = SimpleNamespace(content=[SimpleNamespace(type="text", text=payload)])
        client = mock.Mock()
        client.messages.create.return_value = fake
        with mock.patch.object(type(api.assistant), "mode", new_callable=mock.PropertyMock,
                               return_value="claude"), \
             mock.patch.object(api.assistant, "_get_client", return_value=client):
            cands = api.assistant.reflect_candidates({"telemetry": {}})
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["kind"], "prompt_addendum")     # model can only propose this
        self.assertEqual(cands[0]["title"], "Batch reads")
        self.assertEqual(cands[0]["payload"]["source"], "model")

    def test_model_reflection_merges_into_inbox_without_auto_apply(self):
        from unittest import mock
        api = self._api()
        model_cand = [{"kind": "prompt_addendum", "title": "Prefer concise replies",
                       "rationale": "long answers", "payload": {"text": "Be concise.", "source": "model"}}]
        with mock.patch.object(api.assistant, "reflect_candidates", return_value=model_cand):
            api.evolve.reflect()
        props = api.evolve.list_proposals()
        model = [p for p in props if p["title"] == "Prefer concise replies"]
        self.assertEqual(len(model), 1)
        self.assertEqual(model[0]["status"], "pending")   # advisory — never auto-applies
        self.assertEqual(model[0]["kind"], "prompt_addendum")
        self.assertEqual(model[0]["source"], "model")     # provenance surfaced to the UI

    def test_heuristic_proposals_tagged_source(self):
        api = self._api()
        api.memory_append("Likes espresso")
        api.memory_append("Likes espresso")            # duplicate → prune finding
        api.evolve.reflect()
        prune = [p for p in api.evolve.list_proposals() if p["kind"] == "memory_prune"]
        self.assertTrue(prune)
        self.assertEqual(prune[0]["source"], "heuristic")   # default provenance

    def test_model_reflection_failure_is_swallowed(self):
        from unittest import mock
        api = self._api()
        with mock.patch.object(api.assistant, "reflect_candidates", side_effect=RuntimeError("boom")):
            created = api.evolve.reflect()  # must not raise
        self.assertIsInstance(created, list)

    def test_reflect_automation_action(self):
        api = self._api()
        api.memory_append("x"); api.memory_append("x")
        rule = api.automations.create_rule({
            "name": "nightly reflect", "trigger": {"type": "daily", "time": "00:00"},
            "action": {"type": "reflect"}})
        api.automations._fire(rule)
        notifs = api.automations.notifications_after(0)["notifications"]
        self.assertTrue(any("Reflection" in n["body"] or "proposal" in n["body"] for n in notifs))


class NeedsEscalationTests(unittest.TestCase):
    def test_detects_low_confidence(self):
        for t in ["I'm not sure about that", "It's unclear to me",
                  "I don't have enough information", "[ESCALATE] please help",
                  "I'm unsure how to proceed"]:
            self.assertTrue(assistant.needs_escalation(t), t)

    def test_ignores_confident_text(self):
        for t in ["Sure, here it is.", "Done.", "Version 2.1 fixed it.", ""]:
            self.assertFalse(assistant.needs_escalation(t), t)


class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_FakeBlock(text)]
        self.stop_reason = stop_reason


class _FakeStream:
    def __init__(self, response):
        self._response = response
        self.text_stream = [b.text for b in response.content]
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def get_final_message(self): return self._response


class _FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
    def create(self, **kw):
        self.calls.append(kw)
        return self.script.pop(0)
    def stream(self, **kw):
        self.calls.append(kw)
        return _FakeStream(self.script.pop(0))


class _FakeClient:
    def __init__(self, script):
        self.messages = _FakeMessages(script)


class EscalationTests(unittest.TestCase):
    def _assistant(self, script):
        from unittest import mock
        data_dir = Path(tempfile.mkdtemp())
        api = server.Api(offline=True, data_dir=data_dir)
        a = api.assistant
        a._client = _FakeClient(script)
        self._mode = mock.patch.object(type(a), "mode", new_callable=mock.PropertyMock,
                                       return_value="claude")
        self._mode.start()
        self.addCleanup(self._mode.stop)
        return a, api

    def test_low_confidence_triggers_advisor_then_confident_answer(self):
        a, api = self._assistant([
            _FakeResponse("Honestly I'm not sure about that."),  # 1st chat: uncertain
            _FakeResponse("Advisor: check the changelog and compare versions."),  # advise()
            _FakeResponse("Here is the confident, complete answer."),  # 2nd chat
        ])
        result = a._chat_claude([{"role": "user", "content": "which version fixed it?"}], {})
        self.assertTrue(result["escalated"])
        self.assertEqual(result["content"][0]["text"], "Here is the confident, complete answer.")
        self.assertEqual(len(a._client.messages.calls), 3)
        # the advisor call carries no tools and the advisor system prompt
        advisor_call = a._client.messages.calls[1]
        self.assertNotIn("tools", advisor_call)
        self.assertIn("senior advisor", advisor_call["system"])
        # telemetry recorded the escalation
        self.assertEqual(api.telemetry.summary()["escalations"], 1)

    def test_confident_reply_does_not_escalate(self):
        a, _ = self._assistant([_FakeResponse("Sure — version 2.1 fixed it.")])
        result = a._chat_claude([{"role": "user", "content": "which version?"}], {})
        self.assertFalse(result["escalated"])
        self.assertEqual(len(a._client.messages.calls), 1)

    def test_tool_use_turn_is_not_escalated(self):
        # even if the text looks uncertain, a tool_use turn is a real action
        a, _ = self._assistant([_FakeResponse("I'm not sure", stop_reason="tool_use")])
        result = a._chat_claude([{"role": "user", "content": "add a task"}], {})
        self.assertFalse(result["escalated"])
        self.assertEqual(len(a._client.messages.calls), 1)

    def test_advise_returns_none_when_deep_budget_exhausted(self):
        a, _ = self._assistant([_FakeResponse("unused")])
        a.router = router_mod.Router(max_deep_per_hour=0)
        self.assertIsNone(a.advise("stuck on something"))
        self.assertEqual(len(a._client.messages.calls), 0)  # never called the model

    def test_budget_exhaustion_skips_escalation_in_chat(self):
        a, _ = self._assistant([_FakeResponse("I'm not sure at all.")])
        a.router = router_mod.Router(max_deep_per_hour=0)
        result = a._chat_claude([{"role": "user", "content": "?"}], {})
        self.assertFalse(result["escalated"])
        self.assertEqual(len(a._client.messages.calls), 1)  # first reply only


class KillSwitchTests(unittest.TestCase):
    def _api(self):
        return server.Api(offline=True, data_dir=Path(tempfile.mkdtemp()))

    def test_freeze_blocks_firing_and_resume_restores(self):
        api = self._api()
        api.automations.create_rule({
            "name": "d", "trigger": {"type": "daily", "time": "00:00"},
            "action": {"type": "notify", "message": "hi"}})
        api.automations._data["rules"][0]["state"] = {}
        self.assertEqual(api.automations.tick(), 1)          # active → fires

        api.automations._data["rules"][0]["state"] = {}
        api.automations.set_frozen(True)
        self.assertTrue(api.automations.is_frozen())
        self.assertEqual(api.automations.tick(), 0)          # frozen → nothing

        api.automations.set_frozen(False)
        api.automations._data["rules"][0]["state"] = {}
        self.assertEqual(api.automations.tick(), 1)          # resumed → fires

    def test_frozen_state_persists(self):
        data_dir = Path(tempfile.mkdtemp())
        server.Api(offline=True, data_dir=data_dir).automations.set_frozen(True)
        self.assertTrue(server.Api(offline=True, data_dir=data_dir).automations.is_frozen())


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "telemetry.jsonl"

    def test_record_recent_summary(self):
        t = telemetry_mod.Telemetry(self.path)
        t.record({"kind": "route", "task": "chat", "tier": "core", "model": "m"})
        t.record({"kind": "tool", "name": "add_task", "tier": "auto", "ok": True, "approved": None})
        t.record({"kind": "tool", "name": "add_app", "tier": "confirm", "ok": False, "approved": False})
        self.assertEqual(len(t.recent()), 3)
        s = t.summary()
        self.assertEqual(s["tool_calls"], 2)
        self.assertEqual(s["denied"], 1)
        self.assertEqual(s["by_tier"], {"core": 1})
        self.assertEqual(s["by_tool"], {"add_task": 1, "add_app": 1})

    def test_bounded_to_max(self):
        t = telemetry_mod.Telemetry(self.path)
        for i in range(telemetry_mod.Telemetry.MAX + 20):
            t.record({"kind": "tool", "name": f"t{i}", "ok": True})
        self.assertEqual(len(t.recent(10_000)), telemetry_mod.Telemetry.MAX)

    def test_persists_across_reload(self):
        telemetry_mod.Telemetry(self.path).record({"kind": "tool", "name": "x", "ok": True})
        self.assertEqual(len(telemetry_mod.Telemetry(self.path).recent()), 1)

    def test_assistant_logs_route(self):
        data_dir = Path(tempfile.mkdtemp())
        api = server.Api(offline=True, data_dir=data_dir)
        api.assistant._log_route({"task": "chat", "tier": "deep", "model": "opus"})
        events = api.telemetry.recent()
        self.assertEqual(events[-1]["kind"], "route")
        self.assertEqual(events[-1]["tier"], "deep")


class PermissionTierTests(unittest.TestCase):
    def test_tiers_cover_every_shipped_tool(self):
        client_tools = {t["name"] for t in assistant.DASHBOARD_TOOLS}
        server_tools = set(assistant.SERVER_TOOLS)
        for name in client_tools | server_tools:
            self.assertIn(assistant.tool_tier(name), ("auto", "confirm"),
                          f"{name} is unclassified (would be blocked)")

    def test_sensitive_actions_require_confirm(self):
        for name in ("add_app", "open_url", "create_automation", "delete_automation"):
            self.assertEqual(assistant.tool_tier(name), "confirm", name)

    def test_frequent_actions_are_auto(self):
        for name in ("add_task", "complete_task", "add_event", "add_note",
                     "switch_news_topic", "get_news"):
            self.assertEqual(assistant.tool_tier(name), "auto", name)

    def test_unknown_tool_is_blocked(self):
        self.assertEqual(assistant.tool_tier("run_shell"), "blocked")
        self.assertEqual(assistant.tool_tier("send_money"), "blocked")

    def test_status_exposes_permission_map(self):
        perms = assistant.Assistant().status()["permissions"]
        self.assertEqual(perms["add_task"], "auto")
        self.assertEqual(perms["open_url"], "confirm")


class BackupHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server.CACHE.clear()
        data_dir = Path(tempfile.mkdtemp())
        cls.httpd = server.make_server(
            "127.0.0.1", 0, offline=True, data_dir=data_dir)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def request(self, path, payload=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as err:
            return err.code, json.loads(err.read())

    def test_backup_create_list_restore_over_http(self):
        status, _ = self.request("/api/state", {"state": {"version": 1, "layout": []}, "baseRev": None})
        self.assertEqual(status, 200)
        status, info = self.request("/api/backup", {})
        self.assertEqual(status, 200)
        status, listed = self.request("/api/backups")
        self.assertEqual(status, 200)
        self.assertEqual(listed["backups"][0]["name"], info["name"])
        status, result = self.request("/api/backup/restore", {"name": info["name"]})
        self.assertEqual(status, 200)
        self.assertIn("state", result["restored"])

    def test_restore_bad_name_400(self):
        status, _ = self.request("/api/backup/restore", {"name": "nope.txt"})
        self.assertEqual(status, 400)

    def test_telemetry_post_then_get(self):
        status, _ = self.request("/api/assistant/telemetry",
                                 {"name": "add_app", "tier": "confirm", "ok": False, "approved": False})
        self.assertEqual(status, 200)
        status, data = self.request("/api/assistant/telemetry")
        self.assertEqual(status, 200)
        self.assertTrue(any(e["name"] == "add_app" and e["approved"] is False
                            for e in data["events"] if e["kind"] == "tool"))
        self.assertGreaterEqual(data["summary"]["denied"], 1)

    def test_telemetry_post_requires_name(self):
        status, _ = self.request("/api/assistant/telemetry", {"tier": "auto", "ok": True})
        self.assertEqual(status, 400)

    def test_killswitch_get_set_over_http(self):
        status, data = self.request("/api/killswitch")
        self.assertEqual(status, 200)
        self.assertFalse(data["frozen"])
        status, data = self.request("/api/killswitch", {"frozen": True})
        self.assertEqual(status, 200)
        self.assertTrue(data["frozen"])
        status, data = self.request("/api/killswitch")
        self.assertTrue(data["frozen"])
        # cleanup so other tests on this shared server see a resumed state
        self.request("/api/killswitch", {"frozen": False})

    def test_killswitch_rejects_bad_body(self):
        status, _ = self.request("/api/killswitch", {"frozen": "yes"})
        self.assertEqual(status, 400)

    def test_evolve_reflect_and_apply_over_http(self):
        # seed a denied tool so reflection has a pending proposal to act on
        self.request("/api/assistant/telemetry", {"name": "add_app", "ok": False, "approved": False})
        self.request("/api/assistant/telemetry", {"name": "add_app", "ok": False, "approved": False})
        status, data = self.request("/api/evolve/reflect", {})
        self.assertEqual(status, 200)
        status, data = self.request("/api/evolve")
        self.assertEqual(status, 200)
        pending = [p for p in data["proposals"] if p["status"] == "pending"]
        self.assertTrue(pending)
        status, data = self.request("/api/evolve/proposal", {"op": "apply", "id": pending[0]["id"]})
        self.assertEqual(status, 200)
        self.assertIn(data["proposal"]["status"], ("applied", "auto-applied"))

    def test_evolve_proposal_bad_id_404(self):
        status, _ = self.request("/api/evolve/proposal", {"op": "apply", "id": 99999})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
