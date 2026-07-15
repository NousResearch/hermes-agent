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
import router as router_mod  # noqa: E402
import server  # noqa: E402


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
                    "http://192.168.1.1/", "ftp://example.org/"):
            with self.assertRaises(server.ApiError, msg=url):
                self.api.reader({"url": [url]})

    def test_reader_offline_note(self):
        data = self.api.reader({"url": ["https://example.org/story"]})
        self.assertEqual(data["source"], "sample")
        self.assertIn("note", data)

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
        self.assertEqual(actions, [("add_task", {"text": "buy stamps", "list": "Errands"})])

    def test_add_task_defaults_to_today(self):
        actions = self._actions(self._chat("task: water plants"))
        self.assertEqual(actions[0][1]["list"], "Today")

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

        r = a._chat_local([{"role": "user", "content": "every morning at 7 brief me"}], {})
        tools = [b for b in r["content"] if b["type"] == "tool_use"]
        self.assertEqual(tools[0]["input"]["time"], "07:00")

        r = a._chat_local([{"role": "user", "content": "brief me daily at 8pm"}], {})
        tools = [b for b in r["content"] if b["type"] == "tool_use"]
        self.assertEqual(tools[0]["input"]["time"], "20:00")

        r = a._chat_local([{"role": "user", "content": "what's the weather like?"}], {})
        self.assertIn("New York", r["content"][0]["text"])


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
                         {"state", "feeds", "calendars", "automations", "memory"})
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
