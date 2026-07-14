"""Unit tests for the Hermes Hub dashboard server (stdlib only).

Run:  python3 -m unittest discover -s apps/dashboard/tests -v
"""

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import assistant  # noqa: E402
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
