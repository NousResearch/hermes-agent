#!/usr/bin/env python3
"""TP0e + core contracts: unit tests for talaria.core.

Pure-logic, network-free: urlopen is mocked. Covers the service registry,
env-driven config resolution (P1/P2/P3), header shapes (incl. MediaBrowser),
the typed error taxonomy + redaction rule, ticks math, retries, formatting.
"""

import io
import json
import os
import socket
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

PY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PY))

from talaria import core  # noqa: E402


class Env:
    """Context manager: set/patch env vars, restore on exit."""

    def __init__(self, **kw):
        self.kw = kw
        self._saved = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                try:
                    del os.environ[k]
                except KeyError:
                    pass
            else:
                os.environ.update({k: v})
        return self

    def __exit__(self, *a):
        for k, v in self._saved.items():
            if v is None:
                try:
                    del os.environ[k]
                except KeyError:
                    pass
            else:
                os.environ.update({k: v})


class Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body=b"", msg="err"):
    return urllib.error.HTTPError("http://127.0.0.1/x", code, msg, {}, io.BytesIO(body))


FAKE_KEY = "zzfake-secret-key-do-not-printzz"


class TestRegistry(unittest.TestCase):
    def test_seven_services_with_gate_answers(self):
        self.assertEqual(
            sorted(core.SERVICES),
            ["bazarr", "jellyfin", "lidarr", "overseerr", "prowlarr",
             "radarr", "sonarr"])
        jf = core.SERVICES["jellyfin"]
        # JELLYFIN_KEY; uniform port overrides; API base at root
        self.assertEqual((jf.key_env, jf.port_env, jf.default_port, jf.api_base,
                          jf.auth),
                         ("JELLYFIN_KEY", "JELLYFIN_PORT", 8096, "", "mediabrowser"))
        sn = core.SERVICES["sonarr"]
        self.assertEqual((sn.key_env, sn.port_env, sn.default_port, sn.api_base,
                          sn.auth),
                         ("SONARR_KEY", "SONARR_PORT", 8989, "/api/v3",
                          "x-api-key"))
        bz = core.SERVICES["bazarr"]
        self.assertEqual((bz.default_port, bz.api_base), (6767, "/api"))
        self.assertEqual(core.SERVICES["lidarr"].api_base, "/api/v1")
        self.assertEqual(core.SERVICES["prowlarr"].default_port, 9696)
        self.assertEqual(core.SERVICES["overseerr"].api_base, "/api/v1")


class TestHostResolution(unittest.TestCase):
    # TALARIA_HOST is the single host var
    def test_talaria_host_used(self):
        with Env(TALARIA_HOST="10.0.0.1"):
            self.assertEqual(core.host(), "10.0.0.1")

    def test_no_host_is_config_error_with_hint(self):
        with Env(TALARIA_HOST=None):
            with self.assertRaises(core.ConfigError) as ctx:
                core.host()
            self.assertIn("setup", str(ctx.exception))

    def test_override_beats_env(self):
        with Env(TALARIA_HOST="10.0.0.1"):
            self.assertEqual(core.host("127.0.0.1"), "127.0.0.1")


class TestBaseAndHeaders(unittest.TestCase):
    def test_base_url_default_and_port_override(self):
        with Env(TALARIA_HOST="h1", SONARR_PORT=None, JELLYFIN_PORT="18096"):
            self.assertEqual(core.base_url("sonarr"), "http://h1:8989")
            self.assertEqual(core.base_url("jellyfin"), "http://h1:18096")
        with Env(TALARIA_HOST="h1", SONARR_PORT="18989"):
            self.assertEqual(core.base_url("sonarr"), "http://h1:18989")

    def test_headers_api_key_shape(self):
        with Env(RADARR_KEY=FAKE_KEY):
            self.assertEqual(core.headers("radarr"), {"X-Api-Key": FAKE_KEY})

    def test_headers_mediabrowser_shape(self):
        # minimal bare Token form, scheme MediaBrowser
        with Env(JELLYFIN_KEY=FAKE_KEY):
            self.assertEqual(core.headers("jellyfin"),
                             {"Authorization": 'MediaBrowser Token="%s"' % FAKE_KEY})

    def test_missing_key_names_the_var(self):
        with Env(SONARR_KEY=None):
            with self.assertRaises(core.ConfigError) as ctx:
                core.headers("sonarr")
            self.assertIn("SONARR_KEY", str(ctx.exception))
            self.assertIn("setup", str(ctx.exception))


class TestApi(unittest.TestCase):
    def setUp(self):
        self.env = Env(TALARIA_HOST="127.0.0.1", SONARR_KEY="k-sonarr",
                       RADARR_KEY="k-radarr", PROWLARR_KEY="k-prowlarr",
                       JELLYFIN_KEY="k-jf").__enter__()
        self.addCleanup(self.env.__exit__, None, None, None)

    def test_get_builds_url_params_and_headers(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.get_method()
            captured["headers"] = dict(req.header_items())
            captured["timeout"] = timeout
            return Resp(json.dumps([1, 2]).encode())

        with mock.patch("talaria.core.urlopen", side_effect=fake_urlopen):
            out = core.api("sonarr", "GET", "/queue",
                           params={"pageSize": 50, "sortKey": "date"})
        self.assertEqual(out, [1, 2])
        self.assertEqual(captured["method"], "GET")
        self.assertIn("pageSize=50", captured["url"])
        self.assertIn("sortKey=date", captured["url"])
        self.assertEqual(captured["timeout"], 10)
        # key rides the header, NEVER the URL
        self.assertNotIn("k-sonarr", captured["url"])
        hdrs = {k.lower(): v for k, v in captured["headers"].items()}
        self.assertEqual(hdrs.get("x-api-key"), "k-sonarr")

    def test_list_params_encode_repeated(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            return Resp(b"{}")

        with mock.patch("talaria.core.urlopen", side_effect=fake_urlopen):
            core.api("prowlarr", "GET", "/search",
                     params={"query": "a b", "indexerIds": [1, 2]})
        self.assertIn("query=a+b", captured["url"])
        self.assertIn("indexerIds=1", captured["url"])
        self.assertIn("indexerIds=2", captured["url"])

    def test_post_json_body(self):
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["method"] = req.get_method()
            captured["body"] = req.data.decode()
            captured["headers"] = dict(req.header_items())
            return Resp(b'{"id": 7}')

        with mock.patch("talaria.core.urlopen", side_effect=fake_urlopen):
            out = core.api("sonarr", "POST", "/command",
                           body={"name": "RefreshSeries", "seriesId": 3})
        self.assertEqual(out, {"id": 7})
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(json.loads(captured["body"]),
                         {"name": "RefreshSeries", "seriesId": 3})
        hdrs = {k.lower(): v for k, v in captured["headers"].items()}
        self.assertIn("application/json", hdrs.get("content-type", ""))

    def test_401_is_auth_error_and_redacts_key(self):
        with mock.patch("talaria.core.urlopen",
                        side_effect=http_error(401, b'{"message":"bad"}')):
            with self.assertRaises(core.AuthError) as ctx:
                core.api("sonarr", "GET", "/queue")
        msg = str(ctx.exception)
        self.assertEqual(msg, "sonarr rejected the API key")
        self.assertNotIn("k-sonarr", msg)

    def test_403_is_auth_error(self):
        with mock.patch("talaria.core.urlopen",
                        side_effect=http_error(403, b"")):
            with self.assertRaises(core.AuthError):
                core.api("radarr", "GET", "/queue")

    def test_other_status_is_api_error_with_server_message(self):
        body = json.dumps({"message": "series not found"}).encode()
        with mock.patch("talaria.core.urlopen",
                        side_effect=http_error(404, body)):
            with self.assertRaises(core.ApiError) as ctx:
                core.api("sonarr", "GET", "/series/999")
        self.assertIn("404", str(ctx.exception))
        self.assertIn("series not found", str(ctx.exception))

    def test_api_error_redacts_key_even_if_server_echoes_it(self):
        body = json.dumps({"message": "key k-sonarr invalid"}).encode()
        with mock.patch("talaria.core.urlopen",
                        side_effect=http_error(400, body)):
            with self.assertRaises(core.ApiError) as ctx:
                core.api("sonarr", "GET", "/series")
        # redaction rule: no key value in the surfaced message
        self.assertNotIn("k-sonarr", str(ctx.exception))

    def test_timeout_message_names_service_and_base_not_key(self):
        with mock.patch("talaria.core.urlopen",
                        side_effect=socket.timeout("timed out")):
            with self.assertRaises(core.TimeoutError) as ctx:
                core.api("sonarr", "GET", "/queue", timeout=4)
        msg = str(ctx.exception)
        self.assertIn("4", msg)
        self.assertIn("sonarr", msg)
        self.assertIn("http://127.0.0.1:8989", msg)
        self.assertNotIn("k-sonarr", msg)

    def test_refused_is_network_error(self):
        err = urllib.error.URLError(ConnectionRefusedError(111, "refused"))
        with mock.patch("talaria.core.urlopen", side_effect=err):
            with self.assertRaises(core.NetworkError) as ctx:
                core.api("sonarr", "GET", "/queue")
        self.assertIn("sonarr", str(ctx.exception))
        self.assertNotIn("k-sonarr", str(ctx.exception))

    def test_empty_body_returns_empty_dict(self):
        with mock.patch("talaria.core.urlopen", return_value=Resp(b"")):
            out = core.api("sonarr", "DELETE", "/series/1")
        self.assertEqual(out, {})


class TestApiGetRetries(unittest.TestCase):
    def setUp(self):
        self.env = Env(TALARIA_HOST="127.0.0.1", SONARR_KEY="k").__enter__()
        self.addCleanup(self.env.__exit__, None, None, None)

    def test_retries_on_urlerror_then_succeeds(self):
        calls = []

        def flaky(req, timeout=None):
            calls.append(1)
            if len(calls) < 3:
                raise urllib.error.URLError(OSError("boom"))
            return Resp(b"[1]")

        with mock.patch("talaria.core.urlopen", side_effect=flaky), \
                mock.patch("time.sleep") as slept:
            out = core.api_get("sonarr", "/queue")
        self.assertEqual(out, [1])
        self.assertEqual(len(calls), 3)
        self.assertEqual([call.args[0] for call in slept.call_args_list],
                         [0.5, 1.0])

    def test_never_retries_on_http_4xx(self):
        calls = []

        def once(req, timeout=None):
            calls.append(1)
            raise http_error(401, b"")

        with mock.patch("talaria.core.urlopen", side_effect=once):
            with self.assertRaises(core.AuthError):
                core.api_get("sonarr", "/queue")
        self.assertEqual(len(calls), 1)

    def test_gives_up_after_two_retries(self):
        with mock.patch("talaria.core.urlopen",
                        side_effect=urllib.error.URLError(OSError("down"))), \
                mock.patch("time.sleep"):
            with self.assertRaises(core.NetworkError):
                core.api_get("sonarr", "/queue")


class TestTicksAndFormat(unittest.TestCase):
    # .NET ticks — 1e7 ticks per second
    def test_ticks(self):
        self.assertEqual(core.ticks_mmss(0), "0:00")
        self.assertEqual(core.ticks_mmss(900_000_000), "1:30")
        self.assertEqual(core.ticks_mmss(18_400_000_000), "30:40")
        self.assertEqual(core.ticks_mmss(73_200_000_000), "2:02:00")
        self.assertEqual(core.ticks_hours(73_200_000_000), 2.0)
        self.assertEqual(core.ticks_hours(0), 0.0)

    def test_ticks_mmss_plain_seconds(self):
        # 90,000,000 ticks = 9 s -> "0:09"
        self.assertEqual(core.ticks_mmss(90_000_000), "0:09")

    def test_table_aligns_columns(self):
        out = core.table(["Name", "N"], [["Alien", "3"], ["Bo", "42"]])
        lines = out.splitlines()
        self.assertEqual(lines[0].split(), ["Name", "N"])
        self.assertTrue(lines[1].strip().startswith("----"))
        self.assertIn("Alien", lines[2])
        # columns aligned: same index of "3" and "42"
        self.assertEqual(lines[2].index("3"), lines[3].index("42"))

    def test_approx_banner_text(self):
        banner = core.approx_banner()
        self.assertIn("approximate", banner)
        self.assertIn("play-state", banner)
        self.assertNotIn("\n", banner)


class TestVersion(unittest.TestCase):
    def test_version_matches_release(self):
        # Version pinned to the skill release
        import talaria
        self.assertEqual(talaria.__version__, "1.0.0")


if __name__ == "__main__":
    unittest.main()
