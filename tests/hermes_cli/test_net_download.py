"""Tests for hermes_cli.net_download — proxy detection + mirror fallback."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli.net_download import (
    _macos_system_proxy,
    curl_download,
    detect_proxy,
    explicit_proxy,
    fetch_with_fallback,
    mirror_candidates,
    proxy_env_for,
)


class TestExplicitProxy:
    def test_https_proxy_wins(self):
        env = {"HTTPS_PROXY": "http://proxy:8080", "HTTP_PROXY": "http://proxy:8081"}
        assert explicit_proxy(env) == "http://proxy:8080"

    def test_http_proxy_fallback(self):
        assert explicit_proxy({"HTTP_PROXY": "http://proxy:8081"}) == "http://proxy:8081"

    def test_case_insensitive(self):
        assert explicit_proxy({"https_proxy": "http://proxy:8080"}) == "http://proxy:8080"

    def test_all_proxy_last_resort(self):
        assert explicit_proxy({"ALL_PROXY": "socks5://127.0.0.1:1080"}) == "socks5://127.0.0.1:1080"

    def test_empty(self):
        assert explicit_proxy({}) is None


class TestDetectProxy:
    def test_explicit_beats_system(self, monkeypatch):
        env = {"HTTPS_PROXY": "http://user-proxy:3128"}
        monkeypatch.setattr("hermes_cli.net_download._macos_system_proxy", lambda: "http://127.0.0.1:6152")
        assert detect_proxy(env) == "http://user-proxy:3128"

    def test_system_proxy_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.net_download._macos_system_proxy", lambda: "http://127.0.0.1:6152")
        assert detect_proxy({}) == "http://127.0.0.1:6152"

    def test_no_proxy(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.net_download._macos_system_proxy", lambda: None)
        assert detect_proxy({}) is None


class TestMacOSSystemProxy:
    def _fake_run(self, output, returncode=0):
        def fake_run(args, **kwargs):
            return type("R", (), {"returncode": returncode, "stdout": output})()
        return fake_run

    def test_parses_https_proxy_with_port(self, monkeypatch):
        out = (
            "HTTPEnable : 1\n"
            "HTTPProxy : 127.0.0.1\n"
            "HTTPPort : 6152\n"
            "HTTPSEnable : 1\n"
            "HTTPSProxy : 127.0.0.1\n"
            "HTTPSPort : 6152\n"
        )
        monkeypatch.setattr("hermes_cli.net_download.subprocess.run", self._fake_run(out))
        assert _macos_system_proxy() == "http://127.0.0.1:6152"

    def test_port_before_host_still_works(self, monkeypatch):
        # Port keys may appear before their host keys in scutil output.
        out = (
            "HTTPSEnable : 1\n"
            "HTTPSPort : 6152\n"
            "HTTPSProxy : 10.0.0.1\n"
        )
        monkeypatch.setattr("hermes_cli.net_download.subprocess.run", self._fake_run(out))
        assert _macos_system_proxy() == "http://10.0.0.1:6152"

    def test_disabled_proxy_returns_none(self, monkeypatch):
        out = "HTTPSEnable : 0\nHTTPSProxy : 127.0.0.1\nHTTPSPort : 6152\n"
        monkeypatch.setattr("hermes_cli.net_download.subprocess.run", self._fake_run(out))
        assert _macos_system_proxy() is None

    def test_non_darwin_returns_none(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.net_download.platform.system", lambda: "Linux")
        assert _macos_system_proxy() is None


class TestProxyEnvFor:
    def test_injects_proxy_without_mutating(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.net_download._macos_system_proxy", lambda: "http://127.0.0.1:6152")
        base = {"FOO": "bar"}
        out = proxy_env_for(base)
        assert base == {"FOO": "bar"}  # not mutated
        assert out["HTTPS_PROXY"] == "http://127.0.0.1:6152"
        assert out["HTTP_PROXY"] == "http://127.0.0.1:6152"
        assert out["FOO"] == "bar"

    def test_no_proxy_returns_unchanged(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.net_download._macos_system_proxy", lambda: None)
        assert proxy_env_for({"FOO": "bar"}) == {"FOO": "bar"}


class TestMirrorCandidates:
    RAW = "https://raw.githubusercontent.com/trycua/cua/main/libs/cua-driver/scripts/install.sh"

    def test_github_urls_get_mirrors(self):
        urls = mirror_candidates(self.RAW)
        assert urls[0].startswith("https://ghfast.top/")
        assert urls[1].startswith("https://gh-proxy.com/")
        assert self.RAW in urls[0]

    def test_non_github_urls_empty(self):
        assert mirror_candidates("https://example.com/x.sh") == []
        assert mirror_candidates("https://pypi.org/simple/") == []


class TestCurlDownload:
    def _fake_curl_script(self, tmp_path, fail_urls=(), output="fake-content"):
        """Create a fake `curl` executable.

        Fails (exit 1) for URLs in ``fail_urls``, otherwise writes
        ``output`` to the ``-o`` destination.
        """
        script = tmp_path / "fake-curl"
        script.write_text(
            "#!/bin/sh\n"
            "url=''\n"
            "dest=''\n"
            "prev=''\n"
            "for a in \"$@\"; do\n"
            "  if [ \"$prev\" = \"-o\" ]; then\n"
            "    dest=\"$a\"\n"
            "    prev=''\n"
            "  fi\n"
            "  case \"$a\" in\n"
            "    -o) prev='-o' ;;\n"
            "    http*) url=\"$a\" ;;\n"
            "  esac\n"
            "done\n"
            "case \"$url\" in\n"
            + "".join(f"    {u!r}) exit 1 ;;\n" for u in fail_urls)
            + "esac\n"
            "printf '%s' '" + output + "' > \"$dest\"\n"
            "exit 0\n"
        )
        script.chmod(0o755)
        return str(script)

    def test_success(self, tmp_path):
        dest = tmp_path / "out.sh"
        ok, detail = curl_download(
            "https://raw.githubusercontent.com/x/y/main/z.sh",
            str(dest),
            curl_cmd=self._fake_curl_script(tmp_path),
        )
        assert ok is True
        assert detail == ""
        assert dest.read_text() == "fake-content"

    def test_failure_returns_detail(self, tmp_path):
        dest = tmp_path / "out.sh"
        ok, detail = curl_download(
            "https://raw.githubusercontent.com/x/y/main/z.sh",
            str(dest),
            curl_cmd=self._fake_curl_script(tmp_path, fail_urls=("https://raw.githubusercontent.com/x/y/main/z.sh",)),
        )
        assert ok is False
        assert "exit 1" in detail


class TestFetchWithFallback:
    def test_official_success_no_mirror_attempted(self, tmp_path):
        dest = tmp_path / "out.sh"
        ok, _ = fetch_with_fallback(
            "https://raw.githubusercontent.com/x/y/main/z.sh",
            str(dest),
            curl_cmd=TestCurlDownload()._fake_curl_script(tmp_path),
        )
        assert ok is True

    def test_official_fails_mirror_succeeds(self, tmp_path):
        dest = tmp_path / "out.sh"
        url = "https://raw.githubusercontent.com/x/y/main/z.sh"
        curl = TestCurlDownload()._fake_curl_script(
            tmp_path, fail_urls=(url,), output="mirror-content"
        )
        ok, detail = fetch_with_fallback(url, str(dest), curl_cmd=curl)
        assert ok is True
        assert dest.read_text() == "mirror-content"

    def test_all_fail_returns_summary(self, tmp_path):
        dest = tmp_path / "out.sh"
        url = "https://raw.githubusercontent.com/x/y/main/z.sh"
        mirror1 = f"https://ghfast.top/{url}"
        mirror2 = f"https://gh-proxy.com/{url}"
        curl = TestCurlDownload()._fake_curl_script(
            tmp_path, fail_urls=(url, mirror1, mirror2), output="x"
        )
        ok, detail = fetch_with_fallback(url, str(dest), curl_cmd=curl)
        assert ok is False
        assert url in detail
        assert mirror1 in detail
