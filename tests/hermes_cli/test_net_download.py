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
    resolve_dns_doh,
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
    @pytest.fixture(autouse=True)
    def _darwin(self, monkeypatch):
        """Simulate a macOS host for all proxy-parsing tests.

        `_macos_system_proxy` early-returns None when `platform.system()`
        is not "Darwin", so on Linux CI runners the parsing assertions must
        force Darwin or they fail before touching subprocess at all.
        (test_non_darwin_returns_none deliberately overrides this fixture.)
        """
        monkeypatch.setattr("hermes_cli.net_download.platform.system", lambda: "Darwin")

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

    def test_official_fails_mirror_succeeds_when_opted_in(self, tmp_path):
        dest = tmp_path / "out.sh"
        url = "https://raw.githubusercontent.com/x/y/main/z.sh"
        curl = TestCurlDownload()._fake_curl_script(
            tmp_path, fail_urls=(url,), output="mirror-content"
        )
        # Mirrors are opt-in: the mirror path only runs when the caller
        # explicitly passes allow_mirrors=True (and only for non-executed
        # data-class content — executed content can never use mirrors).
        ok, detail = fetch_with_fallback(
            url, str(dest), curl_cmd=curl,
            content_class="data", allow_mirrors=True,
        )
        assert ok is True
        assert dest.read_text() == "mirror-content"

    def test_official_fails_no_mirror_by_default(self, tmp_path):
        """Security contract: executed-content fetches must not fall back to
        third-party mirrors unless the caller opts in. The default keeps
        mirrors off, so an official-URL failure is a hard failure."""
        dest = tmp_path / "out.sh"
        url = "https://raw.githubusercontent.com/x/y/main/z.sh"
        mirror1 = f"https://ghfast.top/{url}"
        mirror2 = f"https://gh-proxy.com/{url}"
        curl = TestCurlDownload()._fake_curl_script(
            tmp_path, fail_urls=(url, mirror1, mirror2), output="mirror-content"
        )
        ok, detail = fetch_with_fallback(url, str(dest), curl_cmd=curl)
        assert ok is False
        assert url in detail
        # Mirrors must NOT be attempted (and therefore must not appear in the
        # failure summary) with the default allow_mirrors=False.
        assert mirror1 not in detail
        assert mirror2 not in detail
        assert not dest.exists() or dest.read_text() == ""

    def test_all_fail_returns_summary_when_opted_in(self, tmp_path):
        dest = tmp_path / "out.sh"
        url = "https://raw.githubusercontent.com/x/y/main/z.sh"
        mirror1 = f"https://ghfast.top/{url}"
        mirror2 = f"https://gh-proxy.com/{url}"
        curl = TestCurlDownload()._fake_curl_script(
            tmp_path, fail_urls=(url, mirror1, mirror2), output="x"
        )
        ok, detail = fetch_with_fallback(
            url, str(dest), curl_cmd=curl,
            content_class="data", allow_mirrors=True,
        )
        assert ok is False
        assert url in detail
        assert mirror1 in detail
        assert mirror2 in detail

    def test_executed_class_ignores_allow_mirrors_true(self, tmp_path):
        """Security contract: content_class='executed' (the default) must
        permanently disable mirrors at the API level — even a caller that
        mistakenly passes allow_mirrors=True must not be able to route
        executed-content bytes through a third-party mirror."""
        dest = tmp_path / "out.sh"
        url = "https://raw.githubusercontent.com/x/y/main/z.sh"
        mirror1 = f"https://ghfast.top/{url}"
        mirror2 = f"https://gh-proxy.com/{url}"
        curl = TestCurlDownload()._fake_curl_script(
            tmp_path, fail_urls=(url, mirror1, mirror2), output="mirror-content"
        )
        # Explicitly asking for mirrors on executed content is ignored:
        ok, detail = fetch_with_fallback(
            url, str(dest), curl_cmd=curl,
            content_class="executed", allow_mirrors=True,
        )
        assert ok is False
        assert url in detail
        assert mirror1 not in detail
        assert mirror2 not in detail
        assert not dest.exists() or dest.read_text() == ""

    def test_data_class_opt_in_mirror(self, tmp_path):
        """content_class='data' keeps mirrors opt-in: allow_mirrors=True
        works for non-executed payloads (model weights, metadata)."""
        dest = tmp_path / "out.bin"
        url = "https://raw.githubusercontent.com/x/y/main/weights.bin"
        curl = TestCurlDownload()._fake_curl_script(
            tmp_path, fail_urls=(url,), output="data-bytes"
        )
        ok, detail = fetch_with_fallback(
            url, str(dest), curl_cmd=curl,
            content_class="data", allow_mirrors=True,
        )
        assert ok is True
        assert dest.read_text() == "data-bytes"

    def test_data_class_default_no_mirror(self, tmp_path):
        """Even data content defaults to mirrors off (allow_mirrors=None)."""
        dest = tmp_path / "out.bin"
        url = "https://raw.githubusercontent.com/x/y/main/weights.bin"
        mirror1 = f"https://ghfast.top/{url}"
        curl = TestCurlDownload()._fake_curl_script(
            tmp_path, fail_urls=(url, mirror1), output="data-bytes"
        )
        ok, detail = fetch_with_fallback(
            url, str(dest), curl_cmd=curl, content_class="data",
        )
        assert ok is False
        assert url in detail
        assert mirror1 not in detail


class TestResolveDnsDoh:
    """DNS-over-HTTPS resolution (DNSPod doh.pub)."""

    def _fake_run(self, output, returncode=0):
        def fake_run(args, **kwargs):
            return type("R", (), {"returncode": returncode, "stdout": output})()
        return fake_run

    def test_parses_a_records(self, monkeypatch):
        out = (
            '{"Status":0,"Answer":['
            '{"name":"example.com.","type":1,"TTL":60,"data":"93.184.216.34"},'
            '{"name":"example.com.","type":1,"TTL":60,"data":"93.184.216.35"}]}'
        )
        monkeypatch.setattr("hermes_cli.net_download.subprocess.run", self._fake_run(out))
        assert resolve_dns_doh("example.com", curl_cmd="curl") == [
            "93.184.216.34", "93.184.216.35",
        ]

    def test_filters_non_a_records_and_malformed(self, monkeypatch):
        out = (
            '{"Answer":['
            '{"type":1,"data":"93.184.216.34"},'
            '{"type":28,"data":"2606:2800:220:1:248:1893:25c8:1946"},'
            '{"type":5,"data":"evil.example.net"},'
            '{"type":1,"data":"not-an-ip"},'
            '{"type":1,"data":"999.999.999.999"}]}'
        )
        monkeypatch.setattr("hermes_cli.net_download.subprocess.run", self._fake_run(out))
        assert resolve_dns_doh("example.com", curl_cmd="curl") == ["93.184.216.34"]

    def test_returns_empty_on_curl_failure(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.net_download.subprocess.run", self._fake_run("", returncode=7)
        )
        assert resolve_dns_doh("example.com", curl_cmd="curl") == []

    def test_returns_empty_on_timeout(self, monkeypatch):
        def boom(args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args, timeout=5)
        monkeypatch.setattr("hermes_cli.net_download.subprocess.run", boom)
        assert resolve_dns_doh("example.com", curl_cmd="curl") == []

    def test_returns_empty_on_invalid_json(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.net_download.subprocess.run", self._fake_run("not json")
        )
        assert resolve_dns_doh("example.com", curl_cmd="curl") == []

    def test_missing_curl_returns_empty(self, monkeypatch):
        monkeypatch.setattr("hermes_cli.net_download.shutil.which", lambda name: None)
        assert resolve_dns_doh("example.com") == []

    def test_doh_query_does_not_use_proxy(self, monkeypatch):
        """The DoH query must not be routed through a proxy — the whole
        point is to bypass poisoned resolution, and doh.pub is reachable
        directly from CN. The child env must not gain a proxy either."""
        captured = {}

        def fake_run(args, **kwargs):
            captured["args"] = args
            captured["env"] = kwargs.get("env")
            return type("R", (), {"returncode": 0, "stdout": '{"Answer":[]}'})()

        monkeypatch.setattr("hermes_cli.net_download.subprocess.run", fake_run)
        resolve_dns_doh(
            "example.com", curl_cmd="curl",
            env={"HTTPS_PROXY": "http://proxy:8080", "FOO": "bar"},
        )
        assert "-x" not in captured["args"]
        assert any("doh.pub" in a for a in captured["args"])
        assert any("application/dns-json" in a for a in captured["args"])
        # env 未被修改（无代理注入），调用方 dict 原样透传
        assert captured["env"] == {"HTTPS_PROXY": "http://proxy:8080", "FOO": "bar"}


class TestCurlDownloadDnsFallback:
    """DNS-pollution fallback inside curl_download."""

    def _fake_curl_dns(self, tmp_path, doh_ips=("93.184.216.34",), output="dns-content"):
        """fake curl that models a poisoned-DNS network:

        - doh.pub URL          -> prints the DNS JSON with ``doh_ips``
        - URL with --resolve   -> writes ``output`` to dest (fallback wins)
        - anything else        -> exit 1 with a DNS error on stderr
        """
        script = tmp_path / "fake-curl-dns"
        answers = ",".join(
            f'{{"name":"h","type":1,"TTL":60,"data":"{ip}"}}' for ip in doh_ips
        )
        script.write_text(
            "#!/bin/sh\n"
            "url=''; dest=''; is_resolve=''; prev=''\n"
            "for a in \"$@\"; do\n"
            "  if [ \"$prev\" = \"-o\" ]; then dest=\"$a\"; prev=''; fi\n"
            "  case \"$a\" in\n"
            "    --resolve) is_resolve='1' ;;\n"
            "    -o) prev='-o' ;;\n"
            "    http*) url=\"$a\" ;;\n"
            "  esac\n"
            "done\n"
            "case \"$url\" in\n"
            "  *doh.pub*)\n"
            f"    printf '%s' '{{\"Status\":0,\"Answer\":[{answers}]}}'\n"
            "    exit 0 ;;\n"
            "esac\n"
            "if [ -n \"$is_resolve\" ]; then\n"
            f"  printf '%s' '{output}' > \"$dest\"\n"
            "  exit 0\n"
            "fi\n"
            "echo 'Could not resolve host: example.com' >&2\n"
            "exit 1\n"
        )
        script.chmod(0o755)
        return str(script)

    def test_dns_fallback_retries_with_resolve(self, tmp_path):
        """Core path: direct fetch fails (poisoned DNS), DoH returns the
        real IP, and the retry with --resolve lands the file."""
        dest = tmp_path / "out.sh"
        url = "https://huggingface.co/api/models"  # known-polluted host
        ok, detail = curl_download(url, str(dest), curl_cmd=self._fake_curl_dns(tmp_path))
        assert ok is True
        assert dest.read_text() == "dns-content"

    def test_first_ip_fails_second_succeeds(self, tmp_path):
        """The fallback tries each DoH IP in order until one connects."""
        dest = tmp_path / "out.sh"
        url = "https://huggingface.co/api/models"
        # 两个 IP：fake curl 对第一个 --resolve 失败、第二个成功
        script = tmp_path / "fake-curl-multi"
        script.write_text(
            "#!/bin/sh\n"
            "url=''; dest=''; resolve_ip=''; prev=''\n"
            "for a in \"$@\"; do\n"
            "  if [ \"$prev\" = \"-o\" ]; then dest=\"$a\"; prev=''; fi\n"
            "  case \"$a\" in\n"
            "    --resolve) resolve_ip='pending' ;;\n"
            "    -o) prev='-o' ;;\n"
            "    http*) url=\"$a\" ;;\n"
            "  esac\n"
            "  case \"$a\" in\n"
            "    *:93.184.216.34) resolve_ip='10.0.0.1' ;;\n"
            "    *:93.184.216.35) resolve_ip='10.0.0.2' ;;\n"
            "  esac\n"
            "done\n"
            "case \"$url\" in\n"
            "  *doh.pub*)\n"
            "    printf '%s' '{\"Status\":0,\"Answer\":[{\"name\":\"h\",\"type\":1,\"TTL\":60,\"data\":\"93.184.216.34\"},{\"name\":\"h\",\"type\":1,\"TTL\":60,\"data\":\"93.184.216.35\"}]}'\n"
            "    exit 0 ;;\n"
            "esac\n"
            "if [ \"$resolve_ip\" = '10.0.0.1' ]; then\n"
            "  echo 'Connection refused' >&2\n"
            "  exit 1\n"
            "fi\n"
            "if [ \"$resolve_ip\" = '10.0.0.2' ]; then\n"
            "  printf 'second-ip-wins' > \"$dest\"\n"
            "  exit 0\n"
            "fi\n"
            "echo 'Could not resolve host' >&2\n"
            "exit 1\n"
        )
        script.chmod(0o755)
        ok, detail = curl_download(url, str(dest), curl_cmd=str(script))
        assert ok is True
        assert dest.read_text() == "second-ip-wins"

    def test_no_doh_query_on_success(self, tmp_path, monkeypatch):
        """Regression guard: a working direct fetch must never trigger DoH."""
        calls = []

        def spy(*args, **kwargs):
            calls.append(args)
            return []

        monkeypatch.setattr("hermes_cli.net_download.resolve_dns_doh", spy)
        script = tmp_path / "ok-curl"
        script.write_text(
            "#!/bin/sh\nurl=''; dest=''; prev=''\n"
            "for a in \"$@\"; do\n"
            "  if [ \"$prev\" = \"-o\" ]; then dest=\"$a\"; prev=''; fi\n"
            "  case \"$a\" in\n"
            "    -o) prev='-o' ;;\n"
            "    http*) url=\"$a\" ;;\n"
            "  esac\n"
            "done\n"
            "printf 'ok' > \"$dest\"\n"
            "exit 0\n"
        )
        script.chmod(0o755)
        dest = tmp_path / "out.sh"
        ok, detail = curl_download(
            "https://raw.githubusercontent.com/x/y/main/z.sh",
            str(dest), curl_cmd=str(script),
        )
        assert ok is True
        assert calls == []

    def test_http_error_does_not_trigger_doh(self, tmp_path, monkeypatch):
        """A 404/403 (real server response) on a non-polluted host must not
        be retried through a different resolver."""
        calls = []

        def spy(*args, **kwargs):
            calls.append(args)
            return []

        monkeypatch.setattr("hermes_cli.net_download.resolve_dns_doh", spy)
        script = tmp_path / "fail-curl"
        script.write_text("#!/bin/sh\necho '404 Not Found' >&2\nexit 1\n")
        script.chmod(0o755)
        dest = tmp_path / "out.sh"
        ok, detail = curl_download(
            "https://example.com/foo.sh", str(dest), curl_cmd=str(script)
        )
        assert ok is False
        assert calls == []
        assert "404" in detail

    def test_doh_failure_preserves_original_error(self, tmp_path, monkeypatch):
        """When DoH itself fails, the original curl error must survive —
        the fallback never masks the root cause."""
        monkeypatch.setattr(
            "hermes_cli.net_download.resolve_dns_doh",
            lambda *a, **k: [],
        )
        script = tmp_path / "fail-curl"
        script.write_text(
            "#!/bin/sh\necho 'Could not resolve host: huggingface.co' >&2\nexit 1\n"
        )
        script.chmod(0o755)
        dest = tmp_path / "out.sh"
        ok, detail = curl_download(
            "https://huggingface.co/api/models", str(dest), curl_cmd=str(script)
        )
        assert ok is False
        assert "Could not resolve host" in detail

    def test_dns_fallback_disabled(self, tmp_path, monkeypatch):
        """dns_fallback=False must skip DoH entirely."""
        calls = []

        def spy(*args, **kwargs):
            calls.append(args)
            return []

        monkeypatch.setattr("hermes_cli.net_download.resolve_dns_doh", spy)
        script = tmp_path / "fail-curl"
        script.write_text(
            "#!/bin/sh\necho 'Could not resolve host: huggingface.co' >&2\nexit 1\n"
        )
        script.chmod(0o755)
        dest = tmp_path / "out.sh"
        ok, detail = curl_download(
            "https://huggingface.co/api/models", str(dest),
            curl_cmd=str(script), dns_fallback=False,
        )
        assert ok is False
        assert calls == []


class TestFetchWithFallbackDns:
    """DNS fallback integrated with the official-then-mirror flow."""

    def test_dns_fallback_succeeds_before_mirror(self, tmp_path):
        """Official fails (poisoned DNS) → DoH fallback wins → mirrors are
        never attempted, even when opted in."""
        dest = tmp_path / "out.sh"
        url = "https://huggingface.co/api/models"
        curl = TestCurlDownloadDnsFallback()._fake_curl_dns(tmp_path)
        ok, detail = fetch_with_fallback(
            url, str(dest), curl_cmd=curl,
            content_class="data", allow_mirrors=True,
        )
        assert ok is True
        assert dest.read_text() == "dns-content"

    def test_dns_fallback_disabled_uses_mirror_when_opted_in(self, tmp_path):
        """dns_fallback=False + data/mirror opt-in → mirror still rescues."""
        dest = tmp_path / "out.bin"
        url = "https://raw.githubusercontent.com/x/y/main/weights.bin"
        mirror1 = f"https://ghfast.top/{url}"
        curl = TestCurlDownload()._fake_curl_script(
            tmp_path, fail_urls=(url,), output="mirror-bytes"
        )
        ok, detail = fetch_with_fallback(
            url, str(dest), curl_cmd=curl,
            content_class="data", allow_mirrors=True, dns_fallback=False,
        )
        assert ok is True
        assert dest.read_text() == "mirror-bytes"
