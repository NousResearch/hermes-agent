"""Unit tests for hermes_cli.security_audit — parsers + OSV plumbing.

These never hit the live OSV API; HTTP is monkeypatched. The live-call path
is exercised in the E2E test embedded in PR validation, not here.
"""

from __future__ import annotations

import http.client
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from hermes_cli import security_audit as sa


# ─── Parsers ──────────────────────────────────────────────────────────────────


class TestRequirementsParser:
    def test_extracts_pinned_versions(self):
        text = "requests==2.20.0\nflask==2.0.1\n"
        assert sa._parse_requirements(text) == [
            ("requests", "2.20.0"),
            ("flask", "2.0.1"),
        ]

    def test_skips_comments_and_options(self):
        text = "# comment\n-r other.txt\n--index-url https://x\nflask==2.0.1\n"
        assert sa._parse_requirements(text) == [("flask", "2.0.1")]




class TestMCPComponentExtraction:
    def test_npx_scoped_pinned(self):
        comp = sa._extract_mcp_component(
            "fs", "npx", ["-y", "@modelcontextprotocol/server-filesystem@0.5.0"]
        )
        assert comp == sa.Component(
            name="@modelcontextprotocol/server-filesystem",
            version="0.5.0",
            ecosystem="npm",
            source="mcp:fs",
        )


    def test_docker_returns_none(self):
        # We don't currently parse docker image refs.
        assert sa._extract_mcp_component("x", "docker", ["run", "-i", "mcp/foo:1.0"]) is None

    def test_empty_args(self):
        assert sa._extract_mcp_component("x", "npx", []) is None


# ─── Plugin discovery ─────────────────────────────────────────────────────────


class TestPluginDiscovery:
    def test_reads_requirements_txt(self, tmp_path: Path):
        plugin = tmp_path / "plugins" / "myplugin"
        plugin.mkdir(parents=True)
        (plugin / "requirements.txt").write_text("requests==2.20.0\n")
        components = sa._discover_plugins(tmp_path)
        assert len(components) == 1
        assert components[0].name == "requests"
        assert components[0].source == "plugin:myplugin"

    def test_skips_when_no_plugins_dir(self, tmp_path: Path):
        assert sa._discover_plugins(tmp_path) == []


# ─── OSV severity extraction ──────────────────────────────────────────────────


class TestSeverityExtraction:
    def test_database_specific_severity(self):
        rec = {"database_specific": {"severity": "HIGH"}}
        assert sa._osv_severity_from_record(rec) == "HIGH"


    def test_fixed_versions_extracted_and_deduped(self):
        rec = {
            "affected": [
                {
                    "ranges": [
                        {
                            "events": [
                                {"introduced": "0"},
                                {"fixed": "2.0.0"},
                            ]
                        }
                    ]
                },
                {"ranges": [{"events": [{"fixed": "2.0.0"}, {"fixed": "1.9.5"}]}]},
            ]
        }
        assert sa._osv_fixed_versions(rec) == ["2.0.0", "1.9.5"]


# ─── End-to-end orchestration with mocked OSV ─────────────────────────────────


class TestRunAudit:
    def test_no_components_returns_empty(self, tmp_path: Path):
        findings = sa.run_audit(
            skip_venv=True, skip_plugins=True, skip_mcp=True, hermes_home=tmp_path
        )
        assert findings == []

    def test_findings_sorted_by_severity_desc(self, tmp_path: Path):
        plugin = tmp_path / "plugins" / "p"
        plugin.mkdir(parents=True)
        (plugin / "requirements.txt").write_text("alpha==1.0.0\nbeta==2.0.0\n")

        def fake_batch(comps):
            return {
                comps[0]: ["LOW-1"],
                comps[1]: ["CRIT-1"],
            }

        def fake_details(ids):
            return {
                "LOW-1": sa.Vulnerability(osv_id="LOW-1", severity="LOW", summary="low"),
                "CRIT-1": sa.Vulnerability(osv_id="CRIT-1", severity="CRITICAL", summary="crit"),
            }

        with patch.object(sa, "_osv_query_batch", side_effect=fake_batch), \
             patch.object(sa, "_osv_fetch_details", side_effect=fake_details):
            findings = sa.run_audit(
                skip_venv=True, skip_plugins=False, skip_mcp=True, hermes_home=tmp_path
            )
        assert len(findings) == 2
        # CRITICAL must come first
        assert findings[0].vuln.osv_id == "CRIT-1"
        assert findings[1].vuln.osv_id == "LOW-1"


# ─── OSV fault tolerance (truncated/aborted HTTP responses) ──────────────────


class TestOSVFaultTolerance:
    """A truncated OSV response (IncompleteRead etc.) must never crash the
    audit: detail fetches degrade to UNKNOWN, batch queries surface as
    RuntimeError. Regression for the 2026-08-01 crash where IncompleteRead
    escaped the except tuple and killed the whole audit."""

    def test_detail_fetch_incomplete_read_degrades(self):
        def fake_urlopen(*args, **kwargs):
            raise http.client.IncompleteRead(b"partial", 1584)

        with patch.object(sa.urllib.request, "urlopen", fake_urlopen):
            details = sa._osv_fetch_details(["GHSA-xxxx-xxxx-xxxx"])
        v = details["GHSA-xxxx-xxxx-xxxx"]
        assert v.severity == "UNKNOWN"
        assert v.osv_id == "GHSA-xxxx-xxxx-xxxx"

    def test_detail_fetch_remote_disconnected_degrades(self):
        def fake_urlopen(*args, **kwargs):
            raise http.client.RemoteDisconnected("closed without response")

        with patch.object(sa.urllib.request, "urlopen", fake_urlopen):
            details = sa._osv_fetch_details(["CVE-2026-0001"])
        assert details["CVE-2026-0001"].severity == "UNKNOWN"

    def test_detail_fetch_bad_json_degrades(self):
        def fake_urlopen(*args, **kwargs):
            class _Resp:
                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    return False

                def read(self):
                    return b"not json {"

            return _Resp()

        with patch.object(sa.urllib.request, "urlopen", fake_urlopen):
            details = sa._osv_fetch_details(["CVE-2026-0002"])
        assert details["CVE-2026-0002"].severity == "UNKNOWN"

    def test_detail_fetch_urlerror_still_degrades(self):
        # Pre-existing tolerance must survive; this is the regression guard.
        def fake_urlopen(*args, **kwargs):
            raise sa.urllib.error.URLError("boom")

        with patch.object(sa.urllib.request, "urlopen", fake_urlopen):
            details = sa._osv_fetch_details(["CVE-2026-0003"])
        assert details["CVE-2026-0003"].severity == "UNKNOWN"

    def test_batch_query_incomplete_read_becomes_runtime_error(self):
        def fake_urlopen(*args, **kwargs):
            raise http.client.IncompleteRead(b"partial", 999)

        comp = SimpleNamespace(name="pkg", ecosystem="PyPI", version="1.0.0")
        with patch.object(sa.urllib.request, "urlopen", fake_urlopen):
            try:
                sa._osv_query_batch([comp])
            except RuntimeError as exc:
                assert "OSV batch query failed" in str(exc)
            else:
                raise AssertionError("expected RuntimeError from truncated batch")


# ─── CLI subcommand exit codes ────────────────────────────────────────────────


class TestExitCodes:
    def _build_args(self, **kwargs):
        import argparse

        defaults = {
            "skip_venv": True,
            "skip_plugins": True,
            "skip_mcp": True,
            "json": False,
            "fail_on": "critical",
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_discovery_runs_once_per_audit(self, tmp_path: Path, monkeypatch, capsys):
        """cmd_security_audit must not scan the venv/plugins/MCP config twice.

        Regression for the double-scan noted in #75485: the component count
        and the audit each ran full discovery independently.
        """
        monkeypatch.setattr(sa, "get_hermes_home", lambda: str(tmp_path))
        calls = {"venv": 0}

        def counting_discover_venv():
            calls["venv"] += 1
            return [sa.Component(name="pkg", version="1.0", ecosystem="PyPI", source="venv")]

        monkeypatch.setattr(sa, "_discover_venv", counting_discover_venv)
        monkeypatch.setattr(sa, "_osv_query_batch", lambda comps: {})
        sa.cmd_security_audit(self._build_args(skip_venv=False))
        capsys.readouterr()
        assert calls["venv"] == 1




    def test_unknown_fail_on_value_exits_two(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.setattr(sa, "get_hermes_home", lambda: str(tmp_path))
        code = sa.cmd_security_audit(self._build_args(fail_on="garbage"))
        assert code == 2
        err = capsys.readouterr().err
        assert "fail-on" in err.lower()

    def test_json_output_shape(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.setattr(sa, "get_hermes_home", lambda: str(tmp_path))
        fake_comp = sa.Component(
            name="pkg", version="1.0", ecosystem="PyPI", source="venv"
        )
        monkeypatch.setattr(sa, "_discover_venv", lambda: [fake_comp])
        monkeypatch.setattr(
            sa, "_osv_query_batch", lambda comps: {fake_comp: ["X-1"]}
        )
        monkeypatch.setattr(
            sa,
            "_osv_fetch_details",
            lambda ids: {
                "X-1": sa.Vulnerability(
                    osv_id="X-1",
                    severity="HIGH",
                    summary="bad",
                    fixed_versions=["1.1"],
                )
            },
        )
        sa.cmd_security_audit(
            self._build_args(skip_venv=False, json=True, fail_on="critical")
        )
        payload = capsys.readouterr().out
        # The bitwarden banner can leak above the json; pick the first { line.
        lines = payload.splitlines()
        json_start = next(i for i, l in enumerate(lines) if l.startswith("{"))
        data = json.loads("\n".join(lines[json_start:]))
        assert data["finding_count"] == 1
        assert data["findings"][0]["severity"] == "HIGH"
        assert data["findings"][0]["fixed_versions"] == ["1.1"]
