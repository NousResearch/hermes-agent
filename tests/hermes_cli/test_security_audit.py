"""Unit tests for hermes_cli.security_audit — parsers + OSV plumbing.

These never hit the live OSV API; HTTP is monkeypatched. The live-call path
is exercised in the E2E test embedded in PR validation, not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch


from hermes_cli import security_audit as sa


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.read_sizes: list[int] = []

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if size is None or size < 0:
            return self.body
        return self.body[:size]


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

    def test_skips_unpinned(self):
        # We deliberately don't try to map >=, ~=, or bare-name deps to OSV.
        text = "requests>=2.0\ntyping-extensions\nflask~=2.0\n"
        assert sa._parse_requirements(text) == []

    def test_handles_extras_and_markers(self):
        text = 'requests[security]==2.20.0\nflask==2.0.1 ; python_version >= "3.8"\n'
        assert sa._parse_requirements(text) == [
            ("requests", "2.20.0"),
            ("flask", "2.0.1"),
        ]

    def test_handles_empty(self):
        assert sa._parse_requirements("") == []
        assert sa._parse_requirements("   \n\n   ") == []


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

    def test_npx_full_path_command(self):
        comp = sa._extract_mcp_component(
            "fetch", "/usr/local/bin/npx", ["mcp-server-fetch@1.2.3"]
        )
        assert comp is not None
        assert comp.name == "mcp-server-fetch"
        assert comp.version == "1.2.3"

    def test_uvx_pinned(self):
        comp = sa._extract_mcp_component("time", "uvx", ["mcp-server-time==2.1.0"])
        assert comp is not None
        assert comp.ecosystem == "PyPI"
        assert comp.name == "mcp-server-time"
        assert comp.version == "2.1.0"

    def test_unpinned_returns_none(self):
        # Bare npx package name = "latest" at runtime; not an audit subject.
        assert sa._extract_mcp_component("x", "npx", ["-y", "some-pkg"]) is None

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

    def test_skips_hidden_dirs(self, tmp_path: Path):
        (tmp_path / "plugins" / ".hidden").mkdir(parents=True)
        (tmp_path / "plugins" / ".hidden" / "requirements.txt").write_text(
            "requests==2.20.0\n"
        )
        assert sa._discover_plugins(tmp_path) == []

    def test_reads_pyproject_dependencies(self, tmp_path: Path):
        plugin = tmp_path / "plugins" / "py"
        plugin.mkdir(parents=True)
        (plugin / "pyproject.toml").write_text(
            '[project]\ndependencies = ["flask==2.0.1", "uvicorn>=0.20"]\n'
        )
        components = sa._discover_plugins(tmp_path)
        # uvicorn>=0.20 is unpinned, so only flask comes through
        assert len(components) == 1
        assert components[0].name == "flask"
        assert components[0].version == "2.0.1"


# ─── OSV severity extraction ──────────────────────────────────────────────────


class TestSeverityExtraction:
    def test_database_specific_severity(self):
        rec = {"database_specific": {"severity": "HIGH"}}
        assert sa._osv_severity_from_record(rec) == "HIGH"

    def test_unknown_when_no_severity(self):
        assert sa._osv_severity_from_record({}) == "UNKNOWN"

    def test_ecosystem_specific_fallback(self):
        rec = {"affected": [{"ecosystem_specific": {"severity": "MODERATE"}}]}
        assert sa._osv_severity_from_record(rec) == "MODERATE"

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


class TestHTTPJson:
    def test_http_get_json_bounds_response_read(self, monkeypatch):
        response = _FakeHTTPResponse(b'{"ok": true}')
        captured = []

        def _urlopen(req, timeout):
            captured.append((req, timeout))
            return response

        monkeypatch.setattr(sa.urllib.request, "urlopen", _urlopen)

        assert sa._http_get_json("https://api.osv.dev/v1/vulns/X") == {"ok": True}
        assert response.read_sizes == [
            sa._HTTP_JSON_RESPONSE_BODY_MAX_BYTES + 1
        ]
        assert captured[0][0].full_url == "https://api.osv.dev/v1/vulns/X"
        assert captured[0][1] == sa.HTTP_TIMEOUT

    def test_osv_batch_wraps_oversized_response_as_runtime_error(self, monkeypatch):
        response = _FakeHTTPResponse(
            b"x" * (sa._HTTP_JSON_RESPONSE_BODY_MAX_BYTES + 1)
        )
        monkeypatch.setattr(
            sa.urllib.request,
            "urlopen",
            lambda *args, **kwargs: response,
        )
        component = sa.Component(
            name="pkg",
            version="1.0",
            ecosystem="PyPI",
            source="venv",
        )

        try:
            sa._osv_query_batch([component])
        except RuntimeError as exc:
            assert "OSV batch query failed" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

        assert response.read_sizes == [
            sa._HTTP_JSON_RESPONSE_BODY_MAX_BYTES + 1
        ]


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

    def test_clean_audit_exits_zero(self, tmp_path: Path, monkeypatch, capsys):
        monkeypatch.setattr(sa, "get_hermes_home", lambda: str(tmp_path))
        # Everything skipped → no components → exit 0
        code = sa.cmd_security_audit(self._build_args())
        assert code == 0
        out = capsys.readouterr().out
        assert "No components" in out or "0 component" in out

    def test_finding_above_threshold_exits_one(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(sa, "get_hermes_home", lambda: str(tmp_path))
        # Force a venv discovery to return one component, OSV to flag it CRITICAL
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
            lambda ids: {"X-1": sa.Vulnerability(osv_id="X-1", severity="CRITICAL")},
        )
        code = sa.cmd_security_audit(
            self._build_args(skip_venv=False, fail_on="critical")
        )
        assert code == 1

    def test_finding_below_threshold_exits_zero(self, tmp_path: Path, monkeypatch):
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
            lambda ids: {"X-1": sa.Vulnerability(osv_id="X-1", severity="MODERATE")},
        )
        code = sa.cmd_security_audit(
            self._build_args(skip_venv=False, fail_on="critical")
        )
        assert code == 0

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
