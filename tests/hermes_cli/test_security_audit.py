"""Unit tests for hermes_cli.security_audit — parsers + OSV plumbing.

These never hit the live OSV API; HTTP is monkeypatched. The live-call path
is exercised in the E2E test embedded in PR validation, not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from importlib.metadata import PathDistribution

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

    def test_fail_on_high_ignores_shadowed_lazy_finding(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """Stale lazy-target aiohttp must not keep --fail-on high red.

        Regression for #92549: the sealed venv is already patched (3.14.3)
        while HERMES_LAZY_INSTALL_TARGET still holds 3.14.1. The shadowed
        copy is reported, but the exit code follows the importable version.
        """
        monkeypatch.setattr(sa, "get_hermes_home", lambda: str(tmp_path))
        venv_comp = sa.Component(
            name="aiohttp", version="3.14.3", ecosystem="PyPI", source=sa.SOURCE_VENV
        )
        shadowed = sa.Component(
            name="aiohttp",
            version="3.14.1",
            ecosystem="PyPI",
            source=sa.SOURCE_LAZY_SHADOWED,
        )
        monkeypatch.setattr(sa, "_discover_venv", lambda: [venv_comp, shadowed])

        def fake_batch(comps):
            out = {}
            for c in comps:
                if c.version == "3.14.1":
                    out[c] = ["GHSA-cq5v-8q36-5273"]
            return out

        monkeypatch.setattr(sa, "_osv_query_batch", fake_batch)
        monkeypatch.setattr(
            sa,
            "_osv_fetch_details",
            lambda ids: {
                "GHSA-cq5v-8q36-5273": sa.Vulnerability(
                    osv_id="GHSA-cq5v-8q36-5273",
                    severity="HIGH",
                    summary="stale lazy copy",
                    fixed_versions=["3.14.3"],
                )
            },
        )
        code = sa.cmd_security_audit(
            self._build_args(skip_venv=False, json=True, fail_on="high")
        )
        payload = capsys.readouterr().out
        lines = payload.splitlines()
        json_start = next(i for i, l in enumerate(lines) if l.startswith("{"))
        data = json.loads("\n".join(lines[json_start:]))
        assert code == 0
        assert data["finding_count"] == 1
        assert data["findings"][0]["source"] == sa.SOURCE_LAZY_SHADOWED
        assert data["findings"][0]["version"] == "3.14.1"

    def test_fail_on_high_still_trips_on_effective_venv(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """A vuln in the imported venv copy must still fail the audit."""
        monkeypatch.setattr(sa, "get_hermes_home", lambda: str(tmp_path))
        venv_comp = sa.Component(
            name="aiohttp", version="3.14.1", ecosystem="PyPI", source=sa.SOURCE_VENV
        )
        shadowed = sa.Component(
            name="aiohttp",
            version="3.14.3",
            ecosystem="PyPI",
            source=sa.SOURCE_LAZY_SHADOWED,
        )
        monkeypatch.setattr(sa, "_discover_venv", lambda: [venv_comp, shadowed])

        def fake_batch(comps):
            out = {}
            for c in comps:
                if c.version == "3.14.1":
                    out[c] = ["GHSA-cq5v-8q36-5273"]
            return out

        monkeypatch.setattr(sa, "_osv_query_batch", fake_batch)
        monkeypatch.setattr(
            sa,
            "_osv_fetch_details",
            lambda ids: {
                "GHSA-cq5v-8q36-5273": sa.Vulnerability(
                    osv_id="GHSA-cq5v-8q36-5273",
                    severity="HIGH",
                    summary="imported copy is vulnerable",
                    fixed_versions=["3.14.3"],
                )
            },
        )
        code = sa.cmd_security_audit(
            self._build_args(skip_venv=False, json=True, fail_on="high")
        )
        payload = capsys.readouterr().out
        lines = payload.splitlines()
        json_start = next(i for i, l in enumerate(lines) if l.startswith("{"))
        data = json.loads("\n".join(lines[json_start:]))
        assert code == 1
        assert data["findings"][0]["source"] == sa.SOURCE_VENV
        assert data["findings"][0]["version"] == "3.14.1"

    def test_fail_on_high_trips_on_effective_lazy_only_package(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        """A package imported only from the lazy target is still fail-on surface."""
        monkeypatch.setattr(sa, "get_hermes_home", lambda: str(tmp_path))
        lazy_comp = sa.Component(
            name="honcho-ai", version="0.1.0", ecosystem="PyPI", source=sa.SOURCE_LAZY
        )
        monkeypatch.setattr(sa, "_discover_venv", lambda: [lazy_comp])
        monkeypatch.setattr(
            sa, "_osv_query_batch", lambda comps: {lazy_comp: ["GHSA-lazy-1"]}
        )
        monkeypatch.setattr(
            sa,
            "_osv_fetch_details",
            lambda ids: {
                "GHSA-lazy-1": sa.Vulnerability(
                    osv_id="GHSA-lazy-1",
                    severity="HIGH",
                    summary="imported from lazy target",
                    fixed_versions=["0.2.0"],
                )
            },
        )
        code = sa.cmd_security_audit(
            self._build_args(skip_venv=False, json=True, fail_on="high")
        )
        capsys.readouterr()
        assert code == 1


# ─── Venv vs durable lazy-target classification ───────────────────────────────


def _write_distinfo(site_dir: Path, name: str, version: str) -> PathDistribution:
    """Create a minimal dist-info tree and return a real PathDistribution."""
    info = site_dir / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    return PathDistribution(info)


class TestVenvLazyClassification:
    def test_shadowed_lazy_aiohttp_is_not_labeled_venv(self, tmp_path: Path):
        """Sealed venv 3.14.3 wins; stale lazy 3.14.1 is lazy-shadowed.

        Mirrors the Docker durable-target layout from #92549:
        HERMES_LAZY_INSTALL_TARGET is appended on sys.path, so the venv
        copy is the one ``import aiohttp`` would load.
        """
        venv_site = tmp_path / "venv" / "lib" / "python3.13" / "site-packages"
        lazy = tmp_path / "lazy-packages"
        venv_dist = _write_distinfo(venv_site, "aiohttp", "3.14.3")
        lazy_dist = _write_distinfo(lazy, "aiohttp", "3.14.1")
        # venv first, lazy last — the durable-target contract.
        search_path = [str(venv_site), str(lazy)]

        components = sa._classify_installed_distributions(
            [venv_dist, lazy_dist],
            search_path=search_path,
            lazy_target=lazy,
        )
        by_source = {(c.source, c.version): c for c in components if c.name.lower() == "aiohttp"}
        assert (sa.SOURCE_VENV, "3.14.3") in by_source
        assert (sa.SOURCE_LAZY_SHADOWED, "3.14.1") in by_source
        assert all(c.source != sa.SOURCE_VENV or c.version != "3.14.1" for c in components)

    def test_lazy_only_package_is_effective_lazy(self, tmp_path: Path):
        """A package that exists only in the lazy target *is* imported."""
        venv_site = tmp_path / "venv" / "site-packages"
        lazy = tmp_path / "lazy-packages"
        venv_site.mkdir(parents=True)
        lazy_dist = _write_distinfo(lazy, "honcho-ai", "1.2.0")
        components = sa._classify_installed_distributions(
            [lazy_dist],
            search_path=[str(venv_site), str(lazy)],
            lazy_target=lazy,
        )
        assert len(components) == 1
        assert components[0].name == "honcho-ai"
        assert components[0].version == "1.2.0"
        assert components[0].source == sa.SOURCE_LAZY

    def test_without_lazy_target_env_everything_stays_venv(self, tmp_path: Path):
        venv_site = tmp_path / "venv" / "site-packages"
        other = tmp_path / "other"
        venv_dist = _write_distinfo(venv_site, "aiohttp", "3.14.3")
        other_dist = _write_distinfo(other, "aiohttp", "3.14.1")
        components = sa._classify_installed_distributions(
            [venv_dist, other_dist],
            search_path=[str(venv_site), str(other)],
            lazy_target=None,
        )
        # Import-path precedence still drops the losing copy so it cannot
        # be labeled as an extra active venv version.
        aio = [c for c in components if c.name.lower() == "aiohttp"]
        assert len(aio) == 1
        assert aio[0].version == "3.14.3"
        assert aio[0].source == sa.SOURCE_VENV

    def test_same_version_in_lazy_is_not_duplicated(self, tmp_path: Path):
        venv_site = tmp_path / "venv" / "site-packages"
        lazy = tmp_path / "lazy-packages"
        venv_dist = _write_distinfo(venv_site, "aiohttp", "3.14.3")
        lazy_dist = _write_distinfo(lazy, "aiohttp", "3.14.3")
        components = sa._classify_installed_distributions(
            [venv_dist, lazy_dist],
            search_path=[str(venv_site), str(lazy)],
            lazy_target=lazy,
        )
        aio = [c for c in components if c.name.lower() == "aiohttp"]
        assert len(aio) == 1
        assert aio[0].source == sa.SOURCE_VENV
        assert aio[0].version == "3.14.3"

    def test_inverted_path_reports_imported_lazy_copy(self, tmp_path: Path):
        """If lazy precedes venv on sys.path, that copy is the imported one."""
        venv_site = tmp_path / "venv" / "site-packages"
        lazy = tmp_path / "lazy-packages"
        venv_dist = _write_distinfo(venv_site, "aiohttp", "3.14.3")
        lazy_dist = _write_distinfo(lazy, "aiohttp", "3.14.1")
        components = sa._classify_installed_distributions(
            [venv_dist, lazy_dist],
            search_path=[str(lazy), str(venv_site)],
            lazy_target=lazy,
        )
        aio = [c for c in components if c.name.lower() == "aiohttp"]
        assert len(aio) == 1
        assert aio[0].source == sa.SOURCE_LAZY
        assert aio[0].version == "3.14.1"

    def test_discover_venv_honors_lazy_target_env(self, tmp_path: Path, monkeypatch):
        """Wire-up: env var + distributions() + classification."""
        venv_site = tmp_path / "venv" / "site-packages"
        lazy = tmp_path / "lazy-packages"
        venv_dist = _write_distinfo(venv_site, "aiohttp", "3.14.3")
        lazy_dist = _write_distinfo(lazy, "aiohttp", "3.14.1")
        monkeypatch.setenv("HERMES_LAZY_INSTALL_TARGET", str(lazy))
        import importlib.metadata as md

        monkeypatch.setattr(md, "distributions", lambda: [venv_dist, lazy_dist])
        captured: dict = {}
        real_classify = sa._classify_installed_distributions

        def fake_classify(dists, *, search_path, lazy_target):
            captured["lazy_target"] = lazy_target
            captured["n_dists"] = sum(1 for _ in dists)
            # Hermetic search_path so this test does not mutate process sys.path.
            return real_classify(
                [venv_dist, lazy_dist],
                search_path=[str(venv_site), str(lazy)],
                lazy_target=lazy_target,
            )

        monkeypatch.setattr(sa, "_classify_installed_distributions", fake_classify)
        components = sa._discover_venv()
        assert captured["lazy_target"] == lazy.resolve()
        assert captured["n_dists"] == 2
        sources = {(c.source, c.version) for c in components if c.name.lower() == "aiohttp"}
        assert (sa.SOURCE_VENV, "3.14.3") in sources
        assert (sa.SOURCE_LAZY_SHADOWED, "3.14.1") in sources

    def test_fail_on_applies_to_importable_sources_only(self):
        assert sa._fail_on_applies(sa.SOURCE_VENV)
        assert sa._fail_on_applies(sa.SOURCE_LAZY)
        assert sa._fail_on_applies("plugin:foo")
        assert sa._fail_on_applies("mcp:bar")
        assert not sa._fail_on_applies(sa.SOURCE_LAZY_SHADOWED)
