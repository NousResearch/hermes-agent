"""Tests for World Monitor PDB situation report helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "worldmonitor-osint"


def _load_module(name: str, filename: str):
    mod_name = f"worldmonitor_osint_{name}_test"
    for key in list(sys.modules):
        if key == mod_name or key.startswith(f"{mod_name}."):
            del sys.modules[key]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        PLUGIN_DIR / filename,
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_high_threats_dedupes_headlines():
    threat_extract = _load_module("threat_extract", "threat_extract.py")
    wm = {
        "sections": {
            "news_digest": {
                "categories": {
                    "politics": {
                        "items": [
                            {
                                "title": "Conflict headline",
                                "url": "https://www.defense.gov/News/Release/1",
                                "threat": {
                                    "level": "THREAT_LEVEL_HIGH",
                                    "category": "conflict",
                                },
                            },
                            {
                                "title": "Conflict headline",
                                "threat": {
                                    "level": "THREAT_LEVEL_HIGH",
                                    "category": "conflict",
                                },
                            },
                        ]
                    }
                }
            },
            "risk_scores": {
                "ciiScores": [
                    {"region": "UA", "combinedScore": 59, "trend": "STABLE"},
                    {"region": "JP", "combinedScore": 40, "trend": "STABLE"},
                ]
            },
        }
    }
    out = threat_extract.extract_high_threats(wm)
    assert out["unique_high_threat_count"] == 1
    assert out["elevated_cii_regions"][0]["region"] == "UA"
    assert out["high_threat_headlines"][0]["url"].startswith("https://")


def test_build_pdb_markdown_milspec_sections():
    situation_report = _load_module("situation_report", "situation_report.py")
    milspec_prose = _load_module("milspec_prose", "milspec_prose.py")
    fusion = {
        "source_mode": "mock",
        "worldmonitor": {"tier": "free"},
        "primary_sources": {"shinka_source_mode": "mock"},
        "shinka_milspec": {
            "success": True,
            "runs": [{"scenario_id": "taiwan_contingency_overview", "total_score": 88.0}],
        },
    }
    threats = {
        "unique_high_threat_count": 1,
        "high_threat_headlines": [
            {
                "title": "Test conflict headline",
                "url": "https://www.state.gov/example",
                "threat_category": "conflict",
            }
        ],
        "high_threat_by_category": {"conflict": ["Test conflict headline"]},
        "elevated_cii_regions": [],
    }
    enrichment = {
        "egov": {
            "citations": [
                {
                    "success": True,
                    "label": "憲法9条",
                    "snippet": "第九条 …",
                    "citation": "[出典: e-Gov — 憲法 第9条] https://laws.e-gov.go.jp/law/x",
                }
            ]
        },
        "stats": {"egov_citations_ok": 1, "headlines_primary_resolved": 0},
    }
    md = situation_report.build_pdb_markdown(
        slot="morning",
        fusion=fusion,
        threats=threats,
        enrichment=enrichment,
    )
    assert "MILSPEC準拠" in md
    assert "SOURCE INTEGRITY" in md
    assert "JAPAN LAW PRIMARY SOURCES" in md
    assert "出典:" in md
    assert "憲法9条" in md
    assert "taiwan_contingency_overview" in md
    assert "IMPLICATIONS FOR JAPAN" not in md


def test_apply_enrichment_to_threats():
    primary_backfill = _load_module("primary_backfill", "primary_backfill.py")
    threats = {
        "high_threat_headlines": [
            {"title": "Headline A", "url": "https://news.example.com/a", "threat_category": "conflict"}
        ]
    }
    enrichment = {
        "headline_backfill": [
            {
                "title": "Headline A",
                "primary_url": "https://www.mod.go.jp/press/",
                "backfill_method": "ddgs_site_primary",
                "source_tier": "PRIMARY",
            }
        ]
    }
    out = primary_backfill.apply_enrichment_to_threats(threats, enrichment)
    assert out["high_threat_headlines"][0]["url"] == "https://www.mod.go.jp/press/"
    assert out["high_threat_headlines"][0]["backfill_method"] == "ddgs_site_primary"


def test_classify_source_tier_primary_gov():
    milspec_prose = _load_module("milspec_prose", "milspec_prose.py")
    assert milspec_prose.classify_source_tier("https://www.mod.go.jp/j/press/") == "PRIMARY"
    assert milspec_prose.classify_source_tier("") == "UNVERIFIED"
    assert milspec_prose.classify_source_tier("https://example.com/news") == "SECONDARY"
