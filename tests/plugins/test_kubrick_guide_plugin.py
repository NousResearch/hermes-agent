"""Contract tests for the bundled Kubrick dashboard guide."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = ROOT / "plugins" / "kubrick-guide" / "dashboard"


def test_kubrick_guide_manifest_and_assets_are_complete():
    manifest = json.loads((DASHBOARD / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["name"] == "kubrick-guide"
    assert manifest["label"] == "Kubrick"
    assert manifest["tab"] == {"path": "/kubrick", "position": "after:skills"}
    assert manifest["icon"] == "Eye"

    for asset_key in ("entry", "css"):
        asset = DASHBOARD / manifest[asset_key]
        assert asset.is_file(), f"missing {asset_key}: {asset}"
        assert asset.stat().st_size > 0


def test_kubrick_guide_preserves_governance_and_chat_handoff_contract():
    bundle = (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8")

    required_phrases = (
        'REGISTRY.register("kubrick-guide", KubrickGuide)',
        "Observed form first. Dramatic function first.",
        "NOT_COMPUTABLE",
        "PROPOSED",
        "provider-neutral intent",
        'navigate("/chat")',
        "Copy and open Chat",
        "Current pressure or imbalance",
        "What must visibly change",
        "What is literally on screen",
    )
    for phrase in required_phrases:
        assert phrase in bundle

    assert "__HERMES_SESSION_TOKEN__" not in bundle
    assert "fetch(" not in bundle


def test_kubrick_guide_covers_first_class_production_surfaces():
    bundle = (DASHBOARD / "dist" / "index.js").read_text(encoding="utf-8")

    for job in ("design", "script", "image", "storyboard", "qa", "continuity"):
        assert f'id: "{job}"' in bundle

    for provider in ("generic", "grok-imagine", "flux", "sd3", "midjourney"):
        assert f'["{provider}"' in bundle
