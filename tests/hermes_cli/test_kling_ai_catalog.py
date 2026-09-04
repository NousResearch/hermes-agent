from pathlib import Path

from hermes_cli.mcp_catalog import _build_server_config, _parse_manifest


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "optional-mcps" / "kling-ai" / "manifest.yaml"


def test_kling_ai_catalog_manifest() -> None:
    entry = _parse_manifest(MANIFEST)

    assert entry.name == "Plugin-Hermes-kling-ai"
    assert entry.transport.type == "http"
    assert entry.transport.url == "https://kling.ai/mcp"
    assert entry.auth.type == "oauth"
    assert entry.auth.env == []
    assert entry.suggest is not None
    assert entry.suggest.keywords == ["kling", "kling ai", "klingai", "可灵", "可灵ai"]
    assert entry.suggest.hosts == ["klingai.com", "kling.ai"]
    assert "KLING_AI_MCP_URL" not in MANIFEST.read_text(encoding="utf-8")


def test_kling_ai_catalog_builds_one_oauth_server() -> None:
    entry = _parse_manifest(MANIFEST)

    assert _build_server_config(entry, {}) == {
        "url": "https://kling.ai/mcp",
        "auth": "oauth",
    }
