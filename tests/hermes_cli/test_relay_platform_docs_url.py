from pathlib import Path


DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/messaging/relay"


def test_relay_platform_override_links_to_existing_docs_page():
    repo_root = Path(__file__).resolve().parents[2]
    web_server_source = (repo_root / "hermes_cli" / "web_server.py").read_text()
    docs_page = repo_root / "website" / "docs" / "user-guide" / "messaging" / "relay.md"

    assert '"relay": {' in web_server_source
    assert f'"docs_url": "{DOCS_URL}"' in web_server_source
    assert docs_page.exists()
