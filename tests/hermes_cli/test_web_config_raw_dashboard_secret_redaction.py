import pytest
"""The raw config editor must not expose or overwrite dashboard credentials."""

import hermes_yaml as yaml

from hermes_cli.web_routers.config_env import DASHBOARD_SECRET_PLACEHOLDER


_HASH = "scrypt$16384$8$1$c2FsdHNhbHQ$ZmFrZWhhc2hmYWtlaGFzaA"
_PASSWORD = "correct-horse-battery-staple"
_SECRET = "s" * 48
_OAUTH_SECRET = "oauth-client-secret-value-1234"


def _raw_yaml():
    return f"""# formatting must survive the redacted GET
agent: {{max_turns: 50}}  # inline comment
dashboard:
  basic_auth:
    username: admin
    password_hash: '{_HASH}'
    password: {_PASSWORD}
    secret: {_SECRET}
  oauth:
    client_id: agent:abc123
    nested:
      client_secret: {_OAUTH_SECRET}
"""


def _client_with_config():
    from starlette.testclient import TestClient

    from hermes_cli.config import get_config_path
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_raw_yaml(), encoding="utf-8")
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return client, path


def test_raw_get_redacts_every_secret_without_reformatting():
    client, _ = _client_with_config()

    response = client.get("/api/config/raw")

    assert response.status_code == 200
    raw = response.json()["yaml"]
    for secret in (_HASH, _PASSWORD, _SECRET, _OAUTH_SECRET):
        assert secret not in response.text
    assert raw.startswith("# formatting must survive the redacted GET\n")
    assert "agent: {max_turns: 50}  # inline comment" in raw
    parsed = yaml.safe_load(raw)
    assert parsed["dashboard"]["basic_auth"]["username"] == "admin"
    assert parsed["dashboard"]["basic_auth"]["password_hash"] == DASHBOARD_SECRET_PLACEHOLDER
    assert parsed["dashboard"]["oauth"]["nested"]["client_secret"] == DASHBOARD_SECRET_PLACEHOLDER


def test_raw_put_round_trip_preserves_secrets():
    client, path = _client_with_config()
    raw = client.get("/api/config/raw").json()["yaml"]
    raw = raw.replace("max_turns: 50", "max_turns: 75")

    response = client.put("/api/config/raw", json={"yaml_text": raw})

    assert response.status_code == 200
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["agent"]["max_turns"] == 75
    assert saved["dashboard"]["basic_auth"]["password_hash"] == _HASH
    assert saved["dashboard"]["basic_auth"]["password"] == _PASSWORD
    assert saved["dashboard"]["basic_auth"]["secret"] == _SECRET
    assert saved["dashboard"]["oauth"]["nested"]["client_secret"] == _OAUTH_SECRET


def test_raw_put_new_secret_replaces_old_value():
    client, path = _client_with_config()
    parsed = yaml.safe_load(client.get("/api/config/raw").json()["yaml"])
    parsed["dashboard"]["basic_auth"]["secret"] = "replacement-secret-value"

    response = client.put("/api/config/raw", json={"yaml_text": yaml.safe_dump(parsed)})

    assert response.status_code == 200
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["dashboard"]["basic_auth"]["secret"] == "replacement-secret-value"
    assert saved["dashboard"]["basic_auth"]["password_hash"] == _HASH
    assert saved["dashboard"]["oauth"]["nested"]["client_secret"] == _OAUTH_SECRET


def test_raw_yaml_redacts_secret_inside_list():
    from hermes_cli.web_routers.analytics import _redact_dashboard_secrets_in_yaml
    text = "dashboard:\n  oauth:\n    clients:\n      - client_id: a\n        client_secret: list-secret-value-1\n"
    out = _redact_dashboard_secrets_in_yaml(text)
    assert "list-secret-value-1" not in out and "client_id: a" in out


@pytest.mark.parametrize("text", [
    # merge key at the dashboard level
    "auth_defaults: &ad\n  basic_auth:\n    password: merged-secret-value-1\ndashboard:\n  <<: *ad\n  theme: x\n",
    # merge key at the root level
    "base: &b\n  dashboard:\n    basic_auth:\n      password: merged-secret-value-1\n<<: *b\nagent: {}\n",
    # merge key inside the auth section
    "pw: &p\n  password: merged-secret-value-1\ndashboard:\n  basic_auth:\n    <<: *p\n    username: u\n",
    # scalar alias
    "x: &s merged-secret-value-1\ndashboard:\n  basic_auth:\n    password: *s\n",
    # whole-section alias
    "c: &c\n  password: merged-secret-value-1\ndashboard:\n  basic_auth: *c\n",
])
def test_raw_yaml_redacts_secrets_reached_through_anchors_and_merges(text):
    # Security-review follow-up: secrets are found from the CONSTRUCTED config, so YAML merge keys and
    # aliases cannot route a dashboard credential around the key-name walk.
    from hermes_cli.web_routers.analytics import _redact_dashboard_secrets_in_yaml
    import hermes_yaml
    assert "merged-secret-value-1" in str(hermes_yaml.safe_load(text).get("dashboard"))  # fixture sanity
    assert "merged-secret-value-1" not in _redact_dashboard_secrets_in_yaml(text)


@pytest.mark.parametrize("literal", ["0x1F2E3D4C", "1_234_567_890", "1.5e10", "0o17777777", "0b1011011101"])
def test_raw_yaml_merged_secret_matches_by_constructed_value_not_text(literal):
    # Security-review follow-up: YAML 1.1 non-canonical number spellings construct to a value whose
    # str() differs from the source text; the merged-secret match must compare constructed values.
    from hermes_cli.web_routers.analytics import _redact_dashboard_secrets_in_yaml
    text = f"defaults: &d\n  basic_auth:\n    password: {literal}\ndashboard:\n  <<: *d\n"
    assert literal not in _redact_dashboard_secrets_in_yaml(text)
