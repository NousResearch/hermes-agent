"""Tests for registry-driven compose generation (hermes_cli.compose_gen)."""

import yaml

from hermes_cli.clients import parse_registry
from hermes_cli.compose_gen import (
    build_service,
    generate_compose,
    host_profile_path,
    isolated_clients,
    render,
)


def _registry():
    return parse_registry({"clients": [
        {"name": "default", "env": "dev"},
        {"name": "alpha", "env": "prod", "telegram_token_ref": "A"},          # shared
        {"name": "bravo", "env": "prod", "telegram_token_ref": "B",
         "isolation": "container"},                                            # isolated
        {"name": "charlie", "env": "dev", "telegram_token_ref": "C",
         "isolation": "container"},                                            # isolated (dev)
    ]})


class TestSelection:
    def test_only_container_clients(self):
        names = {c.name for c in isolated_clients(_registry())}
        assert names == {"bravo", "charlie"}

    def test_env_filter(self):
        names = {c.name for c in isolated_clients(_registry(), env="prod")}
        assert names == {"bravo"}


class TestBuildService:
    def test_service_fields(self):
        c = _registry().get("bravo")
        svc = build_service(c, data_root="/opt/data-prod")
        assert svc["image"] == "hermes-agent"
        assert svc["container_name"] == "hermes-bravo"
        assert svc["restart"] == "unless-stopped"
        assert svc["network_mode"] == "host"
        assert svc["command"] == ["gateway", "run"]
        assert svc["volumes"] == ["/opt/data-prod/profiles/bravo:/opt/data"]
        assert svc["env_file"] == ["/opt/data-prod/secrets/bravo.env"]

    def test_host_profile_path_named_vs_default(self):
        reg = _registry()
        assert host_profile_path(reg.get("bravo"), "/opt/data-prod") == "/opt/data-prod/profiles/bravo"
        assert host_profile_path(reg.get("default"), "/opt/data-prod") == "/opt/data-prod"

    def test_overrides(self):
        c = _registry().get("bravo")
        svc = build_service(c, data_root="/srv", image="custom:1", network_mode="bridge")
        assert svc["image"] == "custom:1"
        assert svc["network_mode"] == "bridge"
        assert svc["volumes"] == ["/srv/profiles/bravo:/opt/data"]


class TestGenerate:
    def test_generate_all_isolated(self):
        compose = generate_compose(_registry(), data_root="/opt/data-prod")
        assert set(compose["services"]) == {"hermes-bravo", "hermes-charlie"}

    def test_generate_env_filtered(self):
        compose = generate_compose(_registry(), env="prod")
        assert set(compose["services"]) == {"hermes-bravo"}

    def test_empty_when_none_isolated(self):
        reg = parse_registry({"clients": [
            {"name": "alpha", "env": "prod", "telegram_token_ref": "A"},
        ]})
        assert generate_compose(reg)["services"] == {}

    def test_render_is_valid_yaml_with_header(self):
        compose = generate_compose(_registry(), env="prod")
        text = render(compose, filename="docker-compose.clients.yml")
        assert text.lstrip().startswith("#")             # header comment
        parsed = yaml.safe_load(text)                    # body parses
        assert "hermes-bravo" in parsed["services"]
        # round-trips to the same structure (header is comments only)
        assert parsed["services"]["hermes-bravo"]["container_name"] == "hermes-bravo"
