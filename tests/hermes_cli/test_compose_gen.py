"""Tests for registry-driven compose generation (hermes_cli.compose_gen)."""

import yaml

from hermes_cli.clients import parse_registry
from hermes_cli.compose_gen import (
    build_service,
    generate_compose,
    host_profile_path,
    isolated_clients,
    isolated_profile_names,
    main,
    render,
    render_isolated_list,
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


class TestIsolatedList:
    """The exclusion artifact (`isolated.list`) the boot gate consumes."""

    def test_profile_names_all(self):
        # Profile names of every isolation: container client, file order.
        assert isolated_profile_names(_registry()) == ("bravo", "charlie")

    def test_profile_names_env_filtered(self):
        assert isolated_profile_names(_registry(), env="prod") == ("bravo",)
        assert isolated_profile_names(_registry(), env="dev") == ("charlie",)

    def test_profile_names_distinct_from_client_name(self):
        # The exclusion set keys on the on-disk PROFILE, not the client name —
        # that's what the boot reconcile and s6 slot (gateway-<profile>) use.
        reg = parse_registry({"clients": [
            {"name": "acme-bot", "profile": "acme", "env": "prod",
             "telegram_token_ref": "T", "isolation": "container"},
        ]})
        assert isolated_profile_names(reg) == ("acme",)

    def test_render_one_profile_per_line_trailing_newline(self):
        text = render_isolated_list(_registry())
        assert text == "bravo\ncharlie\n"

    def test_render_empty_when_none_isolated(self):
        reg = parse_registry({"clients": [
            {"name": "alpha", "env": "prod", "telegram_token_ref": "A"},
        ]})
        # Empty file (no isolated clients) — boot hook treats this as
        # "nothing to exclude", i.e. today's seed-everything behavior.
        assert render_isolated_list(reg) == ""


def _write_registry(tmp_path):
    """Write a minimal registry file and return its path."""
    reg_path = tmp_path / "clients.yaml"
    reg_path.write_text(
        "clients:\n"
        "  - {name: alpha, env: prod, telegram_token_ref: A}\n"
        "  - {name: bravo, env: prod, telegram_token_ref: B, isolation: container}\n",
        encoding="utf-8",
    )
    return reg_path


class TestMainOutputArtifacts:
    """`--output` writes BOTH the compose file and the sibling isolated.list."""

    def test_output_writes_compose_and_isolated_list(self, tmp_path):
        reg_path = _write_registry(tmp_path)
        data_root = tmp_path / "data-prod"
        data_root.mkdir()
        out = tmp_path / "docker-compose.clients.yml"

        rc = main([
            "--registry", str(reg_path),
            "--data-root", str(data_root),
            "--output", str(out),
        ])
        assert rc == 0

        compose = yaml.safe_load(out.read_text())
        assert set(compose["services"]) == {"hermes-bravo"}

        # The sibling exclusion artifact must list the isolated profile so the
        # boot gate keeps it OUT of the shared gateway (no double-run).
        isolated_list = data_root / "isolated.list"
        assert isolated_list.read_text() == "bravo\n"

    def test_output_isolated_list_empty_when_none(self, tmp_path):
        reg_path = tmp_path / "clients.yaml"
        reg_path.write_text(
            "clients:\n"
            "  - {name: alpha, env: prod, telegram_token_ref: A}\n",
            encoding="utf-8",
        )
        data_root = tmp_path / "data-prod"
        data_root.mkdir()
        out = tmp_path / "docker-compose.clients.yml"

        rc = main([
            "--registry", str(reg_path),
            "--data-root", str(data_root),
            "--output", str(out),
        ])
        assert rc == 0
        # isolated.list is still written (empty) so the gate sees a definitive
        # "nothing isolated" rather than a missing-file ambiguity.
        assert (data_root / "isolated.list").read_text() == ""

    def test_stdout_mode_writes_no_isolated_list(self, tmp_path, capsys):
        reg_path = _write_registry(tmp_path)
        data_root = tmp_path / "data-prod"
        data_root.mkdir()

        rc = main([
            "--registry", str(reg_path),
            "--data-root", str(data_root),
        ])
        assert rc == 0
        # No --output → stdout only; no artifact side effects on disk.
        assert not (data_root / "isolated.list").exists()
        assert "hermes-bravo" in capsys.readouterr().out
