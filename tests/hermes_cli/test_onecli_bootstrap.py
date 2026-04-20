from __future__ import annotations

import json
import socket

from hermes_cli.onecli_bootstrap import (
    DEFAULT_CONFIG_URLS,
    _normalize_proxy_url,
    extract_bootstrap_env,
    fetch_first_container_config,
    format_shell_assignments,
    main,
)


def test_extract_bootstrap_env_uses_paths_and_injects_token():
    env = extract_bootstrap_env(
        {
            "proxyUrl": "http://localhost:10255",
            "token": "abc:123",
            "caCertPath": "/tmp/onecli-gateway-ca.pem",
            "combinedCaPath": "/tmp/onecli-combined-ca.pem",
        }
    )

    assert env == {
        "ONECLI_BOOTSTRAP_NODE_EXTRA_CA_CERTS": "/tmp/onecli-gateway-ca.pem",
        "ONECLI_BOOTSTRAP_PROXY_URL": "http://x:abc%3A123@localhost:10255",
        "ONECLI_BOOTSTRAP_SSL_CERT_FILE": "/tmp/onecli-combined-ca.pem",
    }


def test_extract_bootstrap_env_writes_pem_files_when_paths_missing(tmp_path):
    env = extract_bootstrap_env(
        {
            "proxy": {"url": "http://127.0.0.1:10255"},
            "caCertPem": "NODE CA",
            "combinedCaPem": "SSL CA",
        },
        temp_dir=str(tmp_path),
    )

    node_ca = tmp_path / env["ONECLI_BOOTSTRAP_NODE_EXTRA_CA_CERTS"].split("/")[-1]
    ssl_ca = tmp_path / env["ONECLI_BOOTSTRAP_SSL_CERT_FILE"].split("/")[-1]

    assert env["ONECLI_BOOTSTRAP_PROXY_URL"] == "http://127.0.0.1:10255"
    assert node_ca.read_text() == "NODE CA\n"
    assert ssl_ca.read_text() == "SSL CA\n"


def test_extract_bootstrap_env_reuses_node_ca_for_ssl_when_combined_missing():
    env = extract_bootstrap_env({"caCertPath": "/tmp/onecli-gateway-ca.pem"})

    assert env["ONECLI_BOOTSTRAP_NODE_EXTRA_CA_CERTS"] == "/tmp/onecli-gateway-ca.pem"
    assert env["ONECLI_BOOTSTRAP_SSL_CERT_FILE"] == "/tmp/onecli-gateway-ca.pem"


def test_extract_bootstrap_env_supports_live_nested_payload(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.onecli_bootstrap._running_in_container", lambda: False)
    monkeypatch.setattr("socket.gethostbyname", lambda host: (_ for _ in ()).throw(socket.gaierror()))

    env = extract_bootstrap_env(
        {
            "env": {
                "HTTPS_PROXY": "http://x:secret@host.docker.internal:10255",
                "HTTP_PROXY": "http://x:secret@host.docker.internal:10255",
            },
            "caCertificate": "LIVE CA",
        },
        temp_dir=str(tmp_path),
    )

    assert env["ONECLI_BOOTSTRAP_PROXY_URL"] == "http://x:secret@172.17.0.1:10255"
    assert (tmp_path / env["ONECLI_BOOTSTRAP_NODE_EXTRA_CA_CERTS"].split("/")[-1]).read_text() == "LIVE CA\n"
    assert env["ONECLI_BOOTSTRAP_SSL_CERT_FILE"] == env["ONECLI_BOOTSTRAP_NODE_EXTRA_CA_CERTS"]


def test_normalize_proxy_url_leaves_resolvable_host_docker_internal(monkeypatch):
    monkeypatch.setattr("hermes_cli.onecli_bootstrap._running_in_container", lambda: False)
    monkeypatch.setattr("socket.gethostbyname", lambda host: "192.0.2.10")

    assert _normalize_proxy_url("http://host.docker.internal:10255") == "http://host.docker.internal:10255"


def test_format_shell_assignments_quotes_values():
    output = format_shell_assignments({"B": "two words", "A": "simple"})

    assert output == "A=simple\nB='two words'"


def test_main_shell_output_from_mocked_fetch(monkeypatch, capsys):
    def fake_fetch_first(config_urls: list[str], timeout: float = 1.5):
        assert config_urls == list(DEFAULT_CONFIG_URLS)
        return {"proxyUrl": "http://localhost:10255", "token": "secret"}, config_urls[0]

    monkeypatch.setattr("hermes_cli.onecli_bootstrap.fetch_first_container_config", fake_fetch_first)

    assert main(["--shell"]) == 0
    assert capsys.readouterr().out == "ONECLI_BOOTSTRAP_PROXY_URL=http://x:secret@localhost:10255\n"


def test_main_json_output_from_mocked_fetch(monkeypatch, capsys):
    monkeypatch.setattr(
        "hermes_cli.onecli_bootstrap.fetch_first_container_config",
        lambda config_urls, timeout=1.5: ({"caCertPath": "/tmp/ca.pem"}, config_urls[0]),
    )

    assert main([]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "ONECLI_BOOTSTRAP_NODE_EXTRA_CA_CERTS": "/tmp/ca.pem",
        "ONECLI_BOOTSTRAP_SSL_CERT_FILE": "/tmp/ca.pem",
    }


def test_fetch_first_container_config_tries_multiple_urls(monkeypatch):
    calls: list[str] = []

    def fake_fetch(config_url: str, timeout: float = 1.5):
        calls.append(config_url)
        if config_url == "http://172.17.0.1:10254/api/container-config":
            raise OSError("unreachable")
        return {"ok": True}

    monkeypatch.setattr("hermes_cli.onecli_bootstrap.fetch_container_config", fake_fetch)

    config, used_url = fetch_first_container_config(list(DEFAULT_CONFIG_URLS[:2]))

    assert config == {"ok": True}
    assert used_url == "http://127.0.0.1:10254/api/container-config"
    assert calls == list(DEFAULT_CONFIG_URLS[:2])
