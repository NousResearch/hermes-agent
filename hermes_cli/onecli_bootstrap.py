from __future__ import annotations

import argparse
import json
import shlex
import socket
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_URLS = (
    "http://172.17.0.1:10254/api/container-config",
    "http://127.0.0.1:10254/api/container-config",
    "http://localhost:10254/api/container-config",
)


def _first_stripped(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _get_nested(mapping: dict[str, Any], *path: str) -> Any:
    current: Any = mapping
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _running_in_container() -> bool:
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def _host_gateway_alias() -> str:
    return "172.17.0.1"


def _normalize_proxy_url(proxy_url: str) -> str:
    parsed = urllib.parse.urlsplit(proxy_url)
    hostname = parsed.hostname or ""
    if not hostname:
        return proxy_url
    if hostname != "host.docker.internal" or _running_in_container():
        return proxy_url
    try:
        socket.gethostbyname(hostname)
        return proxy_url
    except OSError:
        pass

    # The bootstrap endpoint may return a container-oriented proxy target.
    # When Hermes runs on the host, rewrite that alias to the host gateway.
    replacement_host = _host_gateway_alias()
    netloc = parsed.netloc
    if "@" in netloc:
        userinfo, _, hostport = netloc.rpartition("@")
        host, sep, port = hostport.partition(":")
        host = replacement_host if host == hostname else host
        netloc = f"{userinfo}@{host}{sep}{port}"
    else:
        host, sep, port = netloc.partition(":")
        host = replacement_host if host == hostname else host
        netloc = f"{host}{sep}{port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _write_temp_pem(prefix: str, pem_text: str, temp_dir: str | None = None) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".pem", dir=temp_dir)
    pem_bytes = pem_text if pem_text.endswith("\n") else f"{pem_text}\n"
    with open(fd, "w", encoding="utf-8", closefd=True) as handle:
        handle.write(pem_bytes)
    return path


def _inject_proxy_token(proxy_url: str, token: str | None) -> str:
    if not token:
        return proxy_url
    parsed = urllib.parse.urlsplit(proxy_url)
    if "@" in parsed.netloc:
        return proxy_url
    netloc = parsed.netloc
    if not netloc:
        return proxy_url
    quoted_token = urllib.parse.quote(token, safe="")
    netloc = f"x:{quoted_token}@{netloc}"
    return urllib.parse.urlunsplit(
        (parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment)
    )


def extract_bootstrap_env(config: dict[str, Any], temp_dir: str | None = None) -> dict[str, str]:
    env_cfg = _get_nested(config, "env")
    if not isinstance(env_cfg, dict):
        env_cfg = {}

    proxy_url = _first_stripped(
        config.get("proxyUrl"),
        config.get("proxy_url"),
        config.get("httpsProxy"),
        config.get("https_proxy"),
        config.get("httpProxy"),
        config.get("http_proxy"),
        env_cfg.get("HTTPS_PROXY"),
        env_cfg.get("https_proxy"),
        env_cfg.get("HTTP_PROXY"),
        env_cfg.get("http_proxy"),
        _get_nested(config, "proxy", "url"),
        _get_nested(config, "proxy", "https"),
        _get_nested(config, "proxy", "http"),
    )
    token = _first_stripped(
        config.get("token"),
        config.get("proxyToken"),
        config.get("proxy_token"),
        config.get("authToken"),
        _get_nested(config, "proxy", "token"),
    )
    proxy_url = _inject_proxy_token(proxy_url, token) if proxy_url else None
    proxy_url = _normalize_proxy_url(proxy_url) if proxy_url else None

    node_ca_path = _first_stripped(
        config.get("caCertPath"),
        config.get("ca_cert_path"),
        config.get("gatewayCaPath"),
        config.get("gateway_ca_path"),
        config.get("nodeExtraCaCerts"),
        config.get("node_extra_ca_certs"),
        env_cfg.get("NODE_EXTRA_CA_CERTS"),
    )
    node_ca_pem = _first_stripped(
        config.get("caCertPem"),
        config.get("ca_cert_pem"),
        config.get("caPem"),
        config.get("ca_pem"),
        config.get("gatewayCaPem"),
        config.get("gateway_ca_pem"),
        config.get("caCertificate"),
    )
    if not node_ca_path and node_ca_pem:
        node_ca_path = _write_temp_pem("onecli-gateway-ca-", node_ca_pem, temp_dir=temp_dir)

    ssl_cert_file = _first_stripped(
        config.get("sslCertFile"),
        config.get("ssl_cert_file"),
        config.get("combinedCaPath"),
        config.get("combined_ca_path"),
        env_cfg.get("SSL_CERT_FILE"),
    )
    ssl_cert_pem = _first_stripped(
        config.get("combinedCaPem"),
        config.get("combined_ca_pem"),
        config.get("sslCertPem"),
        config.get("ssl_cert_pem"),
    )
    if not ssl_cert_file and ssl_cert_pem:
        ssl_cert_file = _write_temp_pem("onecli-combined-ca-", ssl_cert_pem, temp_dir=temp_dir)
    if not ssl_cert_file:
        ssl_cert_file = node_ca_path

    env: dict[str, str] = {}
    if proxy_url:
        env["ONECLI_BOOTSTRAP_PROXY_URL"] = proxy_url
    if node_ca_path:
        env["ONECLI_BOOTSTRAP_NODE_EXTRA_CA_CERTS"] = node_ca_path
    if ssl_cert_file:
        env["ONECLI_BOOTSTRAP_SSL_CERT_FILE"] = ssl_cert_file
    return env


def fetch_container_config(config_url: str, timeout: float = 1.5) -> dict[str, Any]:
    request = urllib.request.Request(config_url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("container-config response was not a JSON object")
    return data


def fetch_first_container_config(
    config_urls: list[str] | tuple[str, ...], timeout: float = 1.5
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    for config_url in config_urls:
        try:
            return fetch_container_config(config_url, timeout=timeout), config_url
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("no container-config URLs were provided")


def format_shell_assignments(env: dict[str, str]) -> str:
    return "\n".join(
        f"{name}={shlex.quote(value)}" for name, value in sorted(env.items()) if value
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and normalize OneCLI bootstrap config")
    parser.add_argument(
        "--config-url",
        action="append",
        default=[],
        help="OneCLI container-config endpoint; may be passed multiple times",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional directory for writing CA PEM files extracted from JSON payloads",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print shell-safe KEY=value assignments for eval",
    )
    args = parser.parse_args(argv)
    config_urls = args.config_url or list(DEFAULT_CONFIG_URLS)

    try:
        config, _source_url = fetch_first_container_config(config_urls)
        env = extract_bootstrap_env(config, temp_dir=args.temp_dir)
    except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError):
        return 0

    if args.shell:
        output = format_shell_assignments(env)
        if output:
            sys.stdout.write(f"{output}\n")
        return 0

    json.dump(env, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
