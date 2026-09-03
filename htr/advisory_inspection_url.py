"""Task 29 — offline URL classification (R5-08 / T089 / T154-T160)."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from htr.advisory_inspection_constants import MAX_URL_UTF8_BYTES


@dataclass
class UrlClassification:
    budget_exceeded: bool = False
    scheme_status: str = "link_scheme_not_applicable"
    host_status: str = "link_host_not_applicable"
    port_status: str = "link_port_not_applicable"
    structure_status: str = "link_structure_not_applicable"
    findings: list[str] = field(default_factory=list)
    stopped_early: bool = False


def _has_malformed_percent_escape(url: str) -> bool:
    i = 0
    while i < len(url):
        if url[i] == "%":
            if i + 2 >= len(url):
                return True
            hex_part = url[i + 1 : i + 3]
            if not re.fullmatch(r"[0-9A-Fa-f]{2}", hex_part):
                return True
            i += 3
            continue
        i += 1
    return False


def _has_percent_encoded_traversal(url: str) -> bool:
    lowered = url.lower()
    return "%2e%2e" in lowered or "%2f%2e%2e" in lowered or "%2f%2e" in lowered


def _host_from_brackets(host: str) -> str:
    if host.startswith("[") and host.endswith("]"):
        return host[1:-1]
    return host


def _classify_ipv4_host(host: str, findings: list[str]) -> str:
    if host == "localhost":
        return "link_host_localhost_name_prohibited"
    if host.startswith("127."):
        return "link_host_loopback_prohibited"
    if host.startswith("192.168."):
        return "link_host_private_prohibited"
    if host.startswith("169.254."):
        return "link_host_link_local_prohibited"
    if host.startswith("224.") or host.startswith("239."):
        return "link_host_multicast_prohibited"
    if host == "0.0.0.0":
        return "link_host_unspecified_prohibited"
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", host):
        return "link_host_ipv4_literal"
    return "link_host_unknown"


def _classify_ip_address(addr: ipaddress.IPv4Address | ipaddress.IPv6Address, findings: list[str]) -> str:
    if isinstance(addr, ipaddress.IPv4Address):
        return _classify_ipv4_host(str(addr), findings)

    findings.append("link_host_ipv6_literal")
    if addr.is_loopback:
        return "link_host_loopback_prohibited"
    if addr.is_private:
        return "link_host_private_prohibited"
    if addr.is_link_local:
        return "link_host_link_local_prohibited"
    if addr.is_multicast:
        return "link_host_multicast_prohibited"
    if addr.is_unspecified:
        return "link_host_unspecified_prohibited"
    if addr.is_reserved:
        return "link_host_reserved_prohibited"
    return "link_host_unknown"


def _classify_host_token(host_token: str, findings: list[str]) -> str:
    host = _host_from_brackets(host_token)
    if not host:
        return "link_host_empty_rejected"

    if "%" in host_token:
        return "link_host_percent_encoded_rejected"

    if host.startswith("[") or (":" in host and not re.match(r"^\d+\.\d+\.\d+\.\d+$", host)):
        try:
            if host.lower().startswith("::ffff:"):
                findings.append("link_host_ipv6_literal")
                findings.append("link_host_ipv4_mapped_ipv6")
                mapped = ipaddress.IPv6Address(host).ipv4_mapped
                if mapped is not None:
                    return _classify_ipv4_host(str(mapped), findings)
                return "link_host_unknown"
            addr = ipaddress.IPv6Address(host)
            return _classify_ip_address(addr, findings)
        except ValueError:
            return "link_host_syntax_rejected"

    if re.match(r"^\d+\.\d+\.\d+\.\d+$", host):
        return _classify_ipv4_host(host, findings)

    host_status = "link_host_unknown"
    if any(ord(c) > 127 for c in host):
        findings.append("link_host_unicode_observed")
    if host.startswith("xn--"):
        findings.append("link_host_alabel_observed")
    if host.endswith("."):
        findings.append("link_host_trailing_dot_observed")
    return host_status


def classify_url_full(url: str) -> UrlClassification:
    """Classify a URL string for advisory link inspection."""
    result = UrlClassification()
    findings = result.findings

    if len(url.encode("utf-8")) > MAX_URL_UTF8_BYTES:
        result.budget_exceeded = True
        result.stopped_early = True
        return result

    for ch in url:
        cp = ord(ch)
        if cp <= 0x1F or cp == 0x7F:
            findings.append("link_control_character_rejected")
            break
    if "\\" in url:
        findings.append("link_backslash_rejected")
    if _has_malformed_percent_escape(url):
        findings.append("link_malformed_percent_escape")
    if _has_percent_encoded_traversal(url):
        findings.append("link_percent_encoded_traversal_observed")

    if url.startswith("//"):
        result.scheme_status = "link_scheme_relative_rejected"
        result.host_status = "link_host_parse_not_reached"
        result.port_status = "link_port_parse_not_reached"
        return result

    if "://" not in url and not url.startswith("/"):
        if url.startswith("?") or url.startswith("#"):
            result.scheme_status = "link_relative_reference_rejected"
            result.host_status = "link_host_parse_not_reached"
            result.port_status = "link_port_parse_not_reached"
            return result
        if url.startswith("./") or url.startswith("../") or ("/" not in url and ":" not in url):
            result.scheme_status = "link_relative_reference_rejected"
            result.host_status = "link_host_parse_not_reached"
            result.port_status = "link_port_parse_not_reached"
            return result

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme == "file":
        result.scheme_status = "link_scheme_file_prohibited"
    elif scheme == "javascript":
        result.scheme_status = "link_scheme_javascript_prohibited"
    elif scheme == "data":
        result.scheme_status = "link_scheme_data_prohibited"
    elif scheme == "ftp":
        result.scheme_status = "link_scheme_ftp_prohibited"
    elif scheme in {"http", "https"}:
        result.scheme_status = (
            "link_scheme_http_declared_offline" if scheme == "http" else "link_scheme_https_declared_offline"
        )
        findings.extend(
            [
                "link_remote_reference_not_fetched",
                "link_reachability_not_inspected",
                "link_content_identity_not_verified",
            ]
        )
        if scheme == "http":
            findings.append("link_http_cleartext_risk")
    elif scheme:
        result.scheme_status = "link_scheme_custom_prohibited"
    else:
        result.scheme_status = "link_relative_reference_rejected"
        result.host_status = "link_host_parse_not_reached"
        result.port_status = "link_port_parse_not_reached"
        return result

    if parsed.username or parsed.password:
        findings.append("link_credentials_prohibited")

    netloc = parsed.netloc
    if "@" in netloc:
        host_part = netloc.rsplit("@", 1)[-1]
    else:
        host_part = netloc

    if not host_part:
        result.host_status = "link_host_empty_rejected"
    else:
        host_token = host_part
        if host_token.startswith("[") and "]" in host_token:
            host_token = host_token[: host_token.index("]") + 1]
        elif ":" in host_token and not host_token.startswith("["):
            host_token = host_token.rsplit(":", 1)[0]
        result.host_status = _classify_host_token(host_token, findings)

    if parsed.port is not None:
        if parsed.port < 1 or parsed.port > 65535:
            result.port_status = "link_port_invalid_syntax"
        else:
            result.port_status = "link_port_observed"
    elif scheme in {"http", "https"} and result.host_status not in {
        "link_host_empty_rejected",
        "link_host_percent_encoded_rejected",
        "link_host_syntax_rejected",
        "link_host_parse_not_reached",
    }:
        result.port_status = "link_port_default_implicit"

    if parsed.query:
        findings.append("link_query_observed")
    if parsed.fragment:
        findings.append("link_fragment_observed")
    if "@" in parsed.netloc and parsed.hostname:
        findings.append("link_ambiguous_authority")
    if parsed.query or parsed.fragment or scheme in {"http", "https"}:
        result.structure_status = "link_structure_observed"

    return result
