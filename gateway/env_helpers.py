"""Env/bootstrap helpers extracted from gateway/run.py (#54962).

Eleventh slice of the gateway god-file unpacking: SSL certificate
auto-detection for non-standard systems and the config.yaml -> env
bridge for per-turn iteration budgets. Pure env manipulation with no
module state, so the functions stay directly unit-testable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# SSL certificate auto-detection for NixOS and other non-standard systems.
# Must run BEFORE any HTTP library (discord, aiohttp, etc.) is imported.
# ---------------------------------------------------------------------------
def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE if the system doesn't expose CA certs to Python.

    Windows startup paths (Desktop, Scheduled Tasks, installer children) can
    occasionally inherit a stale SSL_CERT_FILE. Returning just because the
    variable is present makes every later httpx/OpenAI client construction fail
    with FileNotFoundError from ssl.load_verify_locations(). Treat a missing
    path as unset and fall back to certifi instead.
    """
    configured_cert = os.environ.get("SSL_CERT_FILE")
    if configured_cert:
        if os.path.exists(configured_cert):
            return  # user already configured it to a real file
        logging.getLogger(__name__).warning(
            "Ignoring stale SSL_CERT_FILE=%r because the path does not exist",
            configured_cert,
        )
        os.environ.pop("SSL_CERT_FILE", None)

    import ssl

    # 1. Python's compiled-in defaults
    paths = ssl.get_default_verify_paths()
    for candidate in (paths.cafile, paths.openssl_cafile):
        if candidate and os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return

    # 2. certifi (ships its own Mozilla bundle)
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        return
    except ImportError:
        pass

    # 3. Common distro / macOS locations
    for candidate in (
        "/etc/ssl/certs/ca-certificates.crt",               # Debian/Ubuntu/Gentoo
        "/etc/pki/tls/certs/ca-bundle.crt",                 # RHEL/CentOS 7
        "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem", # RHEL/CentOS 8+
        "/etc/ssl/ca-bundle.pem",                            # SUSE/OpenSUSE
        "/etc/ssl/cert.pem",                                 # Alpine / macOS
        "/etc/pki/tls/cert.pem",                             # Fedora
        "/usr/local/etc/openssl@1.1/cert.pem",               # macOS Homebrew Intel
        "/opt/homebrew/etc/openssl@1.1/cert.pem",            # macOS Homebrew ARM
    ):
        if os.path.exists(candidate):
            os.environ["SSL_CERT_FILE"] = candidate
            return


def _bridge_max_turns_from_config(home: "Path") -> None:
    """Bridge config.yaml agent.max_turns into HERMES_MAX_ITERATIONS (a global)."""
    config_path = home / 'config.yaml'
    if not config_path.exists():
        return
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw
        # Presence-sensitive env bridge: raw read is deliberate (only keys the
        # user actually wrote get bridged); overlay + expansion applied below.
        cfg = read_user_config_raw(config_path)
        cfg = _expand_env_vars(cfg)
        if not isinstance(cfg, dict):
            cfg = {}
        # Managed scope: keep administrator-pinned values authoritative on every
        # turn too. This per-turn reload re-bridges config→env, so without the
        # overlay a managed agent.max_turns / timezone / redact_secrets would be
        # replaced by the user's value after the first turn. Fail-open.
        try:
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        except Exception:
            pass
    except Exception:
        return

    agent_cfg = cfg.get("agent", {})
    if isinstance(agent_cfg, dict) and "max_turns" in agent_cfg:
        os.environ["HERMES_MAX_ITERATIONS"] = str(agent_cfg["max_turns"])
    # config-authoritative knobs for the session-search index (config.yaml
    # sessions.* wins over stale env; env stays the cross-process carrier).
    sessions_cfg = cfg.get("sessions", {})
    if isinstance(sessions_cfg, dict):
        if "cjk_fts" in sessions_cfg:
            os.environ["HERMES_CJK_FTS"] = str(sessions_cfg["cjk_fts"])
        if "search_slow_ms" in sessions_cfg:
            os.environ["HERMES_SEARCH_SLOW_MS"] = str(sessions_cfg["search_slow_ms"])
