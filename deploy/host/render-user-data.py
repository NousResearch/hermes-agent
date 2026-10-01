#!/usr/bin/env python3
"""Render per-matter cloud-init user-data for a litco-agent matter host.

The LitKit control plane calls this with a JSON spec and passes the output as
``user_data`` when it creates a droplet from a ``litco-agent-host-<version>``
snapshot::

    render-user-data.py spec.json > user-data.yaml
    render-user-data.py - < spec.json      # spec on stdin

Spec (camelCase, as the control plane sends it)::

    matterId          str   required  LitKit matter id
    instanceUrl       str   required  https://<firm>.litco.ai
    hostSecret        str   required  secret; >= 32 chars
    agentToken        str   required  secret; lkm_...
    matterHome        str   optional  default /home/hermes/matter
    modelProvider     str   required  Hermes provider slug (anthropic, openrouter, custom, ...)
    model             str   optional  model name, e.g. anthropic/claude-opus-4.6
    modelBaseUrl      str   optional  OpenAI-compatible base URL (provider custom)
    modelKey          str   optional  secret; lands in the provider's key variable
    typesafeApiKey    str   optional  secret; TYPESAFE_API_KEY, for the litkit_jev first-pass screen
    tailscaleAuthKey  str   optional  secret; joins the tailnet with SSH when given
    hostname          str   optional  matter-<shortid>; derived from matterId when absent
    egressPolicy      str   required  "open" | "allowlist"
    egressAllowlist   list  optional  [{"cidr": "203.0.113.0/24", "port": 443, "proto": "tcp"}]
    approvalsMode     str   optional  "" (upstream default) | manual | smart | off
    turnHost          str   optional  bind address, default 0.0.0.0 (ufw admits tailscale0 only)
    turnPort          int   optional  default 8765
    channelEnv        dict  optional  native Slack/Telegram settings, from CHANNEL_ENV_KEYS

Secrets appear in the output only inside the base64 content of
/etc/litco-agent/env (and the Tailscale key file). Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import ipaddress
import json
import re
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("cloud-init.yaml.tmpl")
USER_DATA_LIMIT = 64 * 1024  # DigitalOcean's user_data cap

DEFAULT_MATTER_HOME = "/home/hermes/matter"
DEFAULT_TURN_HOST = "0.0.0.0"
DEFAULT_TURN_PORT = 8765
APPROVAL_MODES = ("", "manual", "smart", "off")
EGRESS_POLICIES = ("open", "allowlist")

# Hermes reads each provider's key from its own variable. "custom" reads the
# variable named by model.key_env; the env file then carries the non-secret
# LITCO_MODEL_KEY_ENV=LITCO_MODEL_API_KEY, which litco-agent-init turns into key_env.
PROVIDER_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "nous-api": "NOUS_API_KEY",
    "zai": "GLM_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "huggingface": "HF_TOKEN",
    "custom": "LITCO_MODEL_API_KEY",
}

# The TypeSafe key for litkit_jev (Jev first-pass screen). Hermes's litkit
# toolset reads it from the gateway's environment.
TYPESAFE_KEY_ENV = "TYPESAFE_API_KEY"

# Native channel settings the gateway reads from its environment. Values are
# treated as secrets (they include bot tokens) and go only into the env file.
CHANNEL_ENV_KEYS = (
    "SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "SLACK_ALLOWED_USERS", "SLACK_HOME_CHANNEL",
    "SLACK_IGNORED_CHANNELS", "TELEGRAM_BOT_TOKEN", "TELEGRAM_ALLOWED_USERS",
    "TELEGRAM_GROUP_ALLOWED_USERS", "TELEGRAM_GROUP_ALLOWED_CHATS",
)

_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_HOSTNAME = re.compile(r"^matter-[a-z0-9]([a-z0-9-]{0,54}[a-z0-9])?$")
_PROVIDER = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
_PLAIN = re.compile(r"^[A-Za-z0-9._:/@+-]*$")
_TOKEN = re.compile(r"\{\{(\w+)\}\}")


class SpecError(ValueError):
    pass


def _str(spec: dict, key: str, *, required: bool = False, default: str = "") -> str:
    value = spec.get(key, default)
    if value is None:
        value = default
    if not isinstance(value, str):
        raise SpecError(f"{key} must be a string")
    value = value.strip()
    if required and not value:
        raise SpecError(f"{key} is required")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise SpecError(f"{key} contains control characters")
    return value


def short_id(matter_id: str) -> str:
    alnum = re.sub(r"[^a-z0-9]", "", matter_id.lower())
    if not alnum:
        raise SpecError("matterId has no letters or digits to build a hostname from")
    return alnum[:8]


def env_line(key: str, value: str) -> str:
    """One systemd EnvironmentFile line; double quotes, with \\ and " escaped."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'{key}="{escaped}"'


def validate(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    matter_id = _str(spec, "matterId", required=True)
    if not _ID.match(matter_id):
        raise SpecError("matterId must be letters, digits, - or _")
    instance_url = _str(spec, "instanceUrl", required=True)
    if not re.match(r"^https?://[A-Za-z0-9.-]+(:\d+)?(/[A-Za-z0-9._/-]*)?$", instance_url):
        raise SpecError("instanceUrl must be an http(s) URL")
    host_secret = _str(spec, "hostSecret", required=True)
    if len(host_secret) < 32:
        raise SpecError("hostSecret must be at least 32 characters")
    agent_token = _str(spec, "agentToken", required=True)
    if not agent_token.startswith("lkm_"):
        raise SpecError("agentToken must be a matter-pinned LitKit token (lkm_...)")
    matter_home = _str(spec, "matterHome", default=DEFAULT_MATTER_HOME) or DEFAULT_MATTER_HOME
    if not matter_home.startswith("/") or ".." in matter_home.split("/") or not _PLAIN.match(matter_home):
        raise SpecError("matterHome must be a plain absolute path")
    provider = _str(spec, "modelProvider", required=True).lower()
    if not _PROVIDER.match(provider):
        raise SpecError("modelProvider must be a Hermes provider slug")
    model = _str(spec, "model")
    model_base_url = _str(spec, "modelBaseUrl")
    for key, value in (("model", model), ("modelBaseUrl", model_base_url)):
        if not _PLAIN.match(value):
            raise SpecError(f"{key} contains characters that cannot go into the profile")
    model_key = _str(spec, "modelKey")
    if model_key and provider not in PROVIDER_KEY_ENV:
        raise SpecError(f"no known key variable for provider {provider!r}; "
                        f"one of {', '.join(sorted(PROVIDER_KEY_ENV))}")
    typesafe_key = _str(spec, "typesafeApiKey")
    tailscale_key = _str(spec, "tailscaleAuthKey")
    hostname = _str(spec, "hostname") or f"matter-{short_id(matter_id)}"
    if not _HOSTNAME.match(hostname):
        raise SpecError("hostname must look like matter-<shortid> (lowercase letters, digits, -)")
    egress = _str(spec, "egressPolicy", required=True)
    if egress not in EGRESS_POLICIES:
        raise SpecError("egressPolicy must be 'open' or 'allowlist'")
    allowlist = spec.get("egressAllowlist") or []
    if not isinstance(allowlist, list):
        raise SpecError("egressAllowlist must be a list")
    if allowlist and egress != "allowlist":
        raise SpecError("egressAllowlist is only meaningful with egressPolicy=allowlist")
    rules = []
    for entry in allowlist:
        if not isinstance(entry, dict) or "cidr" not in entry:
            raise SpecError("each egressAllowlist entry needs a cidr")
        try:
            net = ipaddress.ip_network(str(entry["cidr"]), strict=False)
        except ValueError as exc:
            raise SpecError(f"bad egressAllowlist cidr: {exc}") from exc
        port = entry.get("port", 443)
        proto = str(entry.get("proto", "tcp")).lower()
        if not isinstance(port, int) or not 0 < port < 65536:
            raise SpecError("egressAllowlist port must be an integer port")
        if proto not in ("tcp", "udp"):
            raise SpecError("egressAllowlist proto must be tcp or udp")
        rules.append((str(net), port, proto))
    approvals = _str(spec, "approvalsMode").lower()
    if approvals not in APPROVAL_MODES:
        raise SpecError("approvalsMode must be empty, manual, smart or off")
    turn_host = _str(spec, "turnHost", default=DEFAULT_TURN_HOST) or DEFAULT_TURN_HOST
    try:
        ipaddress.ip_address(turn_host)
    except ValueError as exc:
        raise SpecError("turnHost must be an IP address") from exc
    turn_port = spec.get("turnPort", DEFAULT_TURN_PORT)
    if not isinstance(turn_port, int) or not 0 < turn_port < 65536:
        raise SpecError("turnPort must be an integer port")
    channel_env = spec.get("channelEnv") or {}
    if not isinstance(channel_env, dict):
        raise SpecError("channelEnv must be an object")
    unknown = sorted(set(channel_env) - set(CHANNEL_ENV_KEYS))
    if unknown:
        raise SpecError(f"channelEnv keys not allowed: {', '.join(unknown)}")
    channels = {key: _str(channel_env, key) for key in CHANNEL_ENV_KEYS if key in channel_env}
    return {
        "matter_id": matter_id, "instance_url": instance_url, "host_secret": host_secret,
        "agent_token": agent_token, "matter_home": matter_home, "provider": provider, "model": model,
        "model_base_url": model_base_url, "model_key": model_key, "typesafe_key": typesafe_key,
        "tailscale_key": tailscale_key, "hostname": hostname, "egress": egress, "rules": rules, "approvals": approvals,
        "turn_host": turn_host, "turn_port": turn_port, "channels": channels,
    }


def build_env(v: dict) -> str:
    """The /etc/litco-agent/env contents: the only place secrets are written."""
    lines = [
        "# litco-agent host environment. Written by cloud-init; root:root 0600.",
        "# Read by systemd for litco-agent.service (EnvironmentFile=). Do not source it in shells.",
        "# -- non-secret --",
        env_line("LITCO_MATTER_ID", v["matter_id"]),
        env_line("LITCO_INSTANCE_URL", v["instance_url"]),
        env_line("LITCO_MATTER_HOME", v["matter_home"]),
        env_line("LITCO_TURN_HOST", v["turn_host"]),
        env_line("LITCO_TURN_PORT", str(v["turn_port"])),
        env_line("LITCO_MODEL_PROVIDER", v["provider"]),
        env_line("LITCO_MODEL", v["model"]),
        env_line("LITCO_MODEL_BASE_URL", v["model_base_url"]),
        env_line("LITCO_MODEL_KEY_ENV", PROVIDER_KEY_ENV["custom"] if v["model_key"] and v["provider"] == "custom" else ""),
        env_line("LITCO_APPROVALS_MODE", v["approvals"]),
        env_line("LITCO_EGRESS_POLICY", v["egress"]),
        "# -- secret --",
        env_line("LITCO_HOST_SECRET", v["host_secret"]),
        env_line("LITCO_AGENT_TOKEN", v["agent_token"]),
    ]
    if v["model_key"]:
        lines.append(env_line(PROVIDER_KEY_ENV[v["provider"]], v["model_key"]))
    if v["typesafe_key"]:
        lines.append(env_line(TYPESAFE_KEY_ENV, v["typesafe_key"]))
    for key, value in v["channels"].items():
        lines.append(env_line(key, value))
    return "\n".join(lines) + "\n"


def build_matter_json(v: dict) -> str:
    """Non-secret description of this host, for operators and the control plane."""
    doc = {
        "schema": 1, "matterId": v["matter_id"], "instanceUrl": v["instance_url"],
        "hostname": v["hostname"], "matterHome": v["matter_home"], "modelProvider": v["provider"],
        "model": v["model"], "egressPolicy": v["egress"], "approvalsMode": v["approvals"],
        "turnPort": v["turn_port"], "tailscale": bool(v["tailscale_key"]),
        "channels": sorted(k.split("_", 1)[0].lower() for k in v["channels"] if k.endswith("_BOT_TOKEN")),
    }
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def apply_blocks(template: str, enabled: dict) -> str:
    """Keep or drop every "# {{NAME_BEGIN}}" ... "# {{NAME_END}}" block."""
    out, stack = [], []
    for line in template.split("\n"):
        marker = re.fullmatch(r"\s*# \{\{(\w+)_(BEGIN|END)\}\}\s*", line)
        if marker:
            name, edge = marker.groups()
            if name not in enabled:
                raise SpecError(f"template block {name} has no rule")
            if edge == "BEGIN":
                stack.append(name)
            elif not stack or stack.pop() != name:
                raise SpecError(f"template block {name} is not balanced")
            continue
        if all(enabled[name] for name in stack):
            out.append(line)
    if stack:
        raise SpecError(f"template block {stack[-1]} is not closed")
    return "\n".join(out)


def render(spec: dict, template: str | None = None) -> str:
    v = validate(spec)
    text = TEMPLATE.read_text(encoding="utf-8") if template is None else template
    text = apply_blocks(text, {"TAILSCALE": bool(v["tailscale_key"]), "EGRESS_ALLOWLIST": v["egress"] == "allowlist"})
    rules = "\n".join(f"  - ufw allow out to {cidr} port {port} proto {proto}" for cidr, port, proto in v["rules"])
    subst = {
        "HOSTNAME": v["hostname"],
        "MATTER_HOME": v["matter_home"],
        "ENV_B64": _b64(build_env(v)),
        "MATTER_JSON_B64": _b64(build_matter_json(v)),
        "TAILSCALE_AUTHKEY_B64": _b64(v["tailscale_key"]) if v["tailscale_key"] else "",
        "EGRESS_ALLOWLIST_RULES": rules,
    }
    unknown = sorted({m for m in _TOKEN.findall(text) if m not in subst})
    if unknown:
        raise SpecError(f"template uses unknown placeholders: {', '.join(unknown)}")
    text = _TOKEN.sub(lambda m: subst[m.group(1)], text)
    if not text.startswith("#cloud-config\n"):
        raise SpecError("rendered user-data must start with #cloud-config")
    if len(text.encode("utf-8")) > USER_DATA_LIMIT:
        raise SpecError("rendered user-data exceeds DigitalOcean's 64 KiB limit")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("spec", help="path to the JSON spec, or - for stdin")
    parser.add_argument("--template", type=Path, default=None, help="override cloud-init template")
    args = parser.parse_args(argv)
    try:
        raw = sys.stdin.read() if args.spec == "-" else Path(args.spec).read_text(encoding="utf-8")
        spec = json.loads(raw)
        template = args.template.read_text(encoding="utf-8") if args.template else None
        sys.stdout.write(render(spec, template))
    except (SpecError, json.JSONDecodeError, OSError) as exc:
        print(f"render-user-data: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
