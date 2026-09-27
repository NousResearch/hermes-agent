"""render-user-data.py: per-matter cloud-init user-data."""

from __future__ import annotations

import base64
import json
import subprocess
import sys

import pytest

from hermes_yaml import safe_load
from tests.host._load import HOST, load_script

rud = load_script("litco_render_user_data", "render-user-data.py")

HOST_SECRET = "hs_" + "A1b2C3d4" * 5
AGENT_TOKEN = "lkm_matterpinned_TOKEN_9f8e7d"
MODEL_KEY = "sk-ant-api03-MODELKEY-0000"
TS_KEY = "tskey-auth-kTAILSCALE-1111"
SLACK_TOKEN = "xoxb-SLACK-2222"
SECRETS = (HOST_SECRET, AGENT_TOKEN, MODEL_KEY, TS_KEY, SLACK_TOKEN)


def spec(**overrides):
    base = {
        "matterId": "4F1C9A2E-77aa-4b1b-9c3d-0e5f6a7b8c9d",
        "instanceUrl": "https://acme.litco.ai",
        "hostSecret": HOST_SECRET,
        "agentToken": AGENT_TOKEN,
        "matterHome": "/home/hermes/matter",
        "modelProvider": "anthropic",
        "model": "anthropic/claude-opus-4.6",
        "modelKey": MODEL_KEY,
        "egressPolicy": "open",
    }
    base.update(overrides)
    return base


def parse(text: str) -> dict:
    assert text.startswith("#cloud-config\n")
    return safe_load(text)


def files(doc: dict) -> dict:
    return {f["path"]: f for f in doc["write_files"]}


def decoded(doc: dict, path: str) -> str:
    entry = files(doc)[path]
    assert entry["encoding"] == "b64"
    return base64.b64decode(entry["content"]).decode()


def env_map(env_text: str) -> dict:
    out = {}
    for line in env_text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        assert value.startswith('"') and value.endswith('"'), line
        out[key] = value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return out


def test_fields_land_in_env_matter_json_and_runcmd():
    doc = parse(rud.render(spec(approvalsMode="manual", tailscaleAuthKey=TS_KEY)))
    env = env_map(decoded(doc, "/etc/litco-agent/env"))
    assert env["LITCO_MATTER_ID"] == "4F1C9A2E-77aa-4b1b-9c3d-0e5f6a7b8c9d"
    assert env["LITCO_INSTANCE_URL"] == "https://acme.litco.ai"
    assert env["LITCO_MATTER_HOME"] == "/home/hermes/matter"
    assert env["LITCO_MODEL_PROVIDER"] == "anthropic"
    assert env["LITCO_MODEL"] == "anthropic/claude-opus-4.6"
    assert env["LITCO_APPROVALS_MODE"] == "manual"
    assert env["LITCO_TURN_HOST"] == "0.0.0.0" and env["LITCO_TURN_PORT"] == "8765"
    assert env["LITCO_HOST_SECRET"] == HOST_SECRET
    assert env["LITCO_AGENT_TOKEN"] == AGENT_TOKEN
    assert env["ANTHROPIC_API_KEY"] == MODEL_KEY

    matter = json.loads(decoded(doc, "/etc/litco-agent/matter.json"))
    assert matter["matterId"] == env["LITCO_MATTER_ID"]
    assert matter["hostname"] == "matter-4f1c9a2e"
    assert matter["tailscale"] is True and matter["egressPolicy"] == "open"

    f = files(doc)
    assert f["/etc/litco-agent/env"]["permissions"] == "0600"
    assert f["/etc/litco-agent/env"]["owner"] == "root:root"
    runcmd = doc["runcmd"]
    assert 'install -d -o hermes -g hermes -m 0750 "/home/hermes/matter"' in runcmd
    assert runcmd[-3:-1] == ["systemctl daemon-reload", "systemctl restart litco-agent.service"]
    assert "systemctl start litco-agent.service" not in runcmd  # a start would be a silent no-op


def test_secrets_appear_only_in_env_file_and_tailscale_key_file():
    text = rud.render(spec(tailscaleAuthKey=TS_KEY, channelEnv={"SLACK_BOT_TOKEN": SLACK_TOKEN}))
    for secret in SECRETS:
        assert secret not in text, "secret rendered in clear"
    doc = parse(text)
    env_text = decoded(doc, "/etc/litco-agent/env")
    matter_text = decoded(doc, "/etc/litco-agent/matter.json")
    ts_text = decoded(doc, "/etc/litco-agent/tailscale-authkey")
    for secret in (HOST_SECRET, AGENT_TOKEN, MODEL_KEY, SLACK_TOKEN):
        assert secret in env_text
        assert secret not in matter_text and secret not in ts_text
    assert ts_text == TS_KEY and TS_KEY not in env_text and TS_KEY not in matter_text
    assert json.loads(matter_text)["channels"] == ["slack"]
    assert files(doc)["/etc/litco-agent/tailscale-authkey"]["permissions"] == "0600"


def test_tailscale_block_only_with_a_key():
    without = parse(rud.render(spec()))
    assert "/etc/litco-agent/tailscale-authkey" not in files(without)
    assert not any("tailscale" in cmd for cmd in without["runcmd"] if "tailscale0" not in cmd)
    with_key = parse(rud.render(spec(tailscaleAuthKey=TS_KEY)))
    joined = [c for c in with_key["runcmd"] if c.startswith("tailscale up")]
    assert joined == ['tailscale up --ssh --hostname="matter-4f1c9a2e" '
                      "--auth-key=file:/etc/litco-agent/tailscale-authkey"]
    assert "shred -u /etc/litco-agent/tailscale-authkey" in with_key["runcmd"]


def test_firewall_denies_inbound_except_tailnet_and_open_egress_has_no_allowlist():
    runcmd = parse(rud.render(spec()))["runcmd"]
    assert "ufw default deny incoming" in runcmd
    assert "ufw allow in on tailscale0" in runcmd
    assert "ufw default allow outgoing" in runcmd
    assert "ufw default deny outgoing" not in runcmd
    assert not any(c.startswith("ufw allow out") for c in runcmd)
    assert runcmd.index("ufw allow in on tailscale0") < runcmd.index("ufw --force enable")


def test_allowlist_block_appears_only_when_requested():
    rules = [{"cidr": "203.0.113.7"}, {"cidr": "198.51.100.0/24", "port": 8443, "proto": "tcp"}]
    runcmd = parse(rud.render(spec(egressPolicy="allowlist", egressAllowlist=rules)))["runcmd"]
    assert "ufw allow out to 203.0.113.7/32 port 443 proto tcp" in runcmd
    assert "ufw allow out to 198.51.100.0/24 port 8443 proto tcp" in runcmd
    assert "ufw allow out 53" in runcmd and "ufw allow out on tailscale0" in runcmd
    # deny-outgoing lands after every allow and before enable
    deny = runcmd.index("ufw default deny outgoing")
    assert all(runcmd.index(c) < deny for c in runcmd if c.startswith("ufw allow out"))
    assert deny < runcmd.index("ufw --force enable")
    # allowlist with no entries still renders the base block
    base = parse(rud.render(spec(egressPolicy="allowlist")))["runcmd"]
    assert "ufw default deny outgoing" in base


def test_hostname_derived_or_explicit():
    assert parse(rud.render(spec()))["hostname"] == "matter-4f1c9a2e"
    doc = parse(rud.render(spec(hostname="matter-acme01")))
    assert doc["hostname"] == "matter-acme01"
    assert 'hostnamectl set-hostname "matter-acme01"' in doc["runcmd"]
    for bad in ("acme01", "matter-UPPER", "matter-", "matter-a;reboot"):
        with pytest.raises(rud.SpecError):
            rud.render(spec(hostname=bad))


@pytest.mark.parametrize("overrides, message", [
    ({"hostSecret": "short"}, "hostSecret"),
    ({"agentToken": "abc"}, "agentToken"),
    ({"matterId": "../etc"}, "matterId"),
    ({"instanceUrl": "ftp://x"}, "instanceUrl"),
    ({"modelProvider": "mystery", "modelKey": "k"}, "no known key variable"),
    ({"egressPolicy": "closed"}, "egressPolicy"),
    ({"egressAllowlist": [{"cidr": "10.0.0.0/8"}]}, "only meaningful"),
    ({"egressPolicy": "allowlist", "egressAllowlist": [{"cidr": "nope"}]}, "cidr"),
    ({"approvalsMode": "yolo"}, "approvalsMode"),
    ({"channelEnv": {"AWS_SECRET_ACCESS_KEY": "x"}}, "channelEnv"),
    ({"hostSecret": HOST_SECRET + "\nLITCO_MATTER_ID=other"}, "control characters"),
    ({"matterHome": "relative/path"}, "matterHome"),
    ({"model": 'x" injected'}, "model"),
])
def test_bad_specs_are_refused(overrides, message):
    with pytest.raises(rud.SpecError, match=message):
        rud.render(spec(**overrides))


def test_env_values_are_escaped_for_systemd():
    tricky = 'q"uo\\te$HOME' + "x" * 30
    doc = parse(rud.render(spec(hostSecret=tricky)))
    env_text = decoded(doc, "/etc/litco-agent/env")
    assert 'LITCO_HOST_SECRET="q\\"uo\\\\te$HOME' in env_text
    assert env_map(env_text)["LITCO_HOST_SECRET"] == tricky


def test_custom_provider_key_goes_to_litco_model_api_key():
    doc = parse(rud.render(spec(modelProvider="custom", modelBaseUrl="https://proxy.litco.ai/v1")))
    env = env_map(decoded(doc, "/etc/litco-agent/env"))
    assert env["LITCO_MODEL_API_KEY"] == MODEL_KEY and "ANTHROPIC_API_KEY" not in env
    assert env["LITCO_MODEL_BASE_URL"] == "https://proxy.litco.ai/v1"
    assert env["LITCO_MODEL_KEY_ENV"] == "LITCO_MODEL_API_KEY"  # the name, for model.key_env
    keyless = env_map(decoded(parse(rud.render(spec(modelProvider="custom", modelKey=None))),
                              "/etc/litco-agent/env"))
    assert keyless["LITCO_MODEL_KEY_ENV"] == "" and "LITCO_MODEL_API_KEY" not in keyless


def test_no_model_key_means_no_key_variable():
    env = env_map(decoded(parse(rud.render(spec(modelKey=None))), "/etc/litco-agent/env"))
    assert not any(k.endswith("_API_KEY") for k in env) and env["LITCO_MODEL_KEY_ENV"] == ""


def test_template_blocks_must_balance():
    with pytest.raises(rud.SpecError, match="not closed"):
        rud.apply_blocks("# {{TAILSCALE_BEGIN}}\nx", {"TAILSCALE": True})
    with pytest.raises(rud.SpecError, match="no rule"):
        rud.apply_blocks("# {{OTHER_BEGIN}}\n# {{OTHER_END}}", {})


def test_cli_reads_spec_file_and_stdin(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec()))
    script = str(HOST / "render-user-data.py")
    out = subprocess.run([sys.executable, script, str(path)], capture_output=True, text=True, check=True)
    assert out.stdout.startswith("#cloud-config")
    out2 = subprocess.run([sys.executable, script, "-"], input=json.dumps(spec()), capture_output=True,
                          text=True, check=True)
    assert out2.stdout == out.stdout
    bad = subprocess.run([sys.executable, script, "-"], input=json.dumps(spec(hostSecret="x")),
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "hostSecret" in bad.stderr and HOST_SECRET not in bad.stderr
