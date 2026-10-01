"""render-user-data.py: per-matter and machine cloud-init user-data."""

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
TYPESAFE_KEY = "ts-TYPESAFE-3333"
SECRETS = (HOST_SECRET, AGENT_TOKEN, MODEL_KEY, TS_KEY, SLACK_TOKEN, TYPESAFE_KEY)


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
    text = rud.render(spec(tailscaleAuthKey=TS_KEY, typesafeApiKey=TYPESAFE_KEY,
                           channelEnv={"SLACK_BOT_TOKEN": SLACK_TOKEN}))
    for secret in SECRETS:
        assert secret not in text, "secret rendered in clear"
    doc = parse(text)
    env_text = decoded(doc, "/etc/litco-agent/env")
    matter_text = decoded(doc, "/etc/litco-agent/matter.json")
    ts_text = decoded(doc, "/etc/litco-agent/tailscale-authkey")
    for secret in (HOST_SECRET, AGENT_TOKEN, MODEL_KEY, SLACK_TOKEN, TYPESAFE_KEY):
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


def test_typesafe_key_lands_in_its_own_variable_only_when_given():
    env = env_map(decoded(parse(rud.render(spec(typesafeApiKey=TYPESAFE_KEY))), "/etc/litco-agent/env"))
    assert env["TYPESAFE_API_KEY"] == TYPESAFE_KEY
    assert "TYPESAFE_API_KEY" not in env_map(decoded(parse(rud.render(spec())), "/etc/litco-agent/env"))


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


# ── per-matter output is frozen ─────────────────────────────────────────────

FIXTURES = HOST.parents[1] / "tests" / "host" / "fixtures"


@pytest.mark.parametrize("name", ["per_matter_full", "per_matter_minimal"])
def test_per_matter_render_is_byte_identical_to_before_machine_mode(name):
    """The four live droplets, every restore and LitKit's TypeScript port depend on this output."""
    fixture_spec = json.loads((FIXTURES / f"{name}.spec.json").read_text(encoding="utf-8"))
    expected = (FIXTURES / f"{name}.user-data.yaml").read_text(encoding="utf-8")
    assert rud.render(fixture_spec) == expected
    assert rud.render({**fixture_spec, "topology": "per_matter"}) == expected


# ── machine topology ────────────────────────────────────────────────────────

SUPERVISOR_SECRET = "sv_" + "Q7r8S9t0" * 5
MACHINE_SECRETS = (SUPERVISOR_SECRET, TS_KEY)


def machine_spec(**overrides):
    base = {
        "topology": "machine",
        "hostname": "litco-host-1a2b3c4d",
        "instanceUrl": "https://acme.litco.ai",
        "supervisorSecret": SUPERVISOR_SECRET,
        "tailscaleAuthKey": TS_KEY,
        "egressPolicy": "open",
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def test_machine_writes_supervisor_env_machine_json_and_tailscale_key():
    doc = parse(rud.render(machine_spec()))
    f = files(doc)
    assert set(f) == {"/etc/litco-supervisor.env", "/etc/litco-agent/machine.json",
                      "/etc/litco-agent/tailscale-authkey"}
    assert (f["/etc/litco-supervisor.env"]["owner"], f["/etc/litco-supervisor.env"]["permissions"]) == ("root:root", "0600")
    assert f["/etc/litco-agent/tailscale-authkey"]["permissions"] == "0600"
    assert f["/etc/litco-agent/machine.json"]["permissions"] == "0644"

    env = env_map(decoded(doc, "/etc/litco-supervisor.env"))
    assert env == {"LITCO_SUPERVISOR_SECRET": SUPERVISOR_SECRET}
    machine = json.loads(decoded(doc, "/etc/litco-agent/machine.json"))
    assert machine == {"schema": 1, "topology": "machine", "hostname": "litco-host-1a2b3c4d",
                       "instanceUrl": "https://acme.litco.ai"}
    assert decoded(doc, "/etc/litco-agent/tailscale-authkey") == TS_KEY
    assert doc["hostname"] == "litco-host-1a2b3c4d"


def test_supervisor_env_parses_as_the_supervisor_reads_it():
    from tests.deploy._supervisor_fake import sup
    tricky = 'q"uo\\te$HOME' + "x" * 30
    doc = parse(rud.render(machine_spec(supervisorSecret=tricky)))
    assert sup.parse_env_file(decoded(doc, "/etc/litco-supervisor.env")) == {"LITCO_SUPERVISOR_SECRET": tricky}


def test_machine_runcmd_order():
    runcmd = parse(rud.render(machine_spec()))["runcmd"]
    assert runcmd == [
        'hostnamectl set-hostname "litco-host-1a2b3c4d"',
        "chown root:root /etc/litco-supervisor.env",
        "chmod 600 /etc/litco-supervisor.env",
        "nft list table inet litco_host >/dev/null 2>&1 || nft -f /etc/nftables.conf",
        "nft list table inet litco_host",
        'tailscale up --ssh --hostname="litco-host-1a2b3c4d" --auth-key=file:/etc/litco-agent/tailscale-authkey',
        "shred -u /etc/litco-agent/tailscale-authkey",
        "systemctl daemon-reload",
        "systemctl restart litco-supervisor.service",
        "systemctl status --no-pager litco-supervisor.service || true",
    ]
    # No ufw (the image's nftables rule set is the firewall) and no per-matter unit.
    assert not any("ufw" in c or "litco-agent.service" in c for c in runcmd)


def test_machine_secrets_appear_only_inside_base64_blocks():
    text = rud.render(machine_spec())
    for secret in MACHINE_SECRETS:
        assert secret not in text
    doc = parse(text)
    env_text = decoded(doc, "/etc/litco-supervisor.env")
    machine_text = decoded(doc, "/etc/litco-agent/machine.json")
    assert SUPERVISOR_SECRET in env_text and TS_KEY not in env_text
    assert SUPERVISOR_SECRET not in machine_text and TS_KEY not in machine_text
    # Outside the write_files contents, nothing decodes to a secret either.
    contents = {entry["content"] for entry in doc["write_files"]}
    for line in text.splitlines():
        if any(c in line for c in contents):
            continue
        for secret in MACHINE_SECRETS:
            assert base64.b64encode(secret.encode()).decode() not in line


def test_machine_carries_no_matter_identity_or_matter_secret():
    text = rud.render(machine_spec())
    doc = parse(text)
    blob = text + "".join(decoded(doc, p) for p in files(doc))
    for marker in ("LITCO_MATTER_ID", "LITCO_HOST_SECRET", "LITCO_AGENT_TOKEN", "_API_KEY", "lkm_", "/etc/litco-agent/env"):
        assert marker not in blob


@pytest.mark.parametrize("overrides, message", [
    ({"hostname": None}, "hostname is required"),
    ({"hostname": "matter-1a2b3c4d"}, "hostname must look like litco-host-"),
    ({"hostname": "litco-host-"}, "hostname"),
    ({"hostname": "litco-host-UPPER"}, "hostname"),
    ({"hostname": "litco-host-a-"}, "hostname"),
    ({"hostname": "litco-host-" + "a" * 43}, "hostname"),
    ({"hostname": 'litco-host-a";reboot'}, "hostname"),
    ({"instanceUrl": None}, "instanceUrl is required"),
    ({"instanceUrl": "ftp://acme.litco.ai"}, "instanceUrl"),
    ({"supervisorSecret": None}, "supervisorSecret is required"),
    ({"supervisorSecret": "short"}, "supervisorSecret must be at least 32"),
    ({"supervisorSecret": SUPERVISOR_SECRET + "\nX=1"}, "control characters"),
    ({"tailscaleAuthKey": None}, "tailscaleAuthKey is required"),
    ({"tailscaleAuthKey": ""}, "tailscaleAuthKey is required"),
    ({"egressPolicy": None}, "egressPolicy is required"),
    ({"egressPolicy": "allowlist"}, "^egressPolicy allowlist is not supported on a machine host$"),
    ({"egressPolicy": "closed"}, "egressPolicy must be 'open'"),
    ({"matterId": "m-1"}, "keys not allowed in a machine spec: matterId"),
    ({"hostSecret": HOST_SECRET}, "keys not allowed in a machine spec: hostSecret"),
    ({"egressAllowlist": []}, "keys not allowed"),
    ({"topology": "firm"}, "topology must be"),
    ({"topology": 1}, "topology must be a string"),
])
def test_bad_machine_specs_are_refused(overrides, message):
    with pytest.raises(rud.SpecError, match=message) as info:
        rud.render(machine_spec(**overrides))
    for secret in MACHINE_SECRETS:
        assert secret not in str(info.value)


def test_machine_hostname_accepts_the_apps_form_and_the_rule_bounds():
    for good in ("litco-host-1a2b3c4d", "litco-host-a", "litco-host-a-b", "litco-host-" + "a" * 42):
        assert parse(rud.render(machine_spec(hostname=good)))["hostname"] == good


def test_user_data_cap_applies_to_both_topologies():
    pad = "#" + "x" * (64 * 1024) + "\n"
    machine_tmpl = rud.MACHINE_TEMPLATE.read_text(encoding="utf-8")
    with pytest.raises(rud.SpecError, match="64 KiB"):
        rud.render(machine_spec(), machine_tmpl + pad)
    with pytest.raises(rud.SpecError, match="64 KiB"):
        rud.render(spec(), rud.TEMPLATE.read_text(encoding="utf-8") + pad)
    assert len(rud.render(machine_spec()).encode()) < 4 * 1024


def test_machine_template_has_no_unknown_placeholders_or_blocks():
    with pytest.raises(rud.SpecError, match="unknown placeholders: MATTER_HOME"):
        rud.render(machine_spec(), "#cloud-config\n{{MATTER_HOME}}\n")


def test_cli_renders_a_machine_spec(tmp_path):
    script = str(HOST / "render-user-data.py")
    out = subprocess.run([sys.executable, script, "-"], input=json.dumps(machine_spec()), capture_output=True,
                         text=True, check=True)
    assert out.stdout == rud.render(machine_spec())
    bad = subprocess.run([sys.executable, script, "-"], input=json.dumps(machine_spec(egressPolicy="allowlist")),
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "egressPolicy allowlist is not supported on a machine host" in bad.stderr
    assert SUPERVISOR_SECRET not in bad.stderr and TS_KEY not in bad.stderr
