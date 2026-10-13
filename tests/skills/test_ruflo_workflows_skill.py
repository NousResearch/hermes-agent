"""Tests for the ruflo-workflows skill.

Structural + internal-consistency checks only (stdlib + pytest, no network).
The skill's runtime claims were validated live against ruflo v3.34.0 MCP
(swarm_init/status/health/shutdown lifecycle) and the TDAI memory gateway
(L0/L1 search). These tests guard the documented contract against drift.
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "skills" / "ruflo-workflows"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPTS = {
    "swarm": SKILL_DIR / "scripts" / "ruflo-swarm.sh",
    "memory": SKILL_DIR / "scripts" / "ruflo-memory.sh",
}
HEALTHCHECK = Path.home() / ".hermes" / "scripts" / "ruflo-healthcheck.sh"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_skill_file_exists():
    assert SKILL_MD.is_file(), f"missing {SKILL_MD}"


def test_frontmatter_present(skill_text: str):
    assert skill_text.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
    assert skill_text.count("---") >= 2, "frontmatter must be delimited by two '---'"


def test_required_sections_present(skill_text: str):
    for heading in (
        "## When to Use",
        "## Prerequisites",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert heading in skill_text, f"missing section: {heading}"


def test_swarm_init_schema_documented_correctly(skill_text: str):
    """Regression guard: v3.34.0 schema is (topology, maxAgents, strategy, config),
    NOT the old invented (goal, agentSpec, topology). The live smoke test proved
    the real shape; the doc must not drift back."""
    assert "ruflo__swarm_init(topology, maxAgents, strategy, config)" in skill_text
    assert "agentSpec" not in skill_text, (
        "agentSpec was removed from swarm_init in v3.34.0 — stale doc reference"
    )


def test_swarm_lifecycle_tools_documented(skill_text: str):
    for tool in (
        "ruflo__swarm_status",
        "ruflo__swarm_health",
        "ruflo__swarm_shutdown",
    ):
        assert tool in skill_text, f"missing lifecycle tool: {tool}"


@pytest.mark.parametrize("name", ["swarm", "memory"])
def test_scripts_executable_and_syntax_ok(name):
    script = SCRIPTS[name]
    assert script.is_file(), f"missing {script}"
    assert shutil.which("bash") is not None
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"{script} syntax error:\n{result.stderr}"


def test_swarm_script_mode_a_payload_matches_real_schema():
    """Mode A JSON payload must match the v3.34.0 swarm_init schema."""
    result = subprocess.run(
        ["bash", str(SCRIPTS["swarm"]), "test goal", "coding", "a"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert '"topology": "hierarchical"' in result.stdout
    assert '"maxAgents": 5' in result.stdout
    assert '"strategy": "specialized"' in result.stdout
    assert '"config"' in result.stdout and '"goal"' in result.stdout
    assert "agentSpec" not in result.stdout, "Mode A payload must not use stale agentSpec"


def test_swarm_script_mode_b_payload():
    """Mode B fallback must produce a delegate_task payload."""
    result = subprocess.run(
        ["bash", str(SCRIPTS["swarm"]), "test goal", "coding", "b"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "delegate_task" in result.stdout


def test_swarm_script_rejects_unknown_type():
    result = subprocess.run(
        ["bash", str(SCRIPTS["swarm"]), "goal", "bogus", "b"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 1
    assert "Unknown swarm type" in result.stderr


def test_memory_script_usage_error_without_network():
    """No-args invocation must exit 1 with usage on stderr, before any network."""
    result = subprocess.run(
        ["bash", str(SCRIPTS["memory"])], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 1
    assert "Usage:" in result.stderr


def test_memory_script_health_requires_no_auth_key():
    """health hits only /health (auth-free); other commands need the key file.
    With no key file, store/search/recent must fail fast with a clear error."""
    key_file = Path.home() / ".memory-tencentdb" / ".gateway-key"
    if not key_file.is_file():
        result = subprocess.run(
            ["bash", str(SCRIPTS["memory"]), "store", "x"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1
        assert "gateway key" in result.stderr


def test_healthcheck_exists_and_syntax_ok():
    if not HEALTHCHECK.is_file():
        pytest.skip("healthcheck not present in this environment")
    result = subprocess.run(
        ["bash", "-n", str(HEALTHCHECK)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"syntax error:\n{result.stderr}"


RUFLO_MCP_WRAPPER = Path.home() / ".hermes" / "scripts" / "ruflo-mcp.sh"


def test_ruflo_mcp_wrapper_exists_and_syntax_ok():
    if not RUFLO_MCP_WRAPPER.is_file():
        pytest.skip("ruflo mcp wrapper not present in this environment")
    result = subprocess.run(
        ["bash", "-n", str(RUFLO_MCP_WRAPPER)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"syntax error:\n{result.stderr}"


def test_ruflo_mcp_wrapper_fails_fast_on_expired_token():
    """Expired OAuth token must fail fast with a clear message, never start MCP."""
    if not RUFLO_MCP_WRAPPER.is_file():
        pytest.skip("ruflo mcp wrapper not present in this environment")
    import shutil
    import tempfile

    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, ".claude"), exist_ok=True)
        with open(os.path.join(tmp, ".claude", ".credentials.json"), "w") as f:
            json.dump({"claudeAiOauth": {"accessToken": "sk-ant-oat01-fake", "expiresAt": 1}}, f)
        result = subprocess.run(
            ["bash", "-c",
             f"HOME={tmp} RUFLO_MCP_SKIP_PROXY_CHECK=1 bash {RUFLO_MCP_WRAPPER} 2>&1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0
        assert "EXPIRED" in result.stdout or "EXPIRED" in result.stderr
    finally:
        shutil.rmtree(tmp)


def test_ruflo_mcp_wrapper_does_not_leak_token_to_config():
    """MCP server config must NOT contain raw OAuth tokens — only the wrapper path."""
    import yaml

    for cfg_path in [
        Path.home() / ".hermes" / "config.yaml",
        *(Path.home() / ".hermes" / "profiles").glob("*/config.yaml"),
    ]:
        if not cfg_path.is_file():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        ruflo = (cfg.get("mcp_servers") or {}).get("ruflo") or {}
        raw = json.dumps(ruflo)
        assert "sk-ant" not in raw, f"raw OAuth token leaked in {cfg_path}"
        cmd = ruflo.get("command", "")
        if cmd:
            assert cmd.endswith("ruflo-mcp.sh"), (
                f"{cfg_path}: ruflo must launch via the ruflo mcp wrapper, got {cmd}"
            )


PATCH_SCRIPT = Path.home() / ".hermes" / "scripts" / "ruflo-patch-provider.sh"
RUFLO_EXEC_CORE = (
    Path.home()
    / ".local/lib/node_modules/ruflo/node_modules/@claude-flow/cli/dist/src/mcp-tools/agent-execute-core.js"
)


def test_patch_script_exists_and_syntax_ok():
    if not PATCH_SCRIPT.is_file():
        pytest.skip("patch script not present in this environment")
    result = subprocess.run(
        ["bash", "-n", str(PATCH_SCRIPT)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, f"syntax error:\n{result.stderr}"


def test_patch_script_is_idempotent_and_self_heals():
    """Patch script must: no-op when applied (exit 0), re-apply after a simulated
    npm wipe (stock source), and stay idempotent after healing."""
    if not PATCH_SCRIPT.is_file() or not RUFLO_EXEC_CORE.is_file():
        pytest.skip("patch script or ruflo install not present")

    original = RUFLO_EXEC_CORE.read_text()
    patched_marker = "ANTHROPIC_BASE_URL ||"
    stock_marker = "https://api.anthropic.com/v1/messages"

    try:
        # 1. Already-applied → exit 0, no mutation
        r1 = subprocess.run(
            ["bash", str(PATCH_SCRIPT)], capture_output=True, text=True, timeout=30
        )
        assert r1.returncode == 0
        assert RUFLO_EXEC_CORE.read_text() == original

        # 2. Simulate npm wipe: restore stock hardcoded URL
        if patched_marker in original and stock_marker not in original:
            stock = original.replace(
                "const baseUrl = (process.env.ANTHROPIC_BASE_URL || 'https://api.anthropic.com').replace(/\\/+$/, '');\n"
                "        const authToken = process.env.ANTHROPIC_AUTH_TOKEN || '';\n"
                "        const res = await fetch(`${baseUrl}/v1/messages`, {\n"
                "            method: 'POST',\n"
                "            headers: {\n"
                "                ...(authToken\n"
                "                    ? { 'Authorization': `Bearer ${authToken}` }\n"
                "                    : { 'x-api-key': anthropicKey }),\n"
                "                'anthropic-version': '2023-06-01',\n"
                "                'content-type': 'application/json',\n"
                "            },",
                "        const res = await fetch('https://api.anthropic.com/v1/messages', {\n"
                "            method: 'POST',\n"
                "            headers: {\n"
                "                'x-api-key': anthropicKey,\n"
                "                'anthropic-version': '2023-06-01',\n"
                "                'content-type': 'application/json',\n"
                "            },",
            )
            RUFLO_EXEC_CORE.write_text(stock)
            assert stock_marker in RUFLO_EXEC_CORE.read_text(), "simulated wipe failed"

        # 3. Re-apply → exit 0, patched marker back
        r2 = subprocess.run(
            ["bash", str(PATCH_SCRIPT)], capture_output=True, text=True, timeout=30
        )
        assert r2.returncode == 0
        healed = RUFLO_EXEC_CORE.read_text()
        assert patched_marker in healed and stock_marker not in healed

        # 4. Idempotent after heal
        r3 = subprocess.run(
            ["bash", str(PATCH_SCRIPT)], capture_output=True, text=True, timeout=30
        )
        assert r3.returncode == 0
        assert RUFLO_EXEC_CORE.read_text() == healed
    finally:
        RUFLO_EXEC_CORE.write_text(original)


def test_skill_references_existing_scripts(skill_text: str):
    """Every script path referenced in SKILL.md must exist."""
    for ref in re.findall(r"ruflo-(?:swarm|memory)\.sh", skill_text):
        assert (SKILL_DIR / "scripts" / ref).is_file(), f"SKILL.md references missing {ref}"
