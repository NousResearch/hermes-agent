import pytest
import os
import yaml
import sqlite3
from pathlib import Path
from hermes_cli.config import load_config
from hermes_cli.profile_activity_ledger import append_event

# Load the plugin from this checkout without a bare ``__init__`` import. The
# latter collides with ``tests/__init__.py`` during full-suite collection and
# the old absolute path silently tested a different checkout.
import importlib.util
import sys

PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "denji-policy-enforcer"
_PLUGIN_SPEC = importlib.util.spec_from_file_location(
    "denji_policy_enforcer_under_test", PLUGIN_DIR / "__init__.py"
)
assert _PLUGIN_SPEC and _PLUGIN_SPEC.loader
_PLUGIN = importlib.util.module_from_spec(_PLUGIN_SPEC)
sys.modules[_PLUGIN_SPEC.name] = _PLUGIN
_PLUGIN_SPEC.loader.exec_module(_PLUGIN)
pre_tool_call_handler = _PLUGIN.pre_tool_call_handler
_engine = _PLUGIN._engine

@pytest.fixture
def mock_hermes_home(tmp_path):
    home = tmp_path / "hermes_home"
    home.mkdir()
    # Set HERMES_HOME for the duration of the test
    old_home = os.getenv("HERMES_HOME")
    os.environ["HERMES_HOME"] = str(home)
    yield home
    if old_home:
        os.environ["HERMES_HOME"] = old_home
    else:
        del os.environ["HERMES_HOME"]

@pytest.fixture
def setup_policies(mock_hermes_home):
    policy_file = mock_hermes_home / "policy.yaml"
    policies = {
        "policies": {
            "write_file": [
                {"forbidden_pattern": "SECRET_KEY", "message": "No secrets!"},
                {"forbidden_prefix": "/etc/", "message": "No system files!"}
            ],
            "terminal": {
                "forbidden_pattern": "rm -rf /",
                "message": "No root deletion!"
            }
        }
    }
    policy_file.write_text(yaml.dump(policies))
    # Force the engine to reload
    _engine._load_policies()
    return policies

def test_policy_allow(setup_policies):
    res = pre_tool_call_handler(
        tool_name="write_file",
        args={"path": "/tmp/test.txt", "content": "Hello world"},
        profile_name="test-prof"
    )
    assert res["action"] == "allow"

def test_policy_block_pattern(setup_policies):
    res = pre_tool_call_handler(
        tool_name="write_file",
        args={"path": "/tmp/test.txt", "content": "my SECRET_KEY is 123"},
        profile_name="test-prof"
    )
    assert res["action"] == "block"
    assert "No secrets!" in res["message"]

def test_policy_block_prefix(setup_policies):
    res = pre_tool_call_handler(
        tool_name="write_file",
        args={"path": "/etc/passwd", "content": "hack"},
        profile_name="test-prof"
    )
    assert res["action"] == "block"
    assert "No system files!" in res["message"]

def test_policy_block_terminal(setup_policies):
    res = pre_tool_call_handler(
        tool_name="terminal",
        args={"command": "rm -rf / --no-preserve-root"},
        profile_name="test-prof"
    )
    assert res["action"] == "block"
    assert "No root deletion!" in res["message"]

def test_policy_unknown_tool(setup_policies):
    res = pre_tool_call_handler(
        tool_name="my_custom_tool",
        args={"foo": "bar"},
        profile_name="test-prof"
    )
    assert res["action"] == "allow"

def test_audit_logging(mock_hermes_home, setup_policies):
    # Ensure ledger is enabled for this test
    # We can monkeypatch record_event_if_enabled or just check the DB
    # Since record_event_if_enabled checks config.yaml, we create one.
    config_file = mock_hermes_home / "config.yaml"
    config_file.write_text(yaml.dump({"governance": {"profile_activity_ledger": {"enabled": True}}}))
    
    # Need to refresh config for the ledger check
    # The ledger module reads config via load_config() which usually caches.
    # In a real test we'd handle the cache, here we just trigger a call.
    
    pre_tool_call_handler(
        tool_name="write_file",
        args={"path": "/etc/passwd", "content": "hack"},
        profile_name="audit-test"
    )
    
    ledger_path = mock_hermes_home / "governance" / "profile-activity-ledger.sqlite"
    assert ledger_path.exists()
    
    conn = sqlite3.connect(ledger_path)
    cur = conn.cursor()
    cur.execute("SELECT event_type, object_id FROM activity_events WHERE event_type='tool_policy_check'")
    row = cur.fetchone()
    assert row is not None
    assert row[0] == "tool_policy_check"
    assert row[1] == "write_file"
    conn.close()
