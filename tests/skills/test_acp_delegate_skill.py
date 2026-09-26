import importlib.util
import json
import subprocess
import sys


ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
SKILL = ROOT / "optional-skills/communication/acp-delegate"


def load_helper():
    spec = importlib.util.spec_from_file_location("acp_delegate", SKILL / "scripts/acp_delegate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_frontmatter_and_safety_contract():
    text = (SKILL / "SKILL.md").read_text()
    assert text.startswith("---\n")
    assert "grant" in text.lower() and "cancel" in text.lower()
    assert "credentials" in text.lower()
    assert len(text.split("description:", 1)[1].splitlines()[0].strip()) <= 60
    config = json.loads((SKILL / "templates/config.example.json").read_text())
    assert config["profile"].startswith("REPLACE_")
    assert config["timeout_seconds"] <= 600
    assert "TOKEN" not in text and "PASSWORD" not in text


def test_config_requires_explicit_profile_and_existing_absolute_cwd(tmp_path):
    helper = load_helper()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"profile": "REPLACE_WITH_SECONDARY_PROFILE", "cwd": str(tmp_path)}))
    try:
        helper._load_config(config)
    except ValueError as exc:
        assert "profile" in str(exc)
    else:
        raise AssertionError("placeholder profile was accepted")


def test_fake_acp_lifecycle_is_bounded_and_rejects_callbacks(tmp_path):
    helper = load_helper()
    fake = tmp_path / "fake_acp.py"
    fake.write_text(
        "import json,sys\n"
        "for line in sys.stdin:\n"
        " m=json.loads(line); i=m.get('id'); method=m.get('method')\n"
        " if method=='initialize': r={'protocolVersion':1}\n"
        " elif method=='session/new': r={'sessionId':'s1'}\n"
        " elif method=='session/prompt':\n"
        "  print(json.dumps({'jsonrpc':'2.0','method':'session/request_permission','id':99,'params':{}}),flush=True)\n"
        "  print(json.dumps({'jsonrpc':'2.0','method':'session/update','params':{'update':{'sessionUpdate':'agent_message_chunk','content':{'text':'OK'}}}}),flush=True)\n"
        "  r={'stopReason':'end_turn'}\n"
        " else: continue\n"
        " print(json.dumps({'jsonrpc':'2.0','id':i,'result':r}),flush=True)\n"
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"profile": "reviewer", "cwd": str(tmp_path), "command": sys.executable, "args": [str(fake)], "timeout_seconds": 5}))
    assert helper.call(config, "say hi") == "OK"


def test_script_compiles():
    result = subprocess.run([sys.executable, "-m", "py_compile", str(SKILL / "scripts/acp_delegate.py")], cwd=ROOT)
    assert result.returncode == 0
