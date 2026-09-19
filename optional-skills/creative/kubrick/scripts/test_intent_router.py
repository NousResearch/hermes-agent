# scripts/test_intent_router.py
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import intent_router as ir
from manifest_contract import load_manifest

PY = sys.executable

EXPECTED_INTENTS = {
    "compile", "retrieve", "ledger", "design", "storyboard", "adapt",
    "visual", "learn", "check", "operate", "mcp", "bundle",
}

# From design spec — must stay complete
EXPECTED_LEGACY = {
    "compile", "retrieve", "ledger", "design-build",
    "storyboard-propagate", "storyboard-compare",
    "adapter-build", "adapt-provider", "adapt-grok", "adapt-flux",
    "adapt-sd3", "adapt-midjourney",
    "visual-normalize", "visual-compare", "visual-correct",
    "correction-govern", "closed-loop-qa",
    "outcome-record", "evolution-propose", "forge-signals",
    "validate-skill", "validate-corpus", "coverage",
    "artifact-validate", "repeatability", "eval",
    "operator", "mcp-server", "grok-review-bundle",
}


def test_intent_count_and_names():
    assert set(ir.INTENT_REGISTRY) == EXPECTED_INTENTS


def test_router_is_derived_from_canonical_manifest():
    manifest = load_manifest()
    assert set(ir.INTENT_REGISTRY) == set(manifest["intents"])
    assert set(ir.ALIAS_TABLE) == set(manifest["legacy_aliases"])
    assert ir.RECIPES == manifest["recipes"]
    for name, spec in ir.INTENT_REGISTRY.items():
        declared = manifest["intents"][name]
        assert spec.description == declared["description"]
        assert spec.actions == declared["actions"]
        assert spec.default_action == declared["default_action"]


def test_every_legacy_maps_exactly_once():
    assert set(ir.ALIAS_TABLE) == EXPECTED_LEGACY
    for name, alias in ir.ALIAS_TABLE.items():
        assert alias.intent in ir.INTENT_REGISTRY, name
        actions = ir.INTENT_REGISTRY[alias.intent].actions
        assert alias.action in actions or alias.passthrough_action, name


def test_resolve_do_adapt_provider():
    call = ir.resolve([
        "do", "adapt", "--action", "provider",
        "--provider", "flux", "--packet", "p.yaml", "--output", "o.yaml",
    ])
    assert call.intent == "adapt"
    assert call.action == "provider"
    assert call.script.name == "adapt_provider.py"
    assert call.argv == ["--provider", "flux", "--packet", "p.yaml", "--output", "o.yaml"]
    assert call.legacy_name is None


def test_resolve_alias_adapt_flux():
    call = ir.resolve(["adapt-flux", "--packet", "p.yaml", "--output", "o.yaml"])
    assert call.intent == "adapt"
    assert call.action == "provider"
    assert call.script.name == "adapt_provider.py"
    assert "--provider" in call.argv and "flux" in call.argv
    assert call.legacy_name == "adapt-flux"


def test_resolve_unknown_intent():
    try:
        ir.resolve(["do", "nope"])
        raise AssertionError("expected RouterError")
    except ir.RouterError as e:
        assert e.exit_code == 2
        assert "compile" in e.message or "valid" in e.message.lower()


def test_ledger_passthrough_action():
    call = ir.resolve(["ledger", "audit", "--ledger", "L.yaml"])
    assert call.intent == "ledger"
    assert call.action == "audit"
    assert call.script.name == "symbolic_ledger.py"
    assert call.argv[0] == "audit"


def test_top_level_help_lists_intents_not_all_aliases():
    text = ir.format_top_level_help()
    assert "adapt" in text and "visual" in text and "learn" in text
    assert "adapt-flux" not in text  # aliases not first-class
    assert "do <intent>" in text or "kubrick do" in text


def test_recipe_storyboard_example():
    argv = ir.resolve_recipe("storyboard-example")
    call = ir.resolve(argv)
    assert call.intent == "compile"
    assert call.script.name == "kubrick_compile.py"


def test_alias_do_parity_matrix():
    cases = [
        (["validate-skill"], ["do", "check", "--action", "skill"]),
        (["closed-loop-qa", "--expected", "e", "--observation-input", "o",
          "--source-graph-id", "g", "--frame-id", "f1", "--out", "out/x"],
         ["do", "visual", "--action", "closed-loop", "--expected", "e",
          "--observation-input", "o", "--source-graph-id", "g",
          "--frame-id", "f1", "--out", "out/x"]),
        (["forge-signals", "--project-id", "p", "--input", "i.yaml", "--output", "o.yaml"],
         ["do", "learn", "--action", "forge-signals", "--project-id", "p",
          "--input", "i.yaml", "--output", "o.yaml"]),
    ]
    for legacy, modern in cases:
        a, b = ir.resolve(legacy), ir.resolve(modern)
        assert a.script == b.script, (legacy, modern, a.script, b.script)
        assert a.argv == b.argv, (legacy, modern, a.argv, b.argv)


def test_mcp_tools_list_includes_kubrick_do():
    """tools/list via mcp-server alias and direct script must expose kubrick_do."""
    req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n"
    for argv in (
        [PY, str(ROOT / "scripts/kubrick.py"), "mcp-server"],
        [PY, str(ROOT / "scripts/mcp_kubrick_server.py")],
    ):
        proc = subprocess.run(
            argv,
            input=req,
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        assert proc.returncode == 0, (argv, proc.stderr, proc.stdout)
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
        assert lines, (argv, proc.stdout)
        listed = json.loads(lines[0])
        names = {t["name"] for t in listed["result"]["tools"]}
        assert "kubrick_do" in names, (argv, names)
        assert names == {"kubrick_do"}, (argv, names)
        schema = listed["result"]["tools"][0]["inputSchema"]
        assert "intent" in schema.get("properties", {})
        assert "intent" in schema.get("required", [])


def test_mcp_kubrick_do_call_unknown_intent_fails_closed():
    """tools/call kubrick_do with bad intent returns isError (exit 2 from router)."""
    req = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "kubrick_do",
                "arguments": {"intent": "nope-not-an-intent"},
            },
        }
    ) + "\n"
    proc = subprocess.run(
        [PY, str(ROOT / "scripts/mcp_kubrick_server.py")],
        input=req,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0
    result = json.loads(proc.stdout.strip().splitlines()[0])["result"]
    assert result.get("isError") is True
    sc = result.get("structuredContent") or {}
    assert sc.get("fail_closed") is True
    assert sc.get("exit_code") == 2


def main():
    test_intent_count_and_names()
    test_router_is_derived_from_canonical_manifest()
    test_every_legacy_maps_exactly_once()
    test_resolve_do_adapt_provider()
    test_resolve_alias_adapt_flux()
    test_resolve_unknown_intent()
    test_ledger_passthrough_action()
    test_top_level_help_lists_intents_not_all_aliases()
    test_recipe_storyboard_example()
    test_alias_do_parity_matrix()
    test_mcp_tools_list_includes_kubrick_do()
    test_mcp_kubrick_do_call_unknown_intent_fails_closed()
    print(
        "intent_router unit: registry + resolve + help/recipe + alias/do parity "
        "+ mcp kubrick_do PASS"
    )


if __name__ == "__main__":
    main()
