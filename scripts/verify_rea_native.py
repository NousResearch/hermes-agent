"""Real native MCP smoke; in-memory candidate config only, no profile install."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    # Isolate all Hermes runtime caches before importing any Hermes modules.
    scratch = Path(os.environ["TMPDIR"])
    with tempfile.TemporaryDirectory(prefix="rea-native-", dir=scratch) as directory:
        base = Path(directory)
        os.environ["HERMES_HOME"] = str(base / "hermes")
        os.environ["HERMES_OPTIONAL_MCPS"] = str(ROOT / "optional-mcps")
        from hermes_cli import mcp_catalog
        from tools.mcp_tool_discovery import register_mcp_servers
        from tools.mcp_tool_lifecycle import shutdown_mcp_servers
        from tools.registry import registry
        rea = Path("/home/hermes/bot-setup/rea")
        entry = mcp_catalog.get_entry("rea")
        assert entry is not None
        cfg = mcp_catalog.card_install_config(entry)
        workspace, temp = base / "workspace", base / "temp"
        workspace.mkdir(); temp.mkdir()
        values = {
            "REA_NODE": "/home/hermes/.hermes/tools/node-26.7.0-linux-x64/bin/node",
            "REA_SCRIPT": str(rea / "node_modules/rea-agents/scripts/rea.mjs"),
            "REA_WORKSPACE": str(workspace), "REA_TMPDIR": str(temp),
            "GHIDRA_INSTALL_DIR": str(rea / "dependencies/ghidra_12.1.4_PUBLIC"),
            "JAVA_HOME": str(rea / "dependencies/jdk-21.0.12.1+1"),
        }
        assert json.loads((rea / "node_modules/rea-agents/package.json").read_text())["version"] == "6.3.0"
        for spec in entry.auth.env:
            assert Path(values[spec.name]).exists()
            cfg = mcp_catalog._inline_non_secret_value(cfg, spec.name, values[spec.name])
        try:
            names = register_mcp_servers({"rea": cfg})
            pinned = json.loads((rea / "verification/pinned-mcp-inventory.json").read_text())
            schemas = [e.schema for e in registry.get_all_entries() if e.name in names]
            for tool in pinned["tools"]:
                assert any(n.endswith("__" + tool) or n.endswith("_" + tool) for n in names), tool
            assert len(pinned["tools"]) == 139
            assert all(isinstance(s.get("parameters"), dict) for s in schemas)
            session = next(n for n in names if n.endswith("binary_session"))
            raw = registry.dispatch(session, {})
            result = json.loads(raw) if isinstance(raw, str) else raw
            assert "error" not in result, result
            payload = json.loads(result["result"])
            assert "error" not in payload, payload
            assert payload["result"]["open"] is False
            prompts_raw = registry.dispatch("mcp__rea__list_prompts", {})
            prompts = json.loads(prompts_raw) if isinstance(prompts_raw, str) else prompts_raw
            assert len(prompts["prompts"]) == 6
            artifact = {"server_tools": len(pinned["tools"]), "registered_count": len(names),
                        "names": names, "schemas": schemas, "handler_result": result,
                        "prompts": prompts,
                        "sampling": cfg["sampling"], "production_install": False}
            out = ROOT / "native-evidence.json"
            out.write_text(json.dumps(artifact, indent=2) + "\n")
            print(json.dumps({"passed": True, "registered": len(names), "server_tools":139,
                              "handler": session, "evidence": str(out)}))
        finally:
            shutdown_mcp_servers()


if __name__ == "__main__":
    main()
