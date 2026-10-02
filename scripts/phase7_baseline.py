"""Phase 7 migration evidence, not a permanent catalog snapshot test.

Run with an isolated HERMES_HOME and the checkout's test interpreter:
  python scripts/phase7_baseline.py capture
  python scripts/phase7_baseline.py check

Capture only on the agreed Phase 4 baseline. Check is a migration parity aid;
intentional later product changes should retire/update the evidence, not freeze it.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack
from dataclasses import asdict, fields
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "refactor" / "phase7"
TARGETS = ("hermes_cli.commands", "hermes_cli.slash_exec",
           "hermes_cli.tools_config", "hermes_cli.toolset_scope")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def inventory():
    roots = ("agent", "hermes_cli", "gateway", "tui_gateway", "acp_adapter",
             "tools", "plugins", "plugin_runtime", "cron", "scripts", "tests")
    paths = list(ROOT.glob("*.py"))
    paths += [p for r in roots for p in (ROOT / r).rglob("*.py")]
    hits = []
    for p in sorted(paths):
        source = p.read_text(encoding="utf-8", errors="replace")
        if not any(t in source for t in TARGETS) and not ("from ." in source or "from hermes_cli import" in source):
            continue
        tree = ast.parse(source)
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports[node.lineno] = node.module
            elif isinstance(node, ast.Import):
                imports[node.lineno] = ",".join(a.name for a in node.names)
        for line, text in enumerate(source.splitlines(), 1):
            if any(t in text for t in TARGETS):
                hits.append({"path": p.relative_to(ROOT).as_posix(), "line": line,
                             "kind": "import" if line in imports else "reference",
                             "text": text.strip()})
        # Include relative sibling imports, which textual full-path searches miss.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module == "hermes_cli" or
                (node.level and p.parent == ROOT / "hermes_cli")
            ) and any(a.name in ("commands", "slash_exec", "tools_config", "toolset_scope")
                      for a in node.names):
                hits.append({"path": p.relative_to(ROOT).as_posix(), "line": node.lineno,
                             "kind": "relative/import", "text": ast.get_source_segment(source, node)})
    entries = json.loads((ROOT / "compat_manifest.json").read_text(encoding="utf-8"))["entries"]
    compat = [e for e in entries if e["facade"] in TARGETS]
    mappings = {}
    for name in ("cli.py", "gateway/run_busy.py", "gateway/slash_commands.py",
                 "tui_gateway/methods_slash.py", "acp_adapter/commands.py"):
        p = ROOT / name
        if not p.exists():
            continue
        source = p.read_text(encoding="utf-8")
        tree = ast.parse(source)
        selected = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [getattr(t, "id", getattr(t, "attr", "")) for t in targets]
                if any("DISPATCH" in n or "COMMAND" in n for n in names):
                    selected.append({"line": node.lineno, "source": ast.get_source_segment(source, node)})
        mappings[name] = selected
    return {"consumers": hits, "compatibility_entries": compat, "execution_mappings": mappings}


def behaviour():
    c = importlib.import_module("commands")
    presentation = importlib.import_module("hermes_cli.commands_presentation")
    e = importlib.import_module("commands.execution")
    tc = importlib.import_module("tools.platform_policy")
    from toolsets import TOOLSETS, resolve_toolset
    commands = [asdict(x) for x in c.COMMAND_REGISTRY]
    aliases = {key: c.resolve_command(key).name for x in c.COMMAND_REGISTRY
               for key in (x.name, *x.aliases, "/" + x.name.upper())}
    assert all(e.resolve_executor(x) for x in c.COMMAND_REGISTRY if x.execute)
    gates = {x.name for x in c.COMMAND_REGISTRY if x.gateway_config_gate}
    command_data = {
        "definitions": commands, "lookup": aliases,
        "desktop": c.desktop_surface_registry(),
        "desktop_metadata": {x.name: c.command_desktop_meta(x) for x in c.COMMAND_REGISTRY},
        "gateway_available_closed": [x.name for x in c.COMMAND_REGISTRY if c.is_gateway_available(x, set())],
        "gateway_available_open": [x.name for x in c.COMMAND_REGISTRY if c.is_gateway_available(x, gates)],
        "cli_catalog": presentation.COMMANDS,
        "gateway_known": sorted(c.GATEWAY_KNOWN_COMMANDS),
        "executors": {k: v.__name__ for k, v in e.EXECUTORS.items()},
        "execution_contracts": {x.__name__: [f.name for f in fields(x)]
                                for x in (e.CommandContext, e.CommandReply)},
    }
    scenarios = {
        "default": {},
        "empty": {"platform_toolsets": {"PLATFORM": []}},
        "explicit": {"platform_toolsets": {"PLATFORM": ["terminal", "memory"]}},
        "composite": {"platform_toolsets": {"PLATFORM": ["COMPOSITE"]}},
        "mixed": {"platform_toolsets": {"PLATFORM": ["COMPOSITE", "phase7_plugin"]}},
        "native_explicit": {"platform_toolsets": {"PLATFORM": ["discord", "discord_admin"]}},
        "plugin_new": {},
        "plugin_declined": {"known_plugin_toolsets": {"PLATFORM": ["phase7_plugin"]}},
        "plugin_explicit": {"platform_toolsets": {"PLATFORM": ["phase7_plugin"]},
                            "known_plugin_toolsets": {"PLATFORM": ["phase7_plugin"]}},
        "mcp_default": {"mcp_servers": {"alpha": {}, "beta": {}, "off": {"enabled": False}}},
        "mcp_explicit": {"platform_toolsets": {"PLATFORM": ["terminal", "alpha"]},
                          "mcp_servers": {"alpha": {}, "beta": {}}},
        "no_mcp": {"platform_toolsets": {"PLATFORM": ["terminal", "no_mcp"]},
                   "mcp_servers": {"alpha": {}, "beta": {}}},
        "disabled": {"platform_toolsets": {"PLATFORM": ["COMPOSITE"]},
                     "agent": {"disabled_toolsets": ["terminal", "memory"]}},
        "disabled_composite": {"agent": {"disabled_toolsets": ["debugging"]}},
        "disabled_string": {"agent": {"disabled_toolsets": "['terminal', 'memory']"}},
        "context_engine": {"context": {"engine": "phase7"}},
        "context_engine_empty": {"platform_toolsets": {"PLATFORM": []}, "context": {"engine": "phase7"}},
        "legacy_kanban": {"toolsets": ["kanban"]},
        "known_builtins": {"platform_toolsets": {"PLATFORM": ["terminal"]},
                           "known_builtin_toolsets": {"PLATFORM": sorted(tc.configurable_toolset_keys())}},
        "list_literal": {"platform_toolsets": {"PLATFORM": "['terminal', 'memory']"}},
        "invalid": {"platform_toolsets": {"PLATFORM": ["unknown_phase7"]}},
    }
    results = []
    platforms = sorted(set(tc.PLATFORM_DEFAULT_TOOLSETS) | {"acp", "gui", "cron", "webhook", "unknown_phase7"})
    with ExitStack() as stack:
        stack.enter_context(patch.dict(TOOLSETS, {"phase7_plugin": {"tools": ["phase7_tool"], "includes": []}}))
        for credentials in (False, True):
            with patch.object(
                    tc, "_homeassistant_credentials_present", return_value=credentials):
                for platform in platforms:
                    for label, template in scenarios.items():
                        cfg = json.loads(json.dumps(template).replace("PLATFORM", platform).replace(
                            "COMPOSITE", tc.platform_default_toolset(platform)))
                        plugin_keys = {"phase7_plugin"} if label.startswith("plugin") or label == "mixed" else set()
                        with patch.object(tc, "get_plugin_toolset_keys", return_value=plugin_keys):
                            for default_mcp in (False, True):
                                enabled = tc.get_platform_tools(cfg, platform, include_default_mcp_servers=default_mcp, xai_credentials_present=credentials)
                                results.append({"platform": platform, "scenario": label,
                                                "credentials": credentials, "default_mcp": default_mcp,
                                                "config": cfg, "plugins": sorted(plugin_keys),
                                                "enabled": sorted(enabled),
                                                "expanded": sorted({t for key in enabled for t in resolve_toolset(key)})})
    return {"commands": command_data, "tool_selection": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("capture", "check", "inventory"))
    args = parser.parse_args()
    if args.action == "inventory":
        data = inventory()
        dump(OUT / "dependency-inventory.json", data)
        print(f"Recorded {len(data['consumers'])} source/test references; {len(data['compatibility_entries'])} compatibility entries")
        return
    actual = json.loads(json.dumps(behaviour()))
    path = OUT / "behaviour-baseline.json"
    if args.action == "capture":
        if path.exists():
            raise SystemExit("Refusing to overwrite frozen baseline")
        dump(path, actual)
    else:
        expected = json.loads(path.read_text(encoding="utf-8"))
        # Phase 7.4 approved correction; keep the frozen evidence unchanged.
        corrected = 0
        for row in expected["tool_selection"]:
            selection = (row["config"].get("platform_toolsets") or {}).get(row["platform"])
            if selection == []:
                corrected += bool(row["enabled"] or row["expanded"])
                row["enabled"] = []
                row["expanded"] = []
        print(f"Approved explicit-empty corrections: {corrected}")
        if actual != expected:
            dump(OUT / "behaviour-current.json", actual)
            raise SystemExit("Behaviour differs; compare behaviour-current.json with baseline")
    print(f"{args.action}: {len(actual['commands']['definitions'])} commands; "
          f"{len(actual['tool_selection'])} selection cases")


if __name__ == "__main__":
    main()
