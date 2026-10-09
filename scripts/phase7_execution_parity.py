"""Phase 7.3 migration evidence against the committed 7.2 implementation.

This is a one-time ownership migration check, not a permanent product snapshot test.
Run from the checkout with its test interpreter. No legacy module is installed or retained.
"""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASELINE = "a50d2b74a5e3430c5519fc7ab26f537824003ee8"


def main():
    source = subprocess.check_output(
        ["git", "show", BASELINE + ":hermes_cli/slash_exec.py"], cwd=ROOT).decode("utf8")
    legacy = types.ModuleType("_phase7_execution_reference")
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, "<committed Phase 7.2 reference>", "exec"), legacy.__dict__)
    with tempfile.TemporaryDirectory(prefix="phase7-execution-") as home, ExitStack() as stack:
        stack.enter_context(patch.dict(os.environ, {"HERMES_HOME": home, "HERMES_LANGUAGE": "en"}))
        from commands import COMMAND_REGISTRY
        from commands import execution as current
        from agent.i18n import t
        from hermes_cli.profiles import profile_command_details
        from gateway.command_presentation import gateway_help_lines

        label = lambda: "Hermes Agent vfixture · canary · local 0123456789ab"
        status = lambda: "Egress proxy status\n\nEnabled: no\nProcess: stopped"
        catalog = lambda allowed: ["`/help` -- Help"] if allowed is not None else [
            "`/help` -- Help", "`/version` -- Version", "`/profile` -- Profile"]
        bundles = [{"slug": "review", "skills": ["audit", "check"], "description": ""},
                   {"slug": "tidy", "skills": [], "description": "Tidy files"}]
        fixtures = {
            "hermes_cli.banner.format_banner_version_label": label,
            "hermes_cli.proxy_cli.format_status_text": status,
            "hermes_cli.profiles.get_active_profile_name": lambda: "fixture",
            "hermes_cli.profiles.get_profile_dir": lambda name: Path(name),
            "hermes_cli.profiles.read_profile_meta": lambda path: {"display_name": str(path).upper()},
            "agent.skill_bundles.list_bundles": lambda: bundles,
            "agent.skill_bundles._bundles_dir": lambda: Path("/fixture/bundles"),
            "agent.skill_commands.get_skill_commands": lambda: {
                "/audit": {"description": ""}, "/tidy": {"description": "Tidy files"}},
            "tools.skills_tool._find_all_skills": lambda: [{"name": "handoff"}, {"name": "tidy"}],
            "gateway.command_presentation.gateway_help_lines": catalog,
        }
        for target, value in fixtures.items():
            stack.enter_context(patch(target, value))
        # Read the patched callable rather than the pre-patch imported binding.
        from gateway.command_presentation import gateway_help_lines
        contexts = [("", {}), ("", {"allowed_commands": {"help"}})]
        contexts += [(arg, {"page_size": size}) for arg in ("oops", "-1", "1", "2", "999")
                     for size in (None, "oops", 0, 2, 15, 20)]
        count = 0
        for name in ("version", "egress", "profile", "bundles", "help", "commands"):
            scenarios = contexts if name == "commands" else contexts[:2]
            if name == "profile":
                scenarios += [("", {"profile_name": n, "home_display": "/profiles/" + n})
                              for n in ("alpha", "beta", "alpha")]
            for surface in ("cli", "gateway", "tui", "acp"):
                for args, options in scenarios:
                    supplied = {**options, "version_label": label, "egress_status": status,
                                "translate": t, "help_lines": gateway_help_lines}
                    if name == "profile":
                        supplied.update(profile_command_details(
                            options.get("profile_name"), options.get("home_display")))
                    before = legacy.execute_command(name, legacy.CommandContext(
                        surface=surface, args=args, options=options))
                    after = current.execute_command(name, current.CommandContext(
                        surface=surface, args=args, options=supplied))
                    assert asdict(before) == asdict(after), (name, surface, args, options)
                    count += 1
        for name in ("CommandContext", "CommandReply"):
            assert [f.name for f in fields(getattr(legacy, name))] == [
                f.name for f in fields(getattr(current, name))]
        assert set(current.EXECUTORS) == {c.execute for c in COMMAND_REGISTRY if c.execute}
        result = {"baseline": BASELINE, "result": "identical", "cases": count,
                  "surfaces": ["cli", "gateway", "tui", "acp"],
                  "fixtures": "deterministic application inputs and existing subsystem APIs",
                  "data_contracts": "unchanged", "execution_keys": len(current.EXECUTORS)}
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
