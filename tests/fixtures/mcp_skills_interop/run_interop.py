#!/usr/bin/env python3
"""Run an independent MCP SDK fixture through a real Hermes candidate."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_candidate(root: Path) -> None:
    required = [
        root / "tools" / "mcp_skills_protocol.py",
        root / "tools" / "mcp_skills_registry.py",
        root / "tools" / "mcp_skills_view.py",
    ]
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Hermes candidate lacks Skills Over MCP host files: " + ", ".join(missing))


def load_json(raw: str, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    check(isinstance(value, dict), f"{label} did not return a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes", required=True, type=Path, help="Hermes source tree to exercise")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for the stdio fixture")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--output", type=Path, help="receipt path; defaults under outputs/")
    args = parser.parse_args()

    fixture_root = Path(__file__).resolve().parent
    hermes = args.hermes.expanduser().resolve()
    require_candidate(hermes)
    output = (args.output or fixture_root / "outputs" / "interop-latest.json").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    temp_root = Path(tempfile.mkdtemp(prefix="public-skills-interop-"))
    home = temp_root / "hermes-home"
    workspace = temp_root / "workspace"
    events_path = output.with_suffix(".events.jsonl")
    home.mkdir()
    workspace.mkdir()
    events_path.unlink(missing_ok=True)

    local_dir = home / "skills" / "fixture" / "portable-demo"
    local_dir.mkdir(parents=True)
    local_dir.joinpath("SKILL.md").write_text(
        "---\nname: portable-demo\ndescription: Local collision control\n---\n\n# Local control\n",
        encoding="utf-8",
    )

    os.environ["HERMES_HOME"] = str(home)
    os.chdir(workspace)
    sys.path.insert(0, str(hermes))

    label = "public-fixture"
    server_config = {
        "command": args.python,
        "args": ["-B", str(fixture_root / "fixture_server.py"), "--events", str(events_path)],
        "skills": {"enabled": True},
        "protocol": "stateless",
        "connect_timeout": min(args.timeout, 15.0),
        "tool_timeout": min(args.timeout, 15.0),
    }
    receipt: dict[str, Any] = {
        "status": "FAIL",
        "fixture": "public-skills-over-mcp",
        "hermes_source": str(hermes),
        "checks": {},
    }

    mcp_tool = None
    try:
        mcp_tool = importlib.import_module("tools.mcp_tool")
        mcp_discovery = importlib.import_module("tools.mcp_tool_discovery")
        mcp_lifecycle = importlib.import_module("tools.mcp_tool_lifecycle")
        mcp_skills_protocol = importlib.import_module("tools.mcp_skills_protocol")
        system_prompt = importlib.import_module("agent.system_prompt")
        skills_tool = importlib.import_module("tools.skills_tool")

        registered = mcp_discovery.register_mcp_servers({label: server_config})
        server = mcp_tool._servers.get(label)
        check(server is not None, "Hermes did not retain the connected fixture server")
        check(server._ready.is_set(), "fixture server was not ready")
        if server._skills_diagnostic:
            receipt["host_conformance_issue"] = server._skills_diagnostic
        check(server._skills_diagnostic is None, f"skill discovery diagnostic: {server._skills_diagnostic}")
        check(len(server._skills_catalog) == 3, "paginated discovery did not return all three skills")
        receipt["checks"]["real_stdio_discovery"] = {
            "registered_tools": registered,
            "catalog_entries": len(server._skills_catalog),
            "directory_read": server._skills_directory_read,
        }

        startup_events = read_events(events_path)
        startup_methods = [event["method"] for event in startup_events]
        check(startup_methods == ["skills/list", "skills/list", "skills/list"],
              f"startup was not metadata-only: {startup_methods}")
        receipt["checks"]["metadata_only_startup"] = startup_methods

        directory_future = asyncio.run_coroutine_threadsafe(
            mcp_skills_protocol.read_directory(server.session, "skill://portable-demo"),
            mcp_tool._mcp_loop,
        )
        directory = directory_future.result(timeout=args.timeout)
        child_names = sorted(resource.name for resource in directory.resources)
        check(child_names == ["SKILL.md", "assets", "references"],
              f"directory/read returned unexpected children: {child_names}")
        receipt["checks"]["directory_read"] = child_names

        session_a = "interop-primary-session"
        agent = SimpleNamespace(
            valid_tool_names=["skill_view", "skills_list"],
            platform="cli",
            session_id=session_a,
            _hermes_home=home,
        )
        prompt = system_prompt._skills_prompt(agent)
        check("Local collision control" in prompt, "local control skill absent from prompt")
        check("Demonstrate portable Skills Over MCP" in prompt, "remote metadata absent from prompt")
        check("mcp:public-fixture:skill://portable-demo/SKILL.md" in prompt,
              "qualified remote identity absent from prompt")
        receipt["checks"]["real_prompt"] = {
            "local_description": True,
            "remote_description": True,
            "qualified_origin": True,
        }

        listed = load_json(skills_tool.skills_list(session_id=session_a), "skills_list")
        matching = [row for row in listed.get("skills", []) if row.get("name") == "portable-demo"]
        check(len(matching) == 3, f"expected local plus two remote collisions, got {len(matching)}")
        check(sum(row.get("source") == "mcp" for row in matching) == 2, "remote collisions were dropped")
        ambiguous = load_json(
            skills_tool.skill_view(
                f"mcp:{label}:portable-demo", task_id=session_a, session_id=session_a),
            "ambiguous skill_view",
        )
        check(ambiguous.get("success") is False and "ambiguous" in ambiguous.get("error", "").lower(),
              "bare same-origin collision did not fail as ambiguous")
        receipt["checks"]["collisions"] = {"visible_entries": 3, "ambiguous_bare_name_rejected": True}

        primary_name = f"mcp:{label}:skill://portable-demo/SKILL.md"
        primary = load_json(
            skills_tool.skill_view(primary_name, task_id=session_a, session_id=session_a),
            "primary skill_view",
        )
        check(primary.get("success") is True, f"primary skill_view failed: {primary}")
        check("REMOTE MCP SKILL" in primary.get("content", ""), "remote origin warning absent")
        check(primary.get("permissions_inert") is True, "remote frontmatter permissions were not inert")

        guide = load_json(
            skills_tool.skill_view(
                primary_name, file_path="references/GUIDE.md", task_id=session_a, session_id=session_a),
            "guide skill_view",
        )
        check(guide.get("success") is True, f"supporting text read failed: {guide}")
        check("cobalt-lantern-27" in guide.get("content", ""), "supporting text was incorrect")

        binary_destination = Path("materialized/pixel.png")
        binary = load_json(
            skills_tool.skill_view(
                primary_name,
                file_path="assets/pixel.png",
                task_id=session_a,
                session_id=session_a,
                materialize=True,
                destination=str(binary_destination),
            ),
            "binary skill_view",
        )
        check(binary.get("success") is True and binary.get("binary") is True,
              f"binary materialization failed: {binary}")
        check(binary.get("scan", {}).get("status") == "not_scanned_binary",
              "opaque binary was incorrectly labeled text-scanned")
        expected_binary = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d49444154789c63606060f80f00010401005fe5c34b0000000049454e44ae426082")
        materialized_path = Path(binary["materialization"]["path"])
        check(materialized_path.read_bytes() == expected_binary, "materialized binary bytes differ")
        receipt["checks"]["native_skill_view"] = {
            "primary_loaded": True,
            "support_text_loaded": True,
            "binary_materialized": True,
            "binary_sha256": hashlib.sha256(expected_binary).hexdigest(),
        }

        session_b = "interop-collision-session"
        collision_name = f"mcp:{label}:skill://catalog-b/portable-demo/SKILL.md"
        collision = load_json(
            skills_tool.skill_view(collision_name, task_id=session_b, session_id=session_b),
            "collision skill_view",
        )
        check(collision.get("success") is True, f"second URI skill_view failed: {collision}")
        check("second same-name skill" in collision.get("raw_content", ""), "wrong collision content loaded")
        receipt["checks"]["separate_activation_sessions"] = [session_a, session_b]

        session_c = "interop-alternate-scheme-session"
        alternate_name = f"mcp:{label}:fixture-skill://catalog-c/portable-alt/SKILL.md"
        alternate = load_json(
            skills_tool.skill_view(alternate_name, task_id=session_c, session_id=session_c),
            "alternate-scheme skill_view",
        )
        check(alternate.get("success") is True, f"alternate-scheme skill_view failed: {alternate}")
        check("URI scheme" in alternate.get("raw_content", ""), "wrong alternate-scheme content loaded")
        receipt["checks"]["alternate_uri_scheme"] = True
        receipt["checks"]["separate_activation_sessions"].append(session_c)

        all_events = read_events(events_path)
        methods = [event["method"] for event in all_events]
        check(methods[:3] == ["skills/list", "skills/list", "skills/list"], "startup ordering changed")
        check(methods.count("skills/get") >= 3, "native loads did not verify point manifests")
        check(methods.count("resources/directory/read") == 1, "directory handler was not exercised exactly once")
        read_uris = [event["uri"] for event in all_events if event["method"] == "resources/read"]
        check("skill://portable-demo/references/GUIDE.md" in read_uris, "guide was not read over MCP")
        check("skill://portable-demo/assets/pixel.png" in read_uris, "binary was not read over MCP")
        receipt["checks"]["actual_wire_events"] = {
            "count": len(all_events),
            "methods": methods,
            "resource_uris": read_uris,
        }

        receipt["status"] = "PASS"
        return_code = 0
    except Exception as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        return_code = 1
    finally:
        if mcp_tool is not None:
            try:
                lifecycle = locals().get("mcp_lifecycle")
                if lifecycle is None:
                    lifecycle = importlib.import_module("tools.mcp_tool_lifecycle")
                lifecycle.shutdown_mcp_servers()
                receipt["checks"]["shutdown_returned"] = True
            except Exception as exc:
                receipt["checks"]["shutdown_returned"] = False
                receipt["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                receipt["status"] = "FAIL"
                return_code = 1
        receipt["elapsed_seconds"] = round(time.monotonic() - started, 3)
        receipt["events_path"] = str(events_path)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.rmtree(temp_root, ignore_errors=True)
        print(json.dumps(receipt, indent=2, sort_keys=True))

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
