#!/usr/bin/env python3
"""Read-only source-surface gate. Never import or run the Hermes backend."""
import argparse
import ast
import json
from pathlib import Path
import subprocess


def surface(source):
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            first = decorator.args[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            function = decorator.func
            if isinstance(function, ast.Name) and function.id == "method":
                found.add(("rpc", first.value))
            elif isinstance(function, ast.Attribute) and function.attr in {"get", "post", "put", "patch", "delete", "websocket"}:
                found.add((function.attr, first.value))
    return found


def check_requirements(root, requirements):
    findings = []
    for relative, required in requirements.items():
        try:
            present = surface((root / relative).read_text())
        except (OSError, SyntaxError, UnicodeError) as exc:
            findings.append(f"{relative}: unreadable source ({type(exc).__name__})")
            continue
        for verb, name in required:
            if (verb, name) not in present:
                findings.append(f"{relative}: missing {verb} {name}")
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hermes-source", type=Path, required=True)
    args = parser.parse_args()
    root = args.hermes_source.resolve()
    manifest = json.loads(Path(__file__).with_name("backend-contract.json").read_text())
    findings = check_requirements(root, manifest["requirements"])
    contract = None
    try:
        tree = ast.parse((root / "tui_gateway/server.py").read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DESKTOP_BACKEND_CONTRACT" for t in node.targets):
                contract = ast.literal_eval(node.value)
        if contract not in manifest["desktop_contracts"]:
            findings.append(f"Unverified desktop protocol contract: {contract!r}. Review adapters and fixtures before release.")
    except (OSError, ValueError, SyntaxError, UnicodeError):
        findings.append("Cannot inspect desktop protocol contract.")
    try:
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "--no-optional-locks", "-C", str(root), "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None
    print(json.dumps({"status": "pass" if not findings else "incompatible", "evidence_class": "diagnostic-only",
                      "check": "source-surface", "backend_head": sha, "backend_dirty": dirty,
                      "desktop_contract": contract, "findings": findings,
                      "runtime_verified": False, "physical_device_verified": False}, indent=2))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
