"""``hermes jit`` — JIT Micro-Tool Synthesizer & Dynamic Sandbox.

Manage, inspect, and test dynamically synthesized runtime micro-tools.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.jit_tool_synthesizer import (
    JITSandboxExecutor,
    JITToolSynthesizer,
)


def jit_list(args: argparse.Namespace) -> int:
    synth = JITToolSynthesizer()
    tools = synth.list_tools()
    if not tools:
        print("No JIT micro-tools currently registered.")
        return 0

    print(f"{'TOOL NAME':<24} {'CALLS':<8} {'EPHEMERAL':<12} {'DESCRIPTION'}")
    print("-" * 75)
    for t in tools:
        eph = "Yes" if t.is_ephemeral else "Durable"
        desc = t.description[:40] + ("..." if len(t.description) > 40 else "")
        print(f"{t.name:<24} {t.call_count:<8} {eph:<12} {desc}")
    return 0


def jit_test(args: argparse.Namespace) -> int:
    synth = JITToolSynthesizer()
    tool = synth.get_tool(args.name)
    if not tool:
        print(f"\033[31m[-] Tool '{args.name}' not found in JIT registry.\033[0m")
        return 1

    print(f"Testing JIT tool '{tool.name}' with {len(tool.test_vectors)} test vectors in sandbox...")
    try:
        compiled = JITSandboxExecutor.compile_and_extract(tool.name, tool.python_source)
        if tool.test_vectors:
            JITSandboxExecutor.fuzz_test_tool(tool.name, compiled, tool.test_vectors)
        print(f"\033[32m[+] All test vectors passed successfully for '{tool.name}'!\033[0m")
        return 0
    except Exception as exc:
        print(f"\033[31m[-] Sandbox test failed: {exc}\033[0m")
        return 1


def jit_prune(args: argparse.Namespace) -> int:
    synth = JITToolSynthesizer()
    count = synth.prune_ephemeral()
    print(f"Pruned {count} ephemeral JIT micro-tools.")
    return 0


def jit_remove(args: argparse.Namespace) -> int:
    synth = JITToolSynthesizer()
    if synth.deregister(args.name):
        print(f"\033[32mSuccessfully removed tool '{args.name}'.\033[0m")
        return 0
    else:
        print(f"\033[31mTool '{args.name}' not found.\033[0m")
        return 1


def build_jit_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "jit",
        help="JIT Micro-Tool Synthesizer & Dynamic Sandbox",
        description="Inspect, test, and manage on-the-fly synthesized micro-tools.",
    )
    sub = parser.add_subparsers(dest="jit_action")

    # list
    p_list = sub.add_parser("list", help="List active synthesized JIT micro-tools")
    p_list.set_defaults(func=jit_list)

    # test
    p_test = sub.add_parser("test", help="Test a synthesized tool against test vectors in sandbox")
    p_test.add_argument("name", help="Name of the JIT tool")
    p_test.set_defaults(func=jit_test)

    # prune
    p_prune = sub.add_parser("prune", help="Prune all ephemeral session-scoped JIT tools")
    p_prune.set_defaults(func=jit_prune)

    # remove
    p_remove = sub.add_parser("remove", help="Deregister and remove a JIT tool")
    p_remove.add_argument("name", help="Name of the JIT tool to remove")
    p_remove.set_defaults(func=jit_remove)
