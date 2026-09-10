"""``hermes chronos`` — Time-Travel / Branching Tree-of-Thought Trajectory Debugger.

Inspect, rewind, and branch multi-step agent execution timelines.
"""

from __future__ import annotations

import argparse
import json
import sys

from agent.chronos_debugger import ChronosTrajectoryDebugger


def chronos_log(args: argparse.Namespace) -> int:
    debugger = ChronosTrajectoryDebugger(session_id=args.session)
    tree_text = debugger.render_ascii_tree()
    print(tree_text)
    return 0


def chronos_inspect(args: argparse.Namespace) -> int:
    debugger = ChronosTrajectoryDebugger(session_id=args.session)
    if args.frame_id not in debugger.frames:
        print(f"\033[31m[-] Frame '{args.frame_id}' not found in trajectory.\033[0m")
        return 1

    f = debugger.frames[args.frame_id]
    print(f"=== Chronos Frame: {f.frame_id} ===")
    print(f"Branch:      {f.branch_name}")
    print(f"Turn Index:  {f.turn_index}")
    print(f"Parent:      {f.parent_frame_id or '(root)'}")
    print(f"Action:      {f.action_name or 'None'}")
    if f.action_args:
        print(f"Action Args: {json.dumps(f.action_args, indent=2)}")
    print("-" * 60)
    print(f"Thought:\n{f.thought}")
    if f.observation:
        print("-" * 60)
        print(f"Observation:\n{f.observation}")
    print("-" * 60)
    print(f"Messages in Snapshot: {len(f.messages_snapshot)}")
    print(f"Memory Keys:          {list(f.memory_snapshot.keys())}")
    return 0


def chronos_rewind(args: argparse.Namespace) -> int:
    debugger = ChronosTrajectoryDebugger(session_id=args.session)
    try:
        frame = debugger.rewind(args.frame_id)
        print(f"\033[32m[+] Successfully rewound timeline to frame '{frame.frame_id}' on branch '{frame.branch_name}'.\033[0m")
        print(f"Turn: {frame.turn_index} | Restored messages: {len(frame.messages_snapshot)}")
        return 0
    except KeyError as exc:
        print(f"\033[31m[-] Rewind failed: {exc}\033[0m")
        return 1


def chronos_fork(args: argparse.Namespace) -> int:
    debugger = ChronosTrajectoryDebugger(session_id=args.session)
    try:
        frame = debugger.fork_branch(args.frame_id, args.branch)
        print(f"\033[32m[+] Forked new timeline branch '{args.branch}' from frame '{frame.frame_id}'.\033[0m")
        return 0
    except (KeyError, ValueError) as exc:
        print(f"\033[31m[-] Fork failed: {exc}\033[0m")
        return 1


def chronos_diff(args: argparse.Namespace) -> int:
    debugger = ChronosTrajectoryDebugger(session_id=args.session)
    try:
        diff = debugger.diff_frames(args.frame_a, args.frame_b)
        print(f"=== Chronos Trajectory Diff ===")
        print(f"Frame A ({diff['branch_a']}): {diff['frame_a']}")
        print(f"Frame B ({diff['branch_b']}): {diff['frame_b']}")
        print(f"Message Count Delta: {diff['messages_count_diff']}")
        print("-" * 60)
        print("Action Differences:")
        print(json.dumps(diff["action_diff"], indent=2))
        print("Memory Differences:")
        print(json.dumps(diff["memory_diff"], indent=2))
        return 0
    except KeyError as exc:
        print(f"\033[31m[-] Diff failed: {exc}\033[0m")
        return 1


def build_chronos_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "chronos",
        help="Chronos Time-Travel / Branching Tree-of-Thought Trajectory Debugger",
        description="Inspect, rewind, and branch multi-step agent reasoning timelines.",
    )
    parser.add_argument("--session", default=None, help="Target agent session ID")
    sub = parser.add_subparsers(dest="chronos_action")

    # log
    p_log = sub.add_parser("log", help="Display ASCII tree of execution frames and branches")
    p_log.set_defaults(func=chronos_log)

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect detailed state at a specific frame")
    p_inspect.add_argument("frame_id", help="Frame identifier")
    p_inspect.set_defaults(func=chronos_inspect)

    # rewind
    p_rewind = sub.add_parser("rewind", help="Rewind active session to a historical frame")
    p_rewind.add_argument("frame_id", help="Target frame identifier")
    p_rewind.set_defaults(func=chronos_rewind)

    # fork
    p_fork = sub.add_parser("fork", help="Branch off an alternate timeline from a frame")
    p_fork.add_argument("frame_id", help="Base frame identifier")
    p_fork.add_argument("branch", help="New branch name")
    p_fork.set_defaults(func=chronos_fork)

    # diff
    p_diff = sub.add_parser("diff", help="Diff two timeline frames")
    p_diff.add_argument("frame_a", help="First frame identifier")
    p_diff.add_argument("frame_b", help="Second frame identifier")
    p_diff.set_defaults(func=chronos_diff)
