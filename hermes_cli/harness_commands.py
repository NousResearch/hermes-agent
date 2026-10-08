"""Command handlers for typed reversible harness overlays."""
from __future__ import annotations

import sys

from hermes_cli import harness_manifest as manifest


def _show(args):
    command = getattr(args, "harness_command", None) or "show"
    state = manifest.show_state()
    if command == "diff":
        defaults = {item.path: item.default for item in manifest.registry()}
        state = {**state, "values": {key: value for key, value in state["values"].items() if value != defaults[key]}}
    if getattr(args, "json", False):
        print(manifest.canonical_json(state))
        return 0
    print(f"Harness manifest: {state['path']}")
    print(f"Stock revision:   {state['stock_revision']}")
    print(f"Fingerprint:      {state['fingerprint']}")
    if command == "diff":
        print("Changed values:")
        for key, value in state["values"].items():
            print(f"  {key}: {value}")
    else:
        print("Overlays:")
        for overlay in state["overlays"]:
            status = "active" if overlay["active"] else "stale"
            print(f"  {overlay['id']} ({status}): {overlay['values']}")
    return 0


def _explain(args):
    result = manifest.explain(args.key)
    print(manifest.canonical_json(result) if getattr(args, "json", False) else _format_explanation(result))
    return 0


def _set(args):
    manifest.set_value(args.key, manifest.parse_cli_value(args.value), overlay=args.overlay, reason=args.reason)
    print(f"Set {args.key} in overlay {args.overlay!r}; fingerprint: {manifest.fingerprint()}")
    return 0


def _revert(args):
    if not args.yes and sys.stdin.isatty():
        answer = input(f"Remove harness overlay {args.overlay!r}? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Cancelled.")
            return 1
    elif not args.yes:
        print("Refusing non-interactive revert without --yes", file=sys.stderr)
        return 2
    manifest.revert_overlay(args.overlay)
    print(f"Reverted harness overlay {args.overlay!r}")
    return 0


_HANDLERS = {"show": _show, "diff": _show, "explain": _explain, "set": _set, "revert": _revert}


def cmd_harness(args):
    try:
        command = getattr(args, "harness_command", None) or "show"
        handler = _HANDLERS.get(command)
        if handler is None:
            raise manifest.HarnessManifestError(f"unknown harness command: {command}")
        return handler(args)
    except manifest.HarnessManifestError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2


def _format_explanation(result):
    lines = [
        f"{result['key']}: {result['description']}",
        f"  type: {result['type']}  safety: {result['safety']}",
        f"  stock: {result['default']!r}",
        f"  active: {result['active_value']!r}",
        f"  revision: {result['stock_revision']}",
    ]
    for source in result["sources"]:
        lines.append(f"  source: {source['overlay']} ({source['reason']}) -> {source['value']!r}")
    return "\n".join(lines)
