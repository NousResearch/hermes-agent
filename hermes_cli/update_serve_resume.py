"""Fresh-process adapter for manual backends paused by historical updaters."""
from __future__ import annotations

import json
from pathlib import Path
import sys


def main(context: Path, result: Path) -> int:
    handled = False
    try:
        request = json.loads(context.read_text(encoding="utf-8-sig"))
        root = Path(request["root"])
        token = request["stopped_serves"]
        if not token.get("pending"):
            handled = True
            return 0
        # A hosted process never pulls a foreign tree onto its import path: the predicate
        # answers the new root BEFORE the insert. This file also runs as a standalone -I -S
        # bootstrap and inside stripped-down update trees, where the Hermes application
        # modules (and any hosted verdict) are absent: there the guard stands down — the
        # in-process callers with a full installation get the refusal.
        try:
            from hermes_cli.hosted_code_fence import HostedCodeRefused, hosted_code_refusal
            refusal = hosted_code_refusal(root)
        except ImportError:
            HostedCodeRefused, refusal = None, ""
        if refusal:
            raise HostedCodeRefused(refusal)
        sys.path.insert(0, str(root))
        from pm.environments import activate_dependencies
        activate_dependencies(root)
        from hermes_cli.dashboard_procs import _filter_dashboard_respawn_candidates
        from hermes_cli.main_dashboard import _respawn_dashboard_processes

        candidates = []
        skipped = 0
        for entry in token.get("entries") or []:
            port = entry.get("port")
            if (entry.get("purpose") not in ("serve", "dashboard")
                    or type(port) is not int or not 0 < port <= 65535):
                skipped += 1
                continue
            command = [sys.executable, str(root / "hermes")]
            profile = entry.get("profile")
            if profile and profile != "default":
                command += ["--profile", str(profile)]
            command.append(entry["purpose"])
            if entry.get("host"):
                command += ["--host", str(entry["host"])]
            command += ["--port", str(port)]
            candidates.append((entry.get("pid", 0), command, entry.get("hermes_home") or None))
        commands = _filter_dashboard_respawn_candidates(candidates, own_home=request.get("home"))
        skipped += len(candidates) - len(commands)
        # An acknowledged attempt is terminal even if spawning failed: replaying
        # the whole token at atexit would duplicate the successful backends.
        failed = _respawn_dashboard_processes(commands) if commands else []
        handled = True
        if skipped or failed:
            print("Some stopped backends could not be relaunched; restart them manually.", file=sys.stderr)
            return 1
        return 0
    except Exception as exc:
        print(f"Stopped serve recovery failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        result.write_text(json.dumps({"serves_handled": handled}), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
