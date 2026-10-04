from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.agent_os_install import apply_rule, audit_targets, readback_rule, target_specs


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"audit", "apply", "readback"}:
        print(
            "usage: python scripts/agent_os_install.py audit|apply|readback [target]",
            file=sys.stderr,
        )
        return 2
    command = argv[0]
    repo_root = Path(__file__).resolve().parents[1]
    if command == "audit":
        print(json.dumps(audit_targets(repo_root=repo_root), indent=2, sort_keys=True))
        return 0
    if len(argv) < 2:
        print("target is required", file=sys.stderr)
        return 2
    target = argv[1]
    if command == "apply":
        print(json.dumps(apply_rule(target, repo_root=repo_root), indent=2, sort_keys=True))
        return 0
    spec = target_specs(Path.home()).get(target)
    if spec is None or not spec.rule_path:
        print(json.dumps({"target": target, "status": "UNSUPPORTED"}, indent=2, sort_keys=True))
        return 0
    print(json.dumps(readback_rule(Path(spec.rule_path)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
