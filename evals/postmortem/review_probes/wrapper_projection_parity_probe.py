"""#71996 wrapper-projection parity probe.

Extract whitespace-containing string literals from the approval/security test
corpus and print four stable verdicts for each. Run from a checkout root, or
pass that root as argv[1]. The deny globs and allowlist patterns are fixed
arguments below; no runtime approval configuration is used.
"""

import ast
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


_DENY_GLOBS = ("rm -rf*", "reboot")
_ALLOWLIST_PATTERNS = ("su *", "doas *", "watch *", "ls")
_TEST_NAME_FRAGMENTS = ("approval", "hardline", "deny", "allowlist", "bypass", "command")


def _corpus(project_root: Path) -> list[str]:
    files = {
        path
        for fragment in _TEST_NAME_FRAGMENTS
        for path in (project_root / "tests").rglob(f"test_*{fragment}*.py")
    }
    commands = set()
    for path in sorted(files):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and any(char.isspace() for char in node.value)
            ):
                commands.add(node.value)
    return sorted(commands)


def _evaluate(commands, deny_globs, allowlist_patterns):
    from tools import approval, approval_context
    from tools.approval_detection import detect_dangerous_command, detect_hardline_command
    from tools.approval_floors import (
        _command_matches_permanent_allowlist,
        _match_user_deny_rule,
    )

    with (
        patch.object(
            approval_context,
            "_get_approval_config",
            return_value={"deny": list(deny_globs)},
        ),
        patch.object(
            approval,
            "_permanent_set",
            side_effect=lambda: set(allowlist_patterns),
        ),
    ):
        return {
            command: {
                "hardline": detect_hardline_command(command)[0],
                "dangerous": detect_dangerous_command(command)[0],
                "deny": bool(_match_user_deny_rule(command)),
                "allowlist": _command_matches_permanent_allowlist(command),
            }
            for command in commands
        }


def main() -> None:
    project_root = Path(sys.argv[1] if len(sys.argv) > 1 else os.getcwd()).resolve()
    sys.path.insert(0, str(project_root))
    commands = _corpus(project_root)

    with tempfile.TemporaryDirectory(prefix="wrapper-parity-") as hermes_home:
        os.environ["HERMES_HOME"] = hermes_home
        os.environ["HERMES_TEST_MODE"] = "1"
        verdicts = _evaluate(commands, _DENY_GLOBS, _ALLOWLIST_PATTERNS)

    output = {
        "deny_globs": list(_DENY_GLOBS),
        "allowlist_patterns": list(_ALLOWLIST_PATTERNS),
        "input_count": len(commands),
        "verdicts": verdicts,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
