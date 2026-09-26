"""Footer-only accounting for skill writes; never enroll them in stop/checkpoint gates."""
import json
import os
from pathlib import Path

from agent.tool_dispatch_helpers import _extract_error_preview


def _targets(args):
    """Resolve through the tool's own profile/create-dir lookup, not the terminal CWD."""
    from hermes_constants import get_hermes_home
    from tools.skill_manager_tool import _find_skill, _resolve_skill_dir

    home = os.path.normcase(str(get_hermes_home().resolve()))
    operations = args.get("operations")
    operations = operations if isinstance(operations, list) and operations else [args]
    for op in operations:
        if not isinstance(op, dict):
            continue
        name = op.get("name") or args.get("name") or "<unnamed skill>"
        if not isinstance(name, str):
            continue
        action = op.get("action")
        # A deletion must not be cleared by an unrelated write to the same path.
        effect = "delete" if action in {"delete", "remove_file"} else "write"
        file_path = op.get("file_path") or "SKILL.md"
        if action in {"create", "edit"} or (action == "patch" and op.get("content")):
            file_path = "SKILL.md"
        if not isinstance(file_path, str):
            continue
        try:
            found = _find_skill(name)
            directory = Path(found["path"]) if found else _resolve_skill_dir(name, op.get("category"))
            target = directory if action == "delete" else directory / file_path
            canonical = os.path.normcase(str(target.resolve()))
            display = target.as_posix()
        except (OSError, ValueError, TypeError, RuntimeError):
            # Failed argument validation or unavailable lookup must not hide the failure.
            canonical = display = f"{name}/{file_path}"
        identity = ("skill_manage", home, canonical, effect)
        yield f"skill:{display} ({effect})", identity


def record_skill_mutation_result(agent, args, result, is_error):
    """Track atomic batch/legacy outcomes. Staging is not evidence of a completed write."""
    state = getattr(agent, "_turn_failed_file_mutations", None)
    if state is None:
        return
    try:
        data = json.loads(result) if isinstance(result, str) else None
    except (ValueError, TypeError):
        data = None
    landed = (isinstance(data, dict) and data.get("success") is True
              and not data.get("error") and not data.get("staged"))
    failed = is_error or (isinstance(data, dict) and (
        bool(data.get("error")) or data.get("success") is False))
    if not landed and not failed:
        return
    targets = list(_targets(args))
    if landed:
        identities = {identity for _, identity in targets}
        for key, info in list(state.items()):
            if info.get("identity") in identities:
                state.pop(key, None)
    else:
        preview = _extract_error_preview(result)
        for label, identity in targets:
            # No stat snapshot: skill batches roll back, and directory mtimes cannot
            # prove an individual edit landed. Only a successful same-target receipt can.
            state.setdefault(label, {"tool": "skill_manage", "identity": identity,
                                     "error_preview": preview})
