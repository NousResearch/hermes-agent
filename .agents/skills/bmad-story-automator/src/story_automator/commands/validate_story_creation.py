from __future__ import annotations

import json
import os
from pathlib import Path

from story_automator.core.artifact_paths import implementation_artifacts_dir
from story_automator.core.runtime_policy import PolicyError
from story_automator.core.success_verifiers import create_story_artifact, resolve_success_contract


ARTIFACT_RESOLUTION_ERRORS = (OSError, ValueError)


def cmd_validate_story_creation(args: list[str]) -> int:
    action = args[0] if args else ""
    rest = args[1:] if args else []
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    default_artifacts_dir: Path | None = None

    def resolve_default_artifacts_dir() -> Path:
        nonlocal default_artifacts_dir
        if default_artifacts_dir is None:
            default_artifacts_dir = implementation_artifacts_dir(project_root)
        return default_artifacts_dir

    def story_prefix(story_id: str) -> str:
        return story_id.replace(".", "-")

    def count_files(story_id: str, folder: Path) -> int:
        return len(list(folder.glob(f"{story_prefix(story_id)}-*.md")))

    def create_check_payload(story_id: str, state_file: str) -> dict[str, object]:
        contract = resolve_success_contract(project_root, "create", state_file=state_file or None)
        return create_story_artifact(project_root=project_root, story_key=story_id, contract=contract)

    def expected_matches(payload: dict[str, object] | None) -> int:
        if payload is None:
            return 1
        value = payload.get("expectedMatches", 1)
        return value if isinstance(value, int) else 1

    def count_reason(created: int, expected: int) -> str:
        if created == expected:
            return "Exactly 1 story file created as expected" if expected == 1 else f"Exactly {expected} story files created as expected"
        if created == 0:
            return "No story file created - session may have failed"
        if created < 0:
            return f"Story files decreased ({created}) - unexpected deletion"
        if created > expected:
            return f"RUNAWAY CREATION: {created} files created instead of {expected}"
        return f"Unexpected story artifact count: {created} files instead of {expected}"

    def build_check_response(
        story_id: str,
        payload: dict[str, object] | None,
        *,
        before_count: int | None = None,
        after_count: int | None = None,
        valid_override: bool | None = None,
        reason_override: str | None = None,
    ) -> dict[str, object]:
        expected = expected_matches(payload)
        actual = payload.get("actualMatches", 0) if payload is not None else 0
        created = actual if isinstance(actual, int) else 0
        valid = bool(payload.get("verified")) if payload is not None else False
        reason = count_reason(created, expected)
        if before_count is not None and after_count is not None:
            created = after_count - before_count
            valid = created == expected
            reason = count_reason(created, expected)
        if valid_override is not None:
            valid = valid_override
        if reason_override is not None:
            reason = reason_override
        response: dict[str, object] = {
            "valid": valid,
            "verified": valid,
            "created_count": created,
            "expected": expected,
            "prefix": story_prefix(story_id),
            "action": "proceed" if valid else "escalate",
            "reason": reason,
            "source": payload.get("source", "") if payload is not None else "",
            "pattern": payload.get("pattern", "") if payload is not None else "",
            "matches": payload.get("matches", []) if payload is not None else [],
        }
        if before_count is not None and after_count is not None:
            response["before"] = before_count
            response["after"] = after_count
        if payload is not None and payload.get("story"):
            response["story"] = payload["story"]
        return response

    def print_check_error(
        story_id: str,
        *,
        reason: str,
        before_count: int | None = None,
        after_count: int | None = None,
    ) -> int:
        response = build_check_response(
            story_id,
            None,
            before_count=before_count,
            after_count=after_count,
            valid_override=False,
            reason_override=reason,
        )
        print(json.dumps(response, separators=(",", ":")))
        return 1

    def parsed_delta_counts(before_value: str | None, after_value: str | None) -> tuple[int | None, int | None]:
        if before_value is None or after_value is None:
            return None, None
        try:
            return int(before_value or ""), int(after_value or "")
        except ValueError:
            return None, None

    if action == "count":
        if not rest:
            print("Usage: validate-story-creation count <story_id>", file=os.sys.stderr)
            return 1
        story_id = rest[0]
        try:
            artifacts_dir: Path | None = None
            idx = 1
            while idx < len(rest):
                arg = rest[idx]
                if arg == "--artifacts-dir":
                    if idx + 1 >= len(rest) or not rest[idx + 1] or rest[idx + 1].startswith("--"):
                        raise ValueError("--artifacts-dir requires a value")
                    artifacts_dir = Path(rest[idx + 1])
                    idx += 2
                    continue
                raise ValueError(f"unsupported count argument: {arg}")
            if artifacts_dir is None:
                artifacts_dir = resolve_default_artifacts_dir()
            print(count_files(story_id, artifacts_dir))
            return 0
        except ARTIFACT_RESOLUTION_ERRORS as exc:
            print(str(exc), file=os.sys.stderr)
            return 1

    if action == "check":
        if not rest:
            return print_check_error("", reason="story_id required")
        story_id = rest[0]
        state_file = ""
        artifacts_dir: Path | None = None
        before_value = after_value = None
        before_seen = after_seen = False
        idx = 1
        while idx < len(rest):
            if rest[idx] == "--before":
                before_seen = True
                if idx + 1 < len(rest):
                    before_value = rest[idx + 1]
                    idx += 2
                else:
                    before_count, after_count = parsed_delta_counts(before_value, after_value)
                    return print_check_error(story_id, reason="--before requires a value", before_count=before_count, after_count=after_count)
                continue
            if rest[idx] == "--after":
                after_seen = True
                if idx + 1 < len(rest):
                    after_value = rest[idx + 1]
                    idx += 2
                else:
                    before_count, after_count = parsed_delta_counts(before_value, after_value)
                    return print_check_error(story_id, reason="--after requires a value", before_count=before_count, after_count=after_count)
                continue
            if rest[idx] == "--artifacts-dir" and idx + 1 < len(rest):
                artifacts_dir = Path(rest[idx + 1])
                idx += 2
                continue
            if rest[idx] == "--artifacts-dir":
                before_count, after_count = parsed_delta_counts(before_value, after_value)
                return print_check_error(story_id, reason="--artifacts-dir requires a value", before_count=before_count, after_count=after_count)
            if rest[idx] == "--state-file" and idx + 1 < len(rest):
                state_file = rest[idx + 1]
                idx += 2
                continue
            if rest[idx] == "--state-file":
                before_count, after_count = parsed_delta_counts(before_value, after_value)
                return print_check_error(story_id, reason="--state-file requires a value", before_count=before_count, after_count=after_count)
            before_count, after_count = parsed_delta_counts(before_value, after_value)
            return print_check_error(story_id, reason=f"unsupported check argument: {rest[idx]}", before_count=before_count, after_count=after_count)
        if before_seen != after_seen:
            return print_check_error(story_id, reason="both --before and --after are required together")
        before_count = after_count = None
        if before_seen and after_seen:
            try:
                before_count = int(before_value or "")
                after_count = int(after_value or "")
            except ValueError:
                return print_check_error(story_id, reason="before/after must be integers")
        try:
            default_artifacts_dir = resolve_default_artifacts_dir()
        except ARTIFACT_RESOLUTION_ERRORS as exc:
            return print_check_error(story_id, reason=str(exc), before_count=before_count, after_count=after_count)
        if artifacts_dir is not None and artifacts_dir != default_artifacts_dir:
            return print_check_error(
                story_id,
                reason="validate-story-creation check no longer supports --artifacts-dir overrides; use count/list for custom folders",
                before_count=before_count,
                after_count=after_count,
            )
        try:
            payload = create_check_payload(story_id, state_file)
            response = build_check_response(story_id, payload, before_count=before_count, after_count=after_count)
        except (FileNotFoundError, PolicyError, OSError, ValueError) as exc:
            return print_check_error(story_id, reason=str(exc), before_count=before_count, after_count=after_count)
        print(json.dumps(response, separators=(",", ":")))
        return 0

    if action == "list":
        if not rest:
            print("Usage: validate-story-creation list <story_id>", file=os.sys.stderr)
            return 1
        story_id = rest[0]
        try:
            artifacts_dir = resolve_default_artifacts_dir()
        except ARTIFACT_RESOLUTION_ERRORS as exc:
            print(str(exc), file=os.sys.stderr)
            return 1
        print(f"Story files matching {story_prefix(story_id)}-*.md:")
        matches = list(artifacts_dir.glob(f"{story_prefix(story_id)}-*.md"))
        if not matches:
            print("  (none found)")
            return 0
        for match in matches:
            info = match.stat()
            print(f"-rw-r--r-- 1 {info.st_mode} {info.st_size} {match}")
        return 0

    if action == "prefix":
        if not rest:
            return 1
        print(story_prefix(rest[0]))
        return 0

    if action and action not in {"count", "check", "list", "prefix"}:
        if not rest:
            return print_check_error(action, reason="both --before and --after are required together")
        if len(rest) == 1:
            return cmd_validate_story_creation(["check", action, "--before", rest[0]])
        return cmd_validate_story_creation(["check", action, "--before", rest[0], "--after", rest[1], *rest[2:]])

    print("Usage: validate-story-creation <action> [args]", file=os.sys.stderr)
    print("", file=os.sys.stderr)
    print("Actions:", file=os.sys.stderr)
    print("  count <story_id>              - Count current story files", file=os.sys.stderr)
    print("  check <story_id> [--state-file PATH]   - Compatibility wrapper for create verifier", file=os.sys.stderr)
    print("  list <story_id>               - List matching files", file=os.sys.stderr)
    print("  prefix <story_id>             - Convert story ID to file prefix", file=os.sys.stderr)
    return 1
