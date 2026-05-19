"""Property-style tests for Hermes patch vs Codex freeform apply_patch.

These tests intentionally avoid Hypothesis to keep the project dependency
surface unchanged.  They use deterministic pseudo-random generation so failures
are reproducible by seed and case index.
"""

import json
import random
import string
from types import SimpleNamespace


from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _normalize_codex_response,
)
from agent.tool_projection import (
    INTENT_KEY,
    PatchToolSurface,
    project_messages_for_patch_surface,
)
from tools.file_tools import apply_patch_tool


SEED = 0xA9917A
CASES = 400


def _text(rng: random.Random, *, min_len: int = 0, max_len: int = 40) -> str:
    alphabet = string.ascii_letters + string.digits + " _-./:[]{}()'\",=+\\"
    n = rng.randint(min_len, max_len)
    return "".join(rng.choice(alphabet) for _ in range(n))


def _path(rng: random.Random) -> str:
    stem = _text(rng, min_len=1, max_len=14).strip(" ./\\") or "file"
    stem = stem.replace("\\", "_").replace(":", "_")
    suffix = rng.choice([".py", ".txt", ".md", ".json", ""])
    directory = rng.choice(["src", "tests", "docs", "pkg/sub dir", ""])
    if directory:
        return f"{directory}/{stem}{suffix}"
    return f"{stem}{suffix}"


def _add_hunk(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(1, 8)):
        lines.append("+" + _text(rng, max_len=60))
    return "\n".join(lines)


def _update_hunk(rng: random.Random) -> str:
    lines = []
    if rng.choice([True, False]):
        hint = _text(rng, min_len=1, max_len=24)
        lines.append(rng.choice(["@@", f"@@ {hint}"]))
    for _ in range(rng.randint(1, 8)):
        prefix = rng.choice([" ", "-", "+"])
        lines.append(prefix + _text(rng, max_len=60))
    if rng.randrange(20) == 0:
        lines.append("*** End of File")
    return "\n".join(lines)


def _codex_freeform_patch(rng: random.Random) -> str:
    """Generate patches accepted by the Codex Lark grammar shape.

    This intentionally avoids ``*** Move to`` because Hermes' existing V4A
    parser does not execute that Codex grammar variant today.
    """
    parts = ["*** Begin Patch"]
    for _ in range(rng.randint(1, 5)):
        kind = rng.choice(["add", "delete", "update"])
        path = _path(rng)
        if kind == "add":
            parts.extend([f"*** Add File: {path}", _add_hunk(rng)])
        elif kind == "delete":
            parts.append(f"*** Delete File: {path}")
        else:
            parts.extend([f"*** Update File: {path}", _update_hunk(rng)])
    parts.append("*** End Patch")
    if rng.choice([True, False]):
        return "\n".join(parts) + "\n"
    return "\n".join(parts)


def _json_args(value: dict) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _apply_hermes_replace(content: str, *, old_string: str, new_string: str, replace_all: bool) -> str:
    return content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)


def _lower_replace_to_apply_patch_call(
    *,
    call_id: str,
    path: str,
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
) -> dict:
    """Lower a Hermes replace call to apply_patch plus private intent.

    This is intentionally a test-local reference lowering.  It proves the
    equivalence boundary we want from production code: preserve original
    structured intent in sidecar metadata while sending the active model only
    concrete freeform patch text.
    """
    expected = _apply_hermes_replace(
        content,
        old_string=old_string,
        new_string=new_string,
        replace_all=replace_all,
    )
    removed_lines = content.split("\n")
    added_lines = expected.split("\n")
    patch = "\n".join([
        "*** Begin Patch",
        f"*** Update File: {path}",
        "@@",
        *(f"-{line}" for line in removed_lines),
        *(f"+{line}" for line in added_lines),
        "*** End Patch",
    ])
    original_args = {
        "mode": "replace",
        "path": path,
        "old_string": old_string,
        "new_string": new_string,
        "replace_all": replace_all,
    }
    return {
        "id": call_id,
        "call_id": call_id,
        "type": "apply_patch",
        "function": {
            "name": "apply_patch",
            "arguments": _json_args({"patch": patch}),
        },
        INTENT_KEY: {
            "canonical_tool": "patch",
            "canonical_arguments": original_args,
            "lowered_to": "apply_patch",
        },
    }


def _project_apply_patch_call_to_hermes_patch(tool_call: dict) -> dict:
    """Recover the canonical Hermes patch call through production projection."""
    projected = project_messages_for_patch_surface(
        [{"role": "assistant", "content": "", "tool_calls": [tool_call]}],
        patch_surface=PatchToolSurface.HERMES_PATCH,
    )
    return projected[0]["tool_calls"][0]


def test_custom_apply_patch_wire_roundtrip_preserves_patch_bytes_for_generated_patches():
    """Freeform apply_patch replay is lossless for raw patch text.

    The invariant is byte-for-byte preservation of the patch string when a
    Codex Responses custom tool call is normalized into Hermes history and
    replayed back to Responses input items.
    """
    rng = random.Random(SEED)
    for case_idx in range(CASES):
        patch = _codex_freeform_patch(rng)
        call_id = f"call_{case_idx}"
        response = SimpleNamespace(
            status="completed",
            output=[
                SimpleNamespace(
                    type="custom_tool_call",
                    id=f"ctc_{case_idx}",
                    call_id=call_id,
                    name="apply_patch",
                    input=patch,
                )
            ],
        )

        message, finish_reason = _normalize_codex_response(response)

        assert finish_reason == "tool_calls"
        assert message.tool_calls[0].type == "apply_patch"
        args = json.loads(message.tool_calls[0].function.arguments)
        assert args == {"patch": patch}, f"case={case_idx}"

        replay = _chat_messages_to_responses_input([
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "call_id": call_id,
                        "type": "apply_patch",
                        "function": {
                            "name": "apply_patch",
                            "arguments": message.tool_calls[0].function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "apply_patch",
                "tool_call_id": call_id,
                "content": "{\"success\": true}",
            },
        ])

        assert replay[0] == {
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": "apply_patch",
            "input": patch,
        }, f"case={case_idx}"
        assert replay[1]["type"] == "custom_tool_call_output"


def test_apply_patch_with_private_intent_projects_back_to_original_patch_replace_calls():
    """With sidecar intent, ``patch`` and ``apply_patch`` are representationally equivalent.

    The model-facing ``apply_patch`` call carries concrete patch text.  The
    private ``_hermes_patch_tool_intent`` field carries the original Hermes
    structured operation, including ``replace_all``.  Projection back to a
    non-native provider can therefore recover the exact original patch call.
    """
    rng = random.Random(SEED ^ 0x1A17E17)
    for case_idx in range(CASES):
        old_string = f"needle_{case_idx % 17}"
        new_string = f"replacement_{case_idx}"
        occurrences = rng.randint(1, 6)
        replace_all = rng.choice([False, True])
        path = _path(rng)
        chunks = []
        for i in range(occurrences):
            chunks.append(f"prefix {case_idx} {i}")
            chunks.append(old_string)
        chunks.append(f"suffix {case_idx}")
        content = "\n".join(chunks)

        tool_call = _lower_replace_to_apply_patch_call(
            call_id=f"call_{case_idx}",
            path=path,
            content=content,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )
        projected = _project_apply_patch_call_to_hermes_patch(tool_call)

        assert projected == {
            "id": f"call_{case_idx}",
            "call_id": f"call_{case_idx}",
            "type": "function",
            "function": {
                "name": "patch",
                "arguments": _json_args({
                    "mode": "replace",
                    "path": path,
                    "old_string": old_string,
                    "new_string": new_string,
                    "replace_all": replace_all,
                }),
            },
        }, f"case={case_idx}"


def test_lowered_apply_patch_is_state_equivalent_to_patch_replace_with_known_file_snapshot():
    """A replace call can be lowered to equivalent concrete patch text for a snapshot.

    This proves the execution equivalence part: for a known file state, the
    lowered patch encodes the same final file content as Hermes replace mode.
    The sidecar intent is what makes the conversion reversible as a tool call.
    """
    rng = random.Random(SEED ^ 0xE9EC71)
    for case_idx in range(CASES):
        old_string = f"token_{case_idx % 23}"
        new_string = f"value_{case_idx}"
        replace_all = rng.choice([False, True])
        occurrences = rng.randint(1, 8)
        lines = []
        for i in range(occurrences):
            lines.append(_text(rng, max_len=24))
            lines.append(old_string)
        lines.append(_text(rng, max_len=24))
        content = "\n".join(lines)

        tool_call = _lower_replace_to_apply_patch_call(
            call_id=f"call_{case_idx}",
            path=_path(rng),
            content=content,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )
        patch = json.loads(tool_call["function"]["arguments"])["patch"]

        removed = [
            line[1:]
            for line in patch.splitlines()
            if line.startswith("-") and not line.startswith("***")
        ]
        added = [
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("***")
        ]

        assert "\n".join(removed) == content
        assert "\n".join(added) == _apply_hermes_replace(
            content,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
        )


def test_apply_patch_execution_alias_delegates_to_hermes_patch_mode_losslessly(monkeypatch):
    """Execution is lossless for the patch-text subset.

    ``apply_patch(patch=P)`` must pass exactly ``P`` to Hermes'
    environment-aware ``patch(mode="patch")`` implementation.  This is the
    remote-terminal safety property: native-looking patch calls still execute
    through Hermes file ops for the active task environment.
    """
    captured = []

    def fake_patch_tool(**kwargs):
        captured.append(kwargs)
        return json.dumps({"success": True, "patch": kwargs["patch"]})

    monkeypatch.setattr("tools.file_tools.patch_tool", fake_patch_tool)

    rng = random.Random(SEED ^ 0xBAD5EED)
    for case_idx in range(CASES):
        patch = _codex_freeform_patch(rng)
        result = json.loads(apply_patch_tool(patch=patch, task_id=f"task-{case_idx}"))

        assert result == {"success": True, "patch": patch}
        assert captured[-1] == {
            "mode": "patch",
            "patch": patch,
            "task_id": f"task-{case_idx}",
        }


def test_project_messages_for_patch_surface_is_idempotent_under_repeated_application():
    """Projecting twice under the same surface equals projecting once.

    Locks in the invariant the codex build_kwargs path relies on: the
    projection layer must be safe to run more than once on the same history
    (e.g. when ``build_api_kwargs`` projects upstream and a transport projects
    again).  A regression here turns a no-op into a silent corruption.
    """
    rng = random.Random(SEED ^ 0x1D3777)
    surfaces = (
        PatchToolSurface.HERMES_PATCH,
        PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH,
    )
    for case_idx in range(CASES):
        patch = _codex_freeform_patch(rng)
        call_id = f"call_{case_idx}"
        # Mix both starting shapes — half stored as canonical ``patch``,
        # half stored as ``apply_patch`` with private intent — so the
        # idempotency check exercises both branches of _project_tool_call.
        if case_idx % 2 == 0:
            messages = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "patch",
                                "arguments": _json_args({"mode": "patch", "patch": patch}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "patch",
                    "tool_call_id": call_id,
                    "content": '{"success": true}',
                },
            ]
        else:
            messages = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "apply_patch",
                            "function": {
                                "name": "apply_patch",
                                "arguments": _json_args({"patch": patch}),
                            },
                            INTENT_KEY: {
                                "canonical_tool": "patch",
                                "canonical_arguments": {"mode": "patch", "patch": patch},
                                "lowered_to": "apply_patch",
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "name": "apply_patch",
                    "tool_call_id": call_id,
                    "content": '{"success": true}',
                },
            ]

        for surface in surfaces:
            once = project_messages_for_patch_surface(messages, patch_surface=surface)
            twice = project_messages_for_patch_surface(once, patch_surface=surface)
            assert twice == once, f"case={case_idx} surface={surface}"


def test_patch_mode_call_round_trips_through_codex_freeform_back_to_canonical_patch():
    """``patch(mode=patch)`` → CODEX_FREEFORM → HERMES_PATCH is lossless.

    For freeform-mode patches the model itself emits, no private intent is
    needed: the canonical Hermes arguments are recoverable from the projected
    apply_patch call alone.  This guards the no-intent return path the model
    will exercise on every Codex turn.
    """
    rng = random.Random(SEED ^ 0x517C77)
    for case_idx in range(CASES):
        patch = _codex_freeform_patch(rng)
        call_id = f"call_{case_idx}"
        canonical_args = {"mode": "patch", "patch": patch}
        canonical = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "patch",
                            "arguments": _json_args(canonical_args),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "patch",
                "tool_call_id": call_id,
                "content": '{"success": true}',
            },
        ]

        lowered = project_messages_for_patch_surface(
            canonical,
            patch_surface=PatchToolSurface.CODEX_FREEFORM_APPLY_PATCH,
        )
        assert lowered[0]["tool_calls"][0]["function"]["name"] == "apply_patch"
        assert lowered[1]["name"] == "apply_patch"

        recovered = project_messages_for_patch_surface(
            lowered,
            patch_surface=PatchToolSurface.HERMES_PATCH,
        )
        recovered_tc = recovered[0]["tool_calls"][0]

        assert recovered_tc["type"] == "function", f"case={case_idx}"
        assert recovered_tc["function"]["name"] == "patch", f"case={case_idx}"
        assert json.loads(recovered_tc["function"]["arguments"]) == canonical_args, f"case={case_idx}"
        assert recovered[1]["name"] == "patch", f"case={case_idx}"


def test_patch_replace_mode_needs_private_intent_for_lossless_call_shape_recovery():
    """Without private intent, ``replace_all`` cannot be recovered from patch text.

    Hermes ``patch`` has replace-mode fields that do not exist in the
    freeform Codex grammar.  The private intent sidecar is what preserves
    those fields when the provider-facing call is lowered to apply_patch.
    """
    replace_call = {
        "mode": "replace",
        "path": "src/example.py",
        "old_string": "value = 1",
        "new_string": "value = 2",
        "replace_all": True,
    }

    lowered_without_intent = {
        "id": "call_without_intent",
        "type": "apply_patch",
        "function": {
            "name": "apply_patch",
            "arguments": _json_args({
                "patch": (
                    "*** Begin Patch\n"
                    "*** Update File: src/example.py\n"
                    "@@\n"
                    "-value = 1\n"
                    "+value = 2\n"
                    "*** End Patch"
                )
            }),
        },
    }

    projected = _project_apply_patch_call_to_hermes_patch(lowered_without_intent)

    assert json.loads(projected["function"]["arguments"]) == {
        "mode": "patch",
        "patch": json.loads(lowered_without_intent["function"]["arguments"])["patch"],
    }
    assert projected["function"]["name"] == "patch"
    assert projected["function"]["arguments"] != _json_args(replace_call)
