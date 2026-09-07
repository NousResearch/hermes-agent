"""Docs drift guard: the hooks.md catalog rows list every kwarg their call site sends.

The catalog header promises "the exact event-specific fields supplied by each call site", so the
expected sets are derived from the code (``_CallIds.hook_kwargs()``, ``_fire_approval_hook``), not
hard-coded: the test passes on any tree where docs and code agree.
"""

import re
from pathlib import Path

import pytest

HOOKS_MD = Path(__file__).resolve().parents[2] / "website" / "docs" / "user-guide" / "features" / "hooks.md"


def _catalog_row_fields(hook: str) -> set:
    for line in HOOKS_MD.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"| `{hook}` |") or line.startswith(f"| [`{hook}`]"):
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            return set(re.findall(r"`([A-Za-z_]+)`", cols[3]))
    raise AssertionError(f"no catalog row for {hook}")


@pytest.mark.parametrize("hook", ["pre_tool_call", "post_tool_call", "transform_tool_result"])
def test_tool_hook_rows_list_call_ids(hook):
    from model_tools import _CallIds

    sent = set(_CallIds().hook_kwargs())  # splatted into every tool hook by model_tools
    missing = sent - _catalog_row_fields(hook)
    assert not missing, sorted(missing)


@pytest.mark.parametrize("hook", ["pre_approval_request", "post_approval_response"])
def test_approval_hook_rows_list_fire_approval_hook_defaults(monkeypatch, hook):
    import hermes_cli.lifecycle as lifecycle
    from tools import approval_context as ctx

    seen = {}
    monkeypatch.setattr(lifecycle, "invoke_hook", lambda name, **kw: seen.update(kw))
    ctx._fire_approval_hook(hook, command="rm -rf /x", description="d", pattern_key="rm_rf",
                            pattern_keys=["rm_rf"], session_key="s", surface="gateway",
                            **({"choice": "once"} if hook == "post_approval_response" else {}))
    assert seen, "hook not dispatched"
    missing = set(seen) - _catalog_row_fields(hook)
    assert not missing, sorted(missing)


def test_decided_by_prose_matches_prompted_surface_contract():
    from tools.approval_gateway_wait import _ApprovalEntry

    text = HOOKS_MD.read_text(encoding="utf-8")
    if "decided_by" in getattr(_ApprovalEntry, "__slots__", ()):
        # The gateway (prompted) surface sends decided_by from Discord buttons and /approve, /deny.
        assert "absent on prompted surfaces" not in text
