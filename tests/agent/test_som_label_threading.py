"""SOM label threading into guarded-run confirmation — RFC #112639.

Behavior contracts: the backend stores per-snapshot (label, role) alongside the
existing snapshot tokens, cleared on the same lifecycle; the executor pulls them
from the backend's last capture so per-action predicates are sharp in
production without the caller threading anything in. Missing/empty snapshot
degrades to the ``window.exists`` fallback (fail open, never fail closed).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.guarded_run_executor import (
    guarded_run_readiness_stop,
    som_labels_of,
)
from tools.computer_use.readiness import ReadinessResult


def _tc(action=None, tool="computer_use", call_id=None, **args):
    payload = dict(args)
    if action is not None:
        payload["action"] = action
    return SimpleNamespace(
        id=call_id or f"call-{tool}-{action or 'x'}",
        function=SimpleNamespace(name=tool, arguments=json.dumps(payload)),
    )


class _Backend:
    """Fake backend with a sticky target, a canned verdict, and snapshot labels."""

    def __init__(self, status="satisfied", snapshot_labels=None):
        self._active_pid = 1234
        self._active_window_id = 5678
        self.seen_kwargs = None
        if snapshot_labels is not None:
            self._snapshot_labels = snapshot_labels
        self._result = ReadinessResult(status=status, detail="driver status: satisfied", duration_ms=5.8)

    def verify_readiness(self, **kwargs):
        self.seen_kwargs = kwargs
        return self._result


class TestSomLabelsOf:
    def test_returns_label_role_map(self):
        backend = SimpleNamespace(_snapshot_labels={2: ("Name", "AXTextField")})
        assert som_labels_of(backend) == {2: ("Name", "AXTextField")}

    def test_empty_snapshot_is_none(self):
        assert som_labels_of(SimpleNamespace(_snapshot_labels={})) is None

    def test_missing_attribute_is_none(self):
        assert som_labels_of(SimpleNamespace()) is None

    def test_blank_entries_are_dropped(self):
        backend = SimpleNamespace(_snapshot_labels={1: ("", ""), 2: ("Name", "")})
        assert som_labels_of(backend) == {2: ("Name", None)}

    def test_malformed_entries_are_skipped(self):
        backend = SimpleNamespace(_snapshot_labels={"x": ("Name", "AXButton"), 3: "nope", 4: ("OK", "AXButton")})
        assert som_labels_of(backend) == {4: ("OK", "AXButton")}


class TestBackendSnapshotLabels:
    def _mocked_backend(self):
        from tools.computer_use.cua_backend import CuaDriverBackend

        backend = CuaDriverBackend()
        backend._session = MagicMock()
        backend._session.supports_capability = lambda cap, tool=None: True
        return backend

    def test_capture_stores_labels_beside_tokens(self):
        backend = self._mocked_backend()
        windows_payload = {"windows": [{
            "app_name": "Demo", "pid": 9, "window_id": 1,
            "is_on_screen": True, "title": "", "z_index": 0,
        }]}

        def fake_call_tool(name, args):
            if name == "list_windows":
                return {"data": "", "images": [], "image_mime_types": [],
                        "structuredContent": windows_payload, "isError": False}
            if name == "get_window_state":
                return {
                    "data": "ok\n", "images": [], "image_mime_types": [],
                    "structuredContent": {"elements": [
                        {"element_index": 1, "role": "AXButton", "label": "OK", "element_token": "t1"},
                        {"element_index": 2, "role": "AXTextField", "label": "Name", "element_token": "t2"},
                    ]},
                    "isError": False,
                }
            return {"data": "", "images": [], "image_mime_types": [],
                    "structuredContent": None, "isError": False}

        backend._session.call_tool.side_effect = fake_call_tool
        backend.capture(mode="ax")

        assert backend._snapshot_labels == {1: ("OK", "AXButton"), 2: ("Name", "AXTextField")}
        assert backend._snapshot_tokens == {1: "t1", 2: "t2"}

    def test_target_change_disarms_labels(self):
        backend = self._mocked_backend()
        backend._snapshot_labels = {1: ("OK", "AXButton")}
        backend._set_active_target({"pid": 9, "window_id": 1})
        assert backend._snapshot_labels == {}

    def test_clear_active_target_forgets_labels(self):
        backend = self._mocked_backend()
        assert backend._snapshot_labels == {}


class TestThreadingIntoConfirmation:
    def _type_then_click(self):
        # Production shape: only the first call may carry ``element`` (a second
        # element use ends the run before that call), so the label pull happens
        # on the first step.
        return [
            _tc("type", element=2, text="hello", call_id="a"),
            _tc("click", call_id="b"),
        ]

    def test_backend_labels_reach_the_exact_expect_payload(self):
        backend = _Backend(snapshot_labels={2: ("Name", "AXTextField")})
        assert guarded_run_readiness_stop(self._type_then_click(), 0, backend) is None
        assert backend.seen_kwargs["expect"] == [{
            "element": {
                "selector": {"label_contains": "Name", "role": "AXTextField"},
                "exists": True,
                "value_equals": "hello",
            }
        }]

    def test_missing_snapshot_degrades_to_window_fallback(self):
        backend = _Backend()
        assert guarded_run_readiness_stop(self._type_then_click(), 0, backend) is None
        assert backend.seen_kwargs["expect"] == [{"window": {"exists": True}}]

    def test_empty_snapshot_degrades_to_window_fallback(self):
        backend = _Backend(snapshot_labels={})
        assert guarded_run_readiness_stop(self._type_then_click(), 0, backend) is None
        assert backend.seen_kwargs["expect"] == [{"window": {"exists": True}}]

    def test_explicit_som_labels_still_win(self):
        backend = _Backend(snapshot_labels={2: ("Name", "AXTextField")})
        guarded_run_readiness_stop(
            self._type_then_click(), 0, backend, som_labels={2: ("Other", "AXButton")}
        )
        assert backend.seen_kwargs["expect"][0]["element"]["selector"] == {
            "label_contains": "Other", "role": "AXButton"
        }
