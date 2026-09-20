#!/usr/bin/env python3
"""FORK-LOCAL: caproute demand attribution headers.

Our fleet routes through caproute, which decides model placement from
recorded demand. caproute can name the process on the other end of the
socket but not what a call was for; only the caller knows that. foxhound
spawns one `hermes chat` per task, so the parent exports CAPROUTE_* and
this process stamps every call with it.

The contract these pin: silent when unasked, bounded when asked, and
never carrying anything but ids.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock


class _Stub:
    """Just the surface the method touches."""

    def __init__(self, existing=None):
        self._client_kwargs = {}
        if existing is not None:
            self._client_kwargs["default_headers"] = existing


def _apply(stub, env):
    import run_agent
    with mock.patch.dict(os.environ, env, clear=False):
        for key in ("CAPROUTE_OPERATION", "CAPROUTE_JOB", "CAPROUTE_RUN_ID",
                    "CAPROUTE_WORK_ITEM_TYPE", "CAPROUTE_WORK_ITEM_ID",
                    "CAPROUTE_APP"):
            if key not in env:
                os.environ.pop(key, None)
        run_agent.AIAgent._apply_caproute_attribution_headers(stub)
    return stub._client_kwargs.get("default_headers")


class CaprouteAttributionTests(unittest.TestCase):
    def test_no_env_means_no_headers_at_all(self):
        """Upstream behaviour must be untouched when nobody asked."""
        self.assertIsNone(_apply(_Stub(), {}))

    def test_no_env_does_not_disturb_existing_headers(self):
        existing = {"User-Agent": "curl/8.7.1"}
        self.assertEqual(_apply(_Stub(existing), {}), existing)

    def test_a_parent_job_is_stamped_on_the_request(self):
        got = _apply(_Stub(), {"CAPROUTE_OPERATION": "execute",
                               "CAPROUTE_RUN_ID": "run-7"})
        self.assertEqual(got["X-Caproute-Operation"], "execute")
        self.assertEqual(got["X-Caproute-Run-Id"], "run-7")
        self.assertEqual(got["X-Caproute-App"], "hermes")
        self.assertEqual(got["X-Caproute-Pid"], str(os.getpid()))

    def test_the_app_name_can_be_overridden_by_the_parent(self):
        got = _apply(_Stub(), {"CAPROUTE_OPERATION": "plan",
                               "CAPROUTE_APP": "foxhound"})
        self.assertEqual(got["X-Caproute-App"], "foxhound")

    def test_existing_headers_survive_alongside(self):
        got = _apply(_Stub({"User-Agent": "curl/8.7.1"}),
                     {"CAPROUTE_JOB": "nightly"})
        self.assertEqual(got["User-Agent"], "curl/8.7.1")
        self.assertEqual(got["X-Caproute-Job"], "nightly")

    def test_control_characters_are_stripped(self):
        """These become HTTP headers and land in a router's log."""
        got = _apply(_Stub(), {"CAPROUTE_JOB": "nig\r\nInjected: yes\thtly"})
        self.assertNotIn("\r", got["X-Caproute-Job"])
        self.assertNotIn("\n", got["X-Caproute-Job"])
        self.assertNotIn("\t", got["X-Caproute-Job"])

    def test_values_are_bounded(self):
        got = _apply(_Stub(), {"CAPROUTE_RUN_ID": "x" * 500})
        self.assertEqual(len(got["X-Caproute-Run-Id"]), 160)

    def test_an_empty_value_is_not_a_request_for_headers(self):
        self.assertIsNone(_apply(_Stub(), {"CAPROUTE_OPERATION": "   "}))

    def test_app_alone_does_not_trigger_stamping(self):
        """caproute already knows we are hermes; only context is new."""
        self.assertIsNone(_apply(_Stub(), {"CAPROUTE_APP": "foxhound"}))

    def test_every_field_maps_to_its_header(self):
        got = _apply(_Stub(), {
            "CAPROUTE_OPERATION": "review", "CAPROUTE_JOB": "j",
            "CAPROUTE_RUN_ID": "r", "CAPROUTE_WORK_ITEM_TYPE": "task",
            "CAPROUTE_WORK_ITEM_ID": "T12"})
        self.assertEqual(got["X-Caproute-Operation"], "review")
        self.assertEqual(got["X-Caproute-Job"], "j")
        self.assertEqual(got["X-Caproute-Run-Id"], "r")
        self.assertEqual(got["X-Caproute-Work-Item-Type"], "task")
        self.assertEqual(got["X-Caproute-Work-Item-Id"], "T12")


if __name__ == "__main__":
    unittest.main()
