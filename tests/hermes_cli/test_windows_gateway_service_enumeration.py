"""An SCM service whose config cannot be read must not abort the whole enumeration.

Regression: `QueryServiceConfigW` fails with ``WinError 15100``
(``ERROR_MUI_FILE_NOT_FOUND``) for a service whose ``DisplayName``/``Description`` are indirect
strings pointing at a MUI the machine never installed — psutil surfaces that as a plain
``OSError``, not ``AccessDenied``. ``find_windows_gateway_services`` only tolerated
``AccessDenied`` while probing ``binpath`` for ownership, so that one unrelated service became
``RuntimeError("SCM service enumeration failed")``, which aborts ``hermes update`` on Windows at
the fleet-pause step (before the pull, so every run fails, forever).

A service this process cannot inspect is one it could not ``sc stop`` either: never Hermes's,
never a reason to abort.
"""

from __future__ import annotations

import pytest

from hermes_cli import gateway


class _UnreadableService:
    """psutil enumerates the service and answers ``name()``, but its config cannot be read."""

    def name(self) -> str:
        return "McpManagementService"

    def binpath(self) -> str:
        raise OSError(15100, "The resource loader cannot find MUI file.")


class _FakePsutil:
    @staticmethod
    def win_service_iter():
        return [_UnreadableService()]


@pytest.mark.windows_only
def test_unreadable_service_config_is_skipped_not_fatal():
    assert gateway.find_windows_gateway_services(psutil_module=_FakePsutil, profile_processes=[]) == []
