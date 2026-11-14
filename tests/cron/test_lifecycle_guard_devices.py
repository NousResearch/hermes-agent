"""The referenced-script walk treats the null device as empty, other devices stay fail-closed (#125378)."""

import os
import sys

import pytest

from cron.lifecycle_guard import scan_gateway_lifecycle

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX device paths")


def test_null_device_is_nothing_to_scan(tmp_path):
    tool = tmp_path / "demo-tool"
    tool.write_text('#!/usr/bin/env python3\nimport sys\nprint("/dev/null" in sys.argv[1:])\n', encoding="utf-8")
    tool.chmod(0o755)
    data = tmp_path / "theme.css"
    data.write_text("a\n", encoding="utf-8")

    assert scan_gateway_lifecycle(f"diff <(sed -n 1,2p {data}) /dev/null | head") == (False, None)
    assert scan_gateway_lifecycle(f"{tool} go") == (False, None)


@pytest.mark.skipif(not os.path.exists("/dev/zero"), reason="needs /dev/zero")
def test_other_character_device_still_fails_closed():
    assert scan_gateway_lifecycle("bash /dev/zero") == (True, "`/dev/zero` is not a regular file")
