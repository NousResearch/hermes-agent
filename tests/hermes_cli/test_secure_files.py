"""Regression tests for sensitive file writers."""

from __future__ import annotations

import json
import os
import stat
import sys
from unittest.mock import patch

import pytest

from hermes_cli.secure_files import write_sensitive_json


pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX mode bits not enforced on Windows",
)


def test_write_sensitive_json_writes_0o600_under_permissive_umask(tmp_path):
    """Token-bearing JSON must not inherit umask-default 0o644."""
    path = tmp_path / ".anthropic_oauth.json"

    old_umask = os.umask(0o022)
    try:
        write_sensitive_json(path, {"accessToken": "access-secret", "refreshToken": "refresh-secret"})
    finally:
        os.umask(old_umask)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"sensitive file mode 0o{mode:o} != 0o600"
    assert json.loads(path.read_text(encoding="utf-8"))["refreshToken"] == "refresh-secret"


def test_write_sensitive_json_restricts_preexisting_permissive_file(tmp_path):
    """Overwriting an older 0o644 token file must tighten it to 0o600."""
    path = tmp_path / ".anthropic_oauth.json"
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o644)

    write_sensitive_json(path, {"refreshToken": "refresh-secret"})

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"sensitive file mode 0o{mode:o} != 0o600"
    assert json.loads(path.read_text(encoding="utf-8"))["refreshToken"] == "refresh-secret"


def test_write_sensitive_json_uses_os_open_with_0o600_mode(tmp_path):
    """Temp files must be created with explicit owner-only permissions."""
    observed_opens: list[tuple[str, int, int]] = []
    real_os_open = os.open

    def spying_os_open(path, flags, mode=0o777, *args, **kwargs):
        observed_opens.append((str(path), flags, mode))
        return real_os_open(path, flags, mode, *args, **kwargs)

    with patch.object(os, "open", spying_os_open):
        write_sensitive_json(tmp_path / ".anthropic_oauth.json", {"refreshToken": "refresh-secret"})

    temp_opens = [(p, fl, m) for (p, fl, m) in observed_opens if ".anthropic_oauth.json.tmp" in p]
    assert temp_opens, f"os.open was never called for the temp file; observed={observed_opens!r}"
    for path, flags, mode in temp_opens:
        assert flags & os.O_CREAT, f"temp open missing O_CREAT: path={path}"
        assert flags & os.O_EXCL, f"temp open missing O_EXCL: path={path}"
        assert mode == stat.S_IRUSR | stat.S_IWUSR
