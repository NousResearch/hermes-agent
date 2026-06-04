"""ACP test fixtures.

Autouse fixtures that apply to every test in tests/acp/:
- no_sensitive_path_check: neutralises the /private/var/folders sensitive-path
  guard so macOS pytest tmp_path (which lives there) doesn't block write_file
  and patch calls.  ACP tests exercise approval logic, not path-safety gating.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_sensitive_path_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch out _check_sensitive_path so macOS tmp_path is always writable."""
    monkeypatch.setattr("tools.file_tools._check_sensitive_path", lambda *_a, **_kw: None)
