"""Instance scoping of ``_argv_scoped_to_other_home`` across platforms.

A second gateway under another ``.hermes`` home must be recognized as a
different instance from its argv tokens — including Windows absolute paths
(drive-letter, forward-slash, UNC), which a ``"/"``-prefix check never sees.
Each arm runs natively on its own host; neither fakes ``sys.platform``.
"""

from __future__ import annotations

import pytest

from hermes_state_holders import _argv_scoped_to_other_home


@pytest.mark.windows_only
class TestArgvScopedToOtherHomeWindows:
    def test_windows_absolute_tokens_scope_to_other_home(self, tmp_path):
        db_path = tmp_path / "mine" / "state.db"
        # A drive-letter token naming another home proves a different instance.
        assert (
            _argv_scoped_to_other_home(
                ["hermes", "--db=C:\\Users\\other\\.hermes\\state.db"], db_path
            )
            is True
        )
        # A token naming our own db keeps the fail-closed suspicion off.
        assert _argv_scoped_to_other_home(["hermes", str(db_path)], db_path) is False


@pytest.mark.linux_only
class TestArgvScopedToOtherHomePosix:
    def test_posix_absolute_tokens_scope_to_other_home(self, tmp_path):
        db_path = tmp_path / "mine" / "state.db"
        # A POSIX token naming another home proves a different instance.
        assert (
            _argv_scoped_to_other_home(
                ["hermes", "--db=/home/other/.hermes/state.db"], db_path
            )
            is True
        )
        # A token naming our own db keeps the fail-closed suspicion off.
        assert _argv_scoped_to_other_home(["hermes", str(db_path)], db_path) is False
