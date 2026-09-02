"""The ancestor exclusion in ``_scan_gateway_pids`` must not hide a real gateway.

``hermes update --gateway`` is spawned BY the gateway when ``/update`` is issued
from a messaging platform, so the gateway sits in the updater's own ancestor
chain. The blanket ancestor exclusion (added for #13242, to stop ``hermes
gateway status`` counting the CLI that invoked it) therefore hid the gateway
from the update pause machinery: nothing was paused, the update mutated the venv
while the gateway still held ``.pyd`` files, and the venv-holder guard aborted
with "Other Hermes processes are running from this install's venv" (#87594).

The exclusion is now gated on the command line: an ancestor that looks like a
gateway runtime stays visible, everything else is still suppressed.

The Windows arm is exercised here because that is where the reported failure
lives (``.pyd`` locking is what makes the pause mandatory), and it is stubbed
end to end so these run on any host.
"""

from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.gateway as gateway_mod

_GATEWAY_CMD = "C:/h/venv/Scripts/pythonw.exe -m hermes_cli.main gateway run --replace"
_UPDATER_CMD = "C:/h/venv/Scripts/python.exe -m hermes_cli.main update --yes --gateway"
_STATUS_CMD = "C:/h/venv/Scripts/python.exe -m hermes_cli.main gateway status"
_RESTART_CMD = "C:/h/venv/Scripts/pythonw.exe -m hermes_cli.main gateway restart"

GATEWAY_PID = 14112
UPDATER_PID = 22001


def _wmic_listing(entries: dict[int, str]) -> str:
    """Render ``wmic process get ProcessId,CommandLine /FORMAT:LIST`` output."""
    blocks = [f"CommandLine={cmd}\nProcessId={pid}\n" for pid, cmd in entries.items()]

    return "\n".join(blocks)


def _scan(
    entries: dict[int, str],
    ancestors: set[int],
    exclude: set[int] | None = None,
    include_restart_managers: bool = False,
):
    """Run the Windows scan arm against a stubbed process table."""
    result = MagicMock(returncode=0, stdout=_wmic_listing(entries))

    with (
        patch("hermes_cli.gateway.is_windows", return_value=True),
        patch("hermes_cli.gateway.shutil.which", return_value="C:/Windows/wmic.exe"),
        patch("hermes_cli.gateway.subprocess.run", return_value=result),
        patch("hermes_cli.gateway._get_ancestor_pids", return_value=ancestors),
        # Stub-collapsing is a separate Windows concern (venv launcher pairs)
        # and would need a live process table; identity keeps it out of the way.
        patch(
            "hermes_cli.gateway._filter_venv_launcher_stubs", side_effect=lambda p: p
        ),
    ):
        return gateway_mod._scan_gateway_pids(
            exclude or set(),
            all_profiles=True,
            include_restart_managers=include_restart_managers,
        )


class TestSuppressionPredicateCannotDiverge:
    """The suppression predicate must never be narrower than the include one.

    Raised in review: ``_suppressed_as_ancestor`` gates on
    ``looks_like_gateway_runtime_command_line`` while the include decision uses
    ``_matches_gateway_runtime``, so a command matching the include predicate
    but not the suppression one would still be hidden and #87594 would return
    in that shape. That set is empty today, and these pin it that way rather
    than leaving the argument in a comment.
    """

    def test_strict_matcher_is_a_subset_of_the_runtime_matcher(self):
        """``run`` is in ``{run, restart}``, so strict implies runtime."""
        from gateway.status import (
            looks_like_gateway_command_line,
            looks_like_gateway_runtime_command_line,
        )

        commands = [
            _GATEWAY_CMD,
            _UPDATER_CMD,
            _STATUS_CMD,
            _RESTART_CMD,
            "python -m hermes_cli.main gateway run",
            "python -m hermes_cli.main gateway dashboard",
            "hermes gateway run --replace",
            "hermes gateway restart",
            "python -m tui_gateway",
            "",
        ]

        for command in commands:
            if looks_like_gateway_command_line(command):
                assert looks_like_gateway_runtime_command_line(command), (
                    f"{command!r} is included by the strict matcher but would "
                    "be suppressed as an ancestor: the two predicates have "
                    "diverged and #87594 is reachable again"
                )

    def test_restart_hosted_runtime_ancestor_stays_visible(self):
        """The other accepted subcommand form, which the first tests omitted.

        A no-supervisor ``gateway restart`` runs ``run_gateway()`` in its own
        process, so it hosts the runtime and holds the ``.pyd`` files while its
        argv still says ``restart``. Windows takes exactly this path, because
        ``include_restart_managers`` is ``not supports_systemd_services()``.

        This does not discriminate between the two candidate gatings: where
        ``include_restart_managers`` is False a restart process is not included
        at all, and where it is True both predicates accept it. The difference
        is robustness under future edits, which
        ``test_strict_matcher_is_a_subset_of_the_runtime_matcher`` is what
        actually guards.
        """
        pids = _scan(
            {GATEWAY_PID: _RESTART_CMD, UPDATER_PID: _UPDATER_CMD},
            ancestors={UPDATER_PID, GATEWAY_PID, 4},
            include_restart_managers=True,
        )

        assert GATEWAY_PID in pids
        assert UPDATER_PID not in pids


class TestAncestorGatewayStaysVisible:
    def test_gateway_that_spawned_us_is_reported(self):
        """The #87594 failure: updater spawned by the gateway saw no gateway."""
        pids = _scan(
            {GATEWAY_PID: _GATEWAY_CMD, UPDATER_PID: _UPDATER_CMD},
            ancestors={UPDATER_PID, GATEWAY_PID, 4},
        )

        assert GATEWAY_PID in pids, (
            "a real `gateway run` in our ancestor chain is the process the "
            "update pause path exists to find"
        )

    def test_updater_ancestor_is_not_reported(self):
        """The updater is in its own ancestor chain and is not a gateway."""
        pids = _scan(
            {GATEWAY_PID: _GATEWAY_CMD, UPDATER_PID: _UPDATER_CMD},
            ancestors={UPDATER_PID, GATEWAY_PID, 4},
        )

        assert UPDATER_PID not in pids

    def test_non_gateway_ancestor_is_still_excluded(self):
        """#13242 must keep holding: the invoking CLI is not a gateway."""
        pids = _scan({UPDATER_PID: _UPDATER_CMD}, ancestors={UPDATER_PID})

        assert pids == []

    def test_gateway_status_ancestor_is_still_excluded(self):
        """`gateway status` is the original #13242 case, verbatim."""
        pids = _scan({4242: _STATUS_CMD}, ancestors={4242})

        assert pids == []

    def test_caller_supplied_exclusions_stay_unconditional(self):
        """``exclude_pids`` belongs to the caller and outranks the matcher."""
        pids = _scan(
            {GATEWAY_PID: _GATEWAY_CMD},
            ancestors=set(),
            exclude={GATEWAY_PID},
        )

        assert pids == []

    def test_unrelated_gateway_is_unaffected(self):
        """A gateway that is not an ancestor was never in question."""
        pids = _scan({GATEWAY_PID: _GATEWAY_CMD}, ancestors={UPDATER_PID})

        assert pids == [GATEWAY_PID]


@pytest.mark.linux_only
class TestAncestorGatewayViaProc:
    """Same contract on the /proc arm, which is what Linux hosts take."""

    def _proc_scan(self, entries: dict[int, str], ancestors: set[int]):
        def _isdir(path):
            return str(path) == "/proc"

        def _listdir(path):
            if str(path) == "/proc":
                return [str(pid) for pid in entries]
            raise FileNotFoundError(path)

        def _open(path, mode="r", **kwargs):
            path_str = str(path)
            if "/cmdline" not in path_str:
                raise FileNotFoundError(path)
            pid = int(path_str.split("/proc/")[1].split("/")[0])
            handle = MagicMock()
            handle.read.return_value = (
                entries.get(pid, "").encode("utf-8").replace(b" ", b"\x00")
            )
            handle.__enter__ = lambda s: s
            handle.__exit__ = MagicMock(return_value=False)

            return handle

        with (
            patch("os.path.isdir", side_effect=_isdir),
            patch("os.listdir", side_effect=_listdir),
            patch("builtins.open", side_effect=_open),
            patch("hermes_cli.gateway._get_ancestor_pids", return_value=ancestors),
            patch("subprocess.run"),
        ):
            return gateway_mod._scan_gateway_pids(set(), all_profiles=True)

    def test_gateway_that_spawned_us_is_reported(self):
        pids = self._proc_scan(
            {
                GATEWAY_PID: "python -m hermes_cli.main gateway run",
                UPDATER_PID: "python -m hermes_cli.main update --yes --gateway",
            },
            ancestors={UPDATER_PID, GATEWAY_PID, 1},
        )

        assert GATEWAY_PID in pids
        assert UPDATER_PID not in pids

    def test_non_gateway_ancestor_is_still_excluded(self):
        pids = self._proc_scan(
            {UPDATER_PID: "python -m hermes_cli.main update --yes --gateway"},
            ancestors={UPDATER_PID},
        )

        assert pids == []
