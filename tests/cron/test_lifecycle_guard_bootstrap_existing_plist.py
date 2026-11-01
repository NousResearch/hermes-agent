"""``launchctl bootstrap`` of an EXISTING non-gateway plist must be ALLOWED.

``contains_launchctl_submit_command`` treats ``submit`` and ``bootstrap``
label-independently because a NEW job's label is chosen by whoever writes the
command (#62891). That is correct for ``submit`` (the label is pure text) and
for ``bootstrap`` of a path that does not exist yet, but it is wrong for
``bootstrap`` of a plist that is ALREADY on disk: launchd reads the ``Label``
key out of that file, so the label is not attacker-chosen at all — it is a
readable fact about the job that will be registered.

Measured: reloading an unrelated daemon after a ``plutil -replace`` edit —

    launchctl bootout system/ai.hermes.fleetreview-router
    launchctl bootstrap system /Library/LaunchDaemons/ai.hermes.fleetreview-router.plist

— was refused even though ``ai.hermes.fleetreview-router`` is not a gateway
label, pushing operators onto the deprecated, label-gated ``launchctl load -w``.

Everything that cannot be READ stays blocked: a non-existent path, an
unreadable/oversized/malformed file, a directory, a FIFO, a path carrying an
unexpanded shell value, a plist whose ``Label`` IS a gateway label, a plist
whose ``ProgramArguments`` invoke a gateway entrypoint, and ``submit`` in
every shape.
"""

from __future__ import annotations

import os
import plistlib
from pathlib import Path

import pytest

from cron import lifecycle_guard
from cron.lifecycle_guard import contains_launchctl_submit_command


def _arg(path) -> str:
    """Render *path* the way a shell command would carry it.

    The guard tokenizes command text with ``shlex(..., posix=True)``
    (``cron/lifecycle_guard.py``), which consumes ``\\`` as an escape. A native
    Windows ``tmp_path`` (``C:\\Users\\...\\x.plist``) therefore reaches the
    guard as ``C:Usersx.plist`` — a path that does not exist — so the reader
    returns ``None``.

    That breaks this file in BOTH directions on Windows: the ALLOWED cases fail
    (the exemption never fires), and — less visibly — the BLOCKED cases pass for
    the wrong reason, fail-closed on an unreadable path rather than on the
    ``Label`` / argv fact each one exists to pin. Rendering every path argument
    POSIX-style keeps the assertions meaningful on every platform.
    """
    return Path(path).as_posix()


def _write_plist(path, payload) -> str:
    """Write *payload* as a binary plist and return its path as a shell argument."""
    with open(path, "wb") as handle:
        plistlib.dump(payload, handle, fmt=plistlib.FMT_BINARY)
    return _arg(path)


@pytest.fixture
def router_plist(tmp_path):
    """A real non-gateway daemon plist shape."""
    return _write_plist(
        tmp_path / "ai.hermes.fleetreview-router.plist",
        {
            "Label": "ai.hermes.fleetreview-router",
            "ProgramArguments": ["/usr/local/libexec/router-service", "--live"],
            "RunAtLoad": True,
        },
    )


class TestExistingNonGatewayPlistAllowed:
    def test_incident_command_shape_allowed(self, router_plist):
        assert not contains_launchctl_submit_command(
            f"sudo launchctl bootstrap system {router_plist}"
        )

    def test_bootout_then_bootstrap_sequence_allowed(self, router_plist):
        assert not contains_launchctl_submit_command(
            "sudo launchctl bootout system/ai.hermes.fleetreview-router && "
            f"sudo launchctl bootstrap system {router_plist}"
        )

    def test_gui_domain_form_allowed(self, router_plist):
        assert not contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {router_plist}"
        )

    def test_quoted_path_allowed(self, tmp_path):
        path = _write_plist(
            tmp_path / "com.example.cert watch.plist",
            {"Label": "com.example.cert-watch", "ProgramArguments": ["/bin/true"]},
        )
        assert not contains_launchctl_submit_command(
            f'launchctl bootstrap system "{path}"'
        )


class TestBootstrapStillBlocked:
    def test_gateway_label_plist_blocked(self, tmp_path):
        """Filename is neutral; only reading the Label catches this."""
        path = _write_plist(
            tmp_path / "reload-helper.plist",
            {"Label": "ai.hermes.gateway", "ProgramArguments": ["/bin/true"]},
        )
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_suffixed_gateway_label_blocked(self, tmp_path):
        path = _write_plist(
            tmp_path / "neutral.plist",
            {"Label": "hermes-gateway-foo", "ProgramArguments": ["/bin/true"]},
        )
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_gateway_entrypoint_in_program_arguments_blocked(self, tmp_path):
        """Neutral Label, but the job runs a gateway."""
        path = _write_plist(
            tmp_path / "com.example.svc.plist",
            {
                "Label": "com.example.svc",
                "ProgramArguments": [
                    "/opt/venv/bin/python",
                    "-m",
                    "hermes_cli.main",
                    "gateway",
                    "run",
                ],
            },
        )
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_gateway_entrypoint_in_program_blocked(self, tmp_path):
        """Same, via the scalar ``Program`` key rather than the argv list."""
        path = _write_plist(
            tmp_path / "com.example.other.plist",
            {"Label": "com.example.other", "Program": "/opt/bin/hermes-gateway"},
        )
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_gateway_entrypoint_split_across_argv_blocked(self, tmp_path):
        """A bare launcher matches no marker token; only the split rule sees it."""
        path = _write_plist(
            tmp_path / "com.example.bare.plist",
            {
                "Label": "com.example.bare",
                "ProgramArguments": ["/usr/local/bin/hermes", "gateway", "run"],
            },
        )
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_argv_runs_gateway_lifecycle_inline_blocked(self, tmp_path):
        """Neutral Label, non-entrypoint argv — but it KICKSTARTS a gateway.

        "Is this plist a gateway job?" is not the question the guard exists to
        answer. This plist is not a gateway job by any tell (neutral Label,
        argv is ``/bin/sh``), yet loading it restarts a gateway — the #62891
        laundering shape with a file instead of a ``submit`` line. It needs no
        root: a user-writable ``$TMPDIR`` path bootstrapped into ``gui/<uid>``
        is enough.
        """
        path = _write_plist(
            tmp_path / "com.example.helper.plist",
            {
                "Label": "com.example.helper",
                "ProgramArguments": [
                    "/bin/sh",
                    "-c",
                    "launchctl kickstart -k system/ai.hermes.gateway-main",
                ],
            },
        )
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_argv_runs_gateway_lifecycle_via_script_blocked(self, tmp_path):
        """Same laundering, one indirection deeper: argv names a SCRIPT.

        The inline-string witness alone would pass a fix that only scanned
        ``sh -c`` payloads; the lifecycle scanner's referenced-script walk is
        what closes this one.
        """
        script = tmp_path / "boot.sh"
        script.write_text(
            "#!/bin/sh\nlaunchctl bootout system/ai.hermes.gateway-main\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        path = _write_plist(
            tmp_path / "com.example.helper2.plist",
            {
                "Label": "com.example.helper2",
                "ProgramArguments": ["/bin/sh", _arg(script)],
            },
        )
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_argv_runs_gateway_lifecycle_direct_argv_blocked(self, tmp_path):
        """The lifecycle command IS the argv — no ``sh -c``, no script.

        ``ProgramArguments`` is already a split command line, so no single
        token contains a lifecycle command; only scanning the JOINED argv
        sees it. Without this witness a fix that scans tokens individually
        passes every other negative here.
        """
        path = _write_plist(
            tmp_path / "com.example.direct.plist",
            {
                "Label": "com.example.direct",
                "ProgramArguments": [
                    "/bin/launchctl",
                    "kickstart",
                    "-k",
                    "system/ai.hermes.gateway-main",
                ],
            },
        )
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_argv_bootstraps_another_plist_blocked(self, tmp_path, monkeypatch):
        """A plist whose argv bootstraps a plist re-enters the argv scan.

        Two plists can reference each other; the scan is depth-bounded and
        fails CLOSED at the bound, so the cycle terminates as a refusal.

        The verdict alone does NOT gate the bound — with the bound removed the
        cycle blows the Python stack and the guard's own ``except Exception``
        fallback still returns the same ``True``. So assert the WORK done: a
        bounded scan reads a handful of plists, an unbounded one reads >100
        before the interpreter gives out.
        """
        first = tmp_path / "com.example.chain-a.plist"
        second = tmp_path / "com.example.chain-b.plist"
        first_arg, second_arg = _arg(first), _arg(second)
        _write_plist(
            first,
            {
                "Label": "com.example.chain-a",
                "ProgramArguments": [
                    "/bin/sh",
                    "-c",
                    f"launchctl bootstrap gui/501 {second_arg}",
                ],
            },
        )
        _write_plist(
            second,
            {
                "Label": "com.example.chain-b",
                "ProgramArguments": [
                    "/bin/sh",
                    "-c",
                    f"launchctl bootstrap gui/501 {first_arg}",
                ],
            },
        )
        real_read = lifecycle_guard._read_plist_payload
        reads = []

        def counting_read(path):
            reads.append(path)
            return real_read(path)

        monkeypatch.setattr(lifecycle_guard, "_read_plist_payload", counting_read)
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {first_arg}"
        )
        assert len(reads) <= 8, f"unbounded plist recursion: {len(reads)} reads"
        # The per-thread depth counter must unwind to 0, or the NEXT scan in
        # this process starts pre-charged and fails closed on a benign plist.
        assert getattr(lifecycle_guard._PLIST_ARGV_SCAN_STATE, "depth", 0) == 0

    def test_nonexistent_path_blocked(self, tmp_path):
        """Nothing to read → the label is still attacker-chosen (#62891)."""
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {_arg(tmp_path / 'not-written-yet.plist')}"
        )

    def test_variable_in_path_blocked(self):
        assert contains_launchctl_submit_command(
            "launchctl bootstrap gui/501 $PLIST"
        )

    def test_variable_in_domain_argument_blocked(self, router_plist):
        """The plist is real and benign, but ANOTHER argument is unexpanded."""
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/$UID {router_plist}"
        )

    def test_command_substitution_in_path_blocked(self):
        assert contains_launchctl_submit_command(
            "launchctl bootstrap gui/501 $(mktemp).plist"
        )

    def test_directory_argument_blocked(self, tmp_path):
        """`bootstrap <domain> <dir>` loads every plist in the directory."""
        directory = tmp_path / "agents"
        directory.mkdir()
        directory = _arg(directory)
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {directory}"
        )

    def test_directory_named_like_a_plist_blocked(self, tmp_path):
        path = tmp_path / "com.example.dir.plist"
        path.mkdir()
        assert lifecycle_guard._read_plist_payload(path) is None
        path = _arg(path)
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    @pytest.mark.skipif(
        not hasattr(os, "mkfifo"), reason="special files need a POSIX filesystem"
    )
    def test_fifo_argument_blocked(self, tmp_path):
        """A FIFO named `.plist` is refused WITHOUT being read."""
        path = tmp_path / "com.example.fifo.plist"
        os.mkfifo(path)
        assert lifecycle_guard._read_plist_payload(path) is None
        path = _arg(path)
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_unparseable_file_blocked(self, tmp_path):
        path = tmp_path / "com.example.broken.plist"
        path.write_text("this is not a plist\n")
        path = _arg(path)
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_plist_without_label_blocked(self, tmp_path):
        path = _write_plist(
            tmp_path / "com.example.nolabel.plist",
            {"ProgramArguments": ["/bin/true"]},
        )
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {path}"
        )

    def test_oversized_file_blocked(self, tmp_path):
        path = tmp_path / "com.example.huge.plist"
        _write_plist(
            path,
            {
                "Label": "com.example.huge",
                "Padding": "x" * (lifecycle_guard._MAX_PLIST_BYTES + 1024),
            },
        )
        assert path.stat().st_size > lifecycle_guard._MAX_PLIST_BYTES
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {_arg(path)}"
        )

    def test_oversized_file_reader_returns_none(self, tmp_path):
        """Direct unit gate on the reader.

        ``_read_plist_payload`` carries THREE overlapping bounds (an
        ``st_size`` pre-check, a bounded read loop, a post-read length check).
        Any one alone produces this ``None``, so no single mutation of them
        can be killed — this asserts the property they jointly guarantee.
        """
        path = tmp_path / "com.example.huge2.plist"
        _write_plist(
            path,
            {
                "Label": "com.example.huge2",
                "Padding": "x" * (lifecycle_guard._MAX_PLIST_BYTES + 1024),
            },
        )
        assert lifecycle_guard._read_plist_payload(path) is None

    def test_reader_accepts_a_normal_plist(self, tmp_path, router_plist):
        """Positive control: the bound is not a blanket refusal."""
        from pathlib import Path

        payload = lifecycle_guard._read_plist_payload(Path(router_plist))
        assert payload is not None
        assert payload["Label"] == "ai.hermes.fleetreview-router"

    def test_mixed_readable_and_unreadable_blocked(self, router_plist, tmp_path):
        """Every plist argument must clear; one unreadable one blocks all."""
        assert contains_launchctl_submit_command(
            f"launchctl bootstrap gui/501 {router_plist} {_arg(tmp_path / 'gone.plist')}"
        )

    def test_no_plist_argument_blocked(self):
        assert contains_launchctl_submit_command("launchctl bootstrap gui/501")


class TestSubmitUnchanged:
    @pytest.mark.parametrize(
        "text",
        [
            "launchctl submit -l ai.hermes.gateway -- /bin/sh helper.sh",
            "launchctl submit -l ai.hermes.fleetreview-router -- /bin/sh helper.sh",
            "launchctl submit -l neutral-name -- /bin/sh helper.sh",
        ],
    )
    def test_submit_blocked_regardless_of_label(self, text):
        """`submit` never reads a file — its label proves nothing."""
        assert contains_launchctl_submit_command(text), f"Should match: {text!r}"

    def test_submit_with_a_readable_plist_argument_still_blocked(self, router_plist):
        """Naming a benign plist must not launder a `submit`."""
        assert contains_launchctl_submit_command(
            f"launchctl submit -l tmp -- /bin/cat {router_plist}"
        )
