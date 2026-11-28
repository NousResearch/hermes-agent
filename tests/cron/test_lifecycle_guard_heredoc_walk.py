"""The referenced-script walk sees the same inert-heredoc-masked view as the direct scan (#110422).

``_direct_lifecycle_scan`` masks provably-inert heredoc bodies (quoted delimiter, allowlisted
data consumer such as ``python3 - <<'PY'``), but ``_contains_unsafe_gateway_action`` walked
referenced scripts and ``sh -c`` payloads on the raw command, so a >1 MiB data path mentioned
inside the Python body failed closed as an oversized "script".
"""

import pytest

from cron.lifecycle_guard import contains_gateway_lifecycle_command_or_referenced_script as guard


def _big_file(tmp_path):
    path = tmp_path / "big_blob.bin"
    path.write_bytes(b"\0" * (2 * 1024 * 1024))
    return path


def test_inert_heredoc_body_path_not_walked_as_script(tmp_path):
    big = _big_file(tmp_path)
    command = f"python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('{big}').stat().st_size)\nPY"
    assert guard(command, cwd=str(tmp_path)) is False


def test_prefixed_inert_heredoc_body_path_not_walked_as_script(tmp_path):
    big = _big_file(tmp_path)
    command = (
        f"cd {tmp_path} && python3 - <<'PY'\n"
        f"from pathlib import Path\nprint(Path('{big}').stat().st_size)\nPY"
    )
    assert guard(command, cwd=str(tmp_path)) is False


@pytest.mark.parametrize(("opener", "body", "blocked"), [
    # A wrapper or a downstream text filter does not change who consumes the body: python3.
    ("timeout 600 python3 - <<'PY'", "d = open('{big}').read()", False),
    ("python3 - <<'PY' 2>&1 | grep -v x", "d = open('{big}').read()", False),
    ("nice -n 5 python3 - <<'PY' | head -n 3", "d = open('{big}').read()", False),
    # A shell owner, a shell downstream of the pipe, or a shell heredoc later on the line still is.
    ("timeout 5 bash <<'PY'", "bash {big}", True),
    ("python3 - <<'PY' | bash", "x = 1\nbash {big}", True),
    ("python3 - <<'PY' | grep x | sh", "x = 1\nbash {big}", True),
    ("python3 - <<'PY'; bash <<'SH'", "print(1)\nPY\nbash {big}\nSH", True),
    ("env -S 'bash -s' python3 <<'PY'", "bash {big}", True),
    ("cat <<'PY' | sort --compress-program=sh", "hermes gateway restart", True),
    ("timeout 600 python3 - <<'PY' 2>&1 | grep -v x", "import os\nos.system('hermes gateway restart')", True),
    # A later command on the line may run what the first one wrote, and a path-qualified filter is
    # whatever that file is (an earlier revision of this fix allowed all three).
    ("cat <<'PY' > run.sh; f=run.sh; bash \"$f\"", "hermes gateway restart", True),
    ("cat <<'PY' | cat > run.sh && bash run.sh", "hermes gateway restart", True),
    ("cat <<'PY' | ./grep", "hermes gateway restart", True),
])
def test_heredoc_body_belongs_to_its_owner_through_wrappers_and_pipes(tmp_path, opener, body, blocked):
    """A wrapped (`timeout 600 python3 - <<'PY'`) or filter-piped Python heredoc that only reads a
    >1 MiB data file is not refused as an unscannable script; shell consumers still are."""
    big = _big_file(tmp_path)
    command = f"{opener}\n{body.format(big=big)}\n" + ("" if body.endswith("SH") else "PY")
    assert guard(command, cwd=str(tmp_path)) is blocked


def test_allowlisted_name_function_heredoc_stays_visible(tmp_path):
    command = "python3() { bash; }; python3 <<'PY'\nhermes gateway restart\nPY"
    assert guard(command, cwd=str(tmp_path)) is True


def test_unquoted_heredoc_body_path_still_walked(tmp_path):
    """An expansion-capable body is not provably inert: the walk still sees it and fails closed."""
    big = _big_file(tmp_path).as_posix()
    assert guard(f"cat > /tmp/x <<EOF\n{big}\nEOF", cwd=str(tmp_path)) is True


class TestCodeConsumerBodyVisibleToLifecycleScan:
    """#127698: python/osascript heredoc bodies are executable payloads, not data. The direct
    lifecycle scan must not have them blanked, or an inline gateway command hides in the body
    while the scanner sees an empty heredoc."""

    def test_python_body_inline_launchctl_blocks(self):
        command = (
            "python3 - <<'PY'\n"
            "import os\n"
            'os.system("launchctl bootout gui/$(id -u)/ai.hermes.gateway")\n'
            "PY"
        )
        assert guard(command, cwd="/tmp") is True

    def test_osascript_do_shell_script_blocks(self):
        command = (
            "osascript <<'EOF'\n"
            'do shell script "launchctl bootout gui/501/ai.hermes.gateway"\n'
            "EOF"
        )
        assert guard(command, cwd="/tmp") is True

    def test_python_body_plain_data_does_not_block(self):
        """Visibility cuts both ways but costs nothing: a body that runs no lifecycle and names
        no path is still allowed (no new false positive from the strict view)."""
        command = "python3 - <<'PY'\nprint(sum(range(10)))\nPY"
        assert guard(command, cwd="/tmp") is False

    def test_data_sink_documentation_body_still_allowed(self):
        """#88336 semantics unchanged: a cat-owned body is documentation, not a command."""
        command = (
            "cat > /tmp/runbook.txt <<'EOF'\n"
            "a human can run: hermes gateway restart\n"
            "EOF"
        )
        assert guard(command, cwd="/tmp") is False


def test_inert_heredoc_body_script_path_still_read(tmp_path):
    """Masking hides the body from the *executed* view only: a lifecycle script named inside a
    Python body is still handed to ``os.system`` at runtime, so its contents must still be read."""
    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    command = f"python3 - <<'PY'\nimport os\nos.system('{script}')\nPY"
    assert guard(command, cwd=str(tmp_path)) is True


def test_mentioned_data_file_that_cannot_be_scanned_is_not_a_verdict(tmp_path, monkeypatch):
    """A file only MENTIONED in an inert body may exhaust the text budget (one >64 KiB line), pull
    in 64+ remote-read misses (a markdown table of paths) or be a live SQLite database: each is
    "nothing to scan", never a block (#113944). The same file *executed* still fails closed."""
    from cron import lifecycle_guard
    from hermes_cli.sqlite_safe_read import connect_tracked

    minified = tmp_path / "minified.json"
    minified.write_text("[" + "1," * 40000 + "1]", encoding="utf-8")
    notes = tmp_path / "notes.md"
    notes.write_text("\n".join(f"| /opt/frag{i}/tool-{i}.md | note |" for i in range(70)), encoding="utf-8")
    db = tmp_path / "state.db"
    conn = connect_tracked(db)
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_REMOTE_READS", 8)
    remote_misses: list[str] = []

    def remote(path: str):
        remote_misses.append(path)

    try:
        for data in (minified, notes, db):
            command = (f"cd {tmp_path.as_posix()} && python3 - <<'PY'\n"
                       f"t = open('{data.as_posix()}', encoding='utf-8').read()\nPY")
            assert guard(command, cwd=str(tmp_path), read_remote_script=remote) is False, data.name
        assert remote_misses  # the notes table was walked and its misses were bounded, not fatal
        unsafe, refusal = lifecycle_guard.scan_gateway_lifecycle(f"bash {minified.as_posix()}")
        assert unsafe is True and "budget" in refusal
        unsafe, refusal = lifecycle_guard.scan_gateway_lifecycle(f"bash {db.as_posix()}")
        assert unsafe is True and "SQLite" in refusal
    finally:
        conn.close()


def test_mentioned_script_with_lifecycle_command_still_blocks(tmp_path):
    """The lenient path only covers "could not scan": a mentioned script whose text IS a lifecycle
    command is still a positive verdict, and the refusal reason stays empty (it is not a scan failure)."""
    from cron import lifecycle_guard

    script = tmp_path / "restart.sh"
    script.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    command = f"cd {tmp_path} && python3 - <<'PY'\nimport os\nos.system('{script}')\nPY"
    assert lifecycle_guard.scan_gateway_lifecycle(command, cwd=str(tmp_path)) == (True, None)


@pytest.mark.parametrize("shape", ["oversized", "sqlite"])
def test_unscannable_own_cron_script_raises_named_refusal(tmp_path, shape):
    """A cron job whose OWN script is oversized or a live SQLite database fails closed with the
    named "could not scan" refusal, never the misattributed "contains a gateway lifecycle command"."""
    from cron.lifecycle_guard import GatewayLifecycleBlocked, check_gateway_lifecycle
    from hermes_cli.sqlite_safe_read import connect_tracked

    if shape == "oversized":
        script, conn = _big_file(tmp_path), None
    else:
        script = tmp_path / "state.db"
        conn = connect_tracked(script)
    try:
        with pytest.raises(GatewayLifecycleBlocked) as excinfo:
            check_gateway_lifecycle("run it", str(script))
    finally:
        if conn is not None:
            conn.close()
    message = str(excinfo.value)
    assert "could not scan" in message and str(script) in message
    assert "contains a gateway lifecycle command" not in message
