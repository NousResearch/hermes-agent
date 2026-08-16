"""Work-budget regressions for the gateway lifecycle guard (#78398).

The referenced-script walk is security-sensitive, so exhausting a budget must
fail closed.  These tests keep the limits tiny and deterministic; production
limits remain generous enough for ordinary wrapper chains.
"""

from __future__ import annotations

import pytest

import cron.lifecycle_guard as lifecycle_guard


def _remote_command(*paths: str) -> str:
    return "\n".join(f"sh {path}" for path in paths)


def test_unique_path_budget_bounds_remote_reads_and_fails_closed(monkeypatch):
    """A wide script graph must not trigger an unbounded number of reads."""
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_PATHS",
        3,
        raising=False,
    )
    paths = tuple(f"/remote/safe-{index}.sh" for index in range(10))
    reads: list[str] = []

    def read_remote_script(path: str) -> str:
        reads.append(path)
        return "printf 'safe\\n'\n"

    verdict = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
        _remote_command(*paths),
        read_remote_script=read_remote_script,
    )

    assert verdict is True
    assert reads == list(paths[:3])


def test_repeated_path_does_not_spend_unique_path_budget(monkeypatch):
    """The existing visited-set dedupe stays effective under the new budget."""
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_PATHS",
        1,
        raising=False,
    )
    path = "/remote/reused-safe.sh"
    reads: list[str] = []

    def read_remote_script(candidate: str) -> str:
        reads.append(candidate)
        return "printf 'safe\\n'\n"

    verdict = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
        _remote_command(*(path for _ in range(10))),
        read_remote_script=read_remote_script,
    )

    assert verdict is False
    assert reads == [path]


def test_cumulative_text_budget_bounds_recursive_scan(monkeypatch):
    """Safe files cannot cumulatively feed unlimited text back into shlex."""
    paths = ("/r/a", "/r/b", "/r/c")
    command = _remote_command(*paths)
    script = "printf 'one bounded script\\n'\n"
    # Exactly enough for the root command and one remote script. The second
    # referenced path must fail closed before another remote read is attempted.
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_BYTES",
        len(command.encode("utf-8")) + len(script.encode("utf-8")),
        raising=False,
    )
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_PATHS",
        10,
        raising=False,
    )
    reads: list[str] = []

    def read_remote_script(path: str) -> str:
        reads.append(path)
        return script

    verdict = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
        command,
        read_remote_script=read_remote_script,
    )

    assert verdict is True
    assert reads == [paths[0]]


def test_line_budget_fails_closed_before_tokenizing_every_line(monkeypatch):
    """Thousands of short lines were the one-lexer-per-line CPU blow-up."""
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_LINES",
        2,
        raising=False,
    )

    verdict = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
        "sh /remote/many-lines.sh",
        read_remote_script=lambda _path: "printf one\nprintf two\nprintf three",
    )

    assert verdict is True


def test_depth_zero_byte_and_line_byte_limits_allow_exact_and_reject_plus_one(
    monkeypatch,
):
    """The root is bounded before shlex, with inclusive byte limits."""
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_LINE_BYTES",
        8,
        raising=False,
    )
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_BYTES",
        8,
        raising=False,
    )

    assert (
        lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script("x" * 8)
        is False
    )

    def fail_if_scanned(_command: str) -> bool:
        raise AssertionError("over-budget root reached the direct scanner")

    monkeypatch.setattr(lifecycle_guard, "_direct_lifecycle_scan", fail_if_scanned)
    assert (
        lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script("x" * 9)
        is True
    )


def test_depth_zero_line_limit_allows_exact_and_rejects_plus_one(monkeypatch):
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_LINES",
        2,
        raising=False,
    )

    assert (
        lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
            "one\ntwo"
        )
        is False
    )
    assert (
        lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
            "one\ntwo\nthree"
        )
        is True
    )


def test_crlf_continuation_line_budget_blocks_before_shlex(monkeypatch):
    """CRLF continuations cannot join compliant lines into one huge token."""
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_LINE_BYTES",
        8,
        raising=False,
    )

    def fail_if_tokenized(*_args, **_kwargs):
        raise AssertionError("over-budget CRLF continuation reached shlex")

    monkeypatch.setattr(lifecycle_guard.shlex, "shlex", fail_if_tokenized)
    assert (
        lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(
            "xxxxx\\\r\nxxxx"
        )
        is True
    )


@pytest.mark.parametrize(
    "command",
    [
        "hermes gate\\\nway restart",
        "hermes gate\\\r\nway restart",
        "launchctl sub\\\r\nmit -l com.foo -- /tmp/helper",
    ],
)
def test_lf_and_crlf_continuations_cannot_split_blocked_keywords(command):
    assert (
        lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script(command)
        is True
    )


def test_depth_zero_budget_counts_utf8_bytes(monkeypatch):
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_BYTES",
        4,
        raising=False,
    )
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_LINE_BYTES",
        4,
        raising=False,
    )

    assert (
        lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script("éé")
        is False
    )
    assert (
        lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script("ééé")
        is True
    )


def test_empty_prompt_does_not_spend_a_synthetic_delimiter_byte(
    monkeypatch,
    tmp_path,
):
    """A script exactly at the root limit is allowed; one extra byte blocks."""
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_BYTES",
        8,
        raising=False,
    )
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_LINE_BYTES",
        8,
        raising=False,
    )
    script = tmp_path / "long-line.sh"
    script.write_text("x" * 8, encoding="utf-8")

    lifecycle_guard.check_gateway_lifecycle("", str(script))

    script.write_text("x" * 9, encoding="utf-8")

    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked):
        lifecycle_guard.check_gateway_lifecycle("", str(script))


@pytest.mark.parametrize("name", ["job.PY", "job"])
def test_scheduler_python_classification_skips_shell_walk_and_is_bounded(
    monkeypatch,
    tmp_path,
    name,
):
    """Uppercase and extensionless scripts use the bounded direct-regex path."""
    (tmp_path / "child.sh").write_text(
        "hermes gateway restart\n",
        encoding="utf-8",
    )
    script = tmp_path / name
    source = '"./child.sh"\n'
    script.write_text(source, encoding="utf-8")
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_BYTES",
        len(source.encode("utf-8")),
        raising=False,
    )
    monkeypatch.setattr(
        lifecycle_guard,
        "_MAX_LIFECYCLE_SCAN_LINE_BYTES",
        len(source.encode("utf-8")),
        raising=False,
    )

    lifecycle_guard.check_gateway_lifecycle("", str(script))

    script.write_text(f"{source}x", encoding="utf-8")
    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked):
        lifecycle_guard.check_gateway_lifecycle("", str(script))


@pytest.mark.parametrize("name", ["job.PY", "job"])
@pytest.mark.parametrize(
    "prompt",
    [
        "launchctl submit -l com.foo -- /tmp/helper",
        "launchctl bootstrap gui/501 /tmp/com.foo.plist",
    ],
)
def test_python_classified_script_blocks_prompt_launchctl_without_scanning_source(
    monkeypatch,
    tmp_path,
    name,
    prompt,
):
    script = tmp_path / name
    script.write_text("print('safe')\n", encoding="utf-8")
    scanned: list[str] = []
    original_scan = lifecycle_guard.contains_launchctl_submit_command

    def track_launchctl_scan(text: str) -> bool:
        scanned.append(text)
        return original_scan(text)

    monkeypatch.setattr(
        lifecycle_guard,
        "contains_launchctl_submit_command",
        track_launchctl_scan,
    )

    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked):
        lifecycle_guard.check_gateway_lifecycle(prompt, str(script))

    assert scanned == [prompt]


@pytest.mark.parametrize("name", ["job.SH", "job.BASH"])
def test_scheduler_shell_classification_is_case_insensitive_and_walks_child(
    tmp_path,
    name,
):
    child = tmp_path / "child.sh"
    child.write_text("hermes gateway restart\n", encoding="utf-8")
    script = tmp_path / name
    script.write_text("./child.sh\n", encoding="utf-8")

    with pytest.raises(lifecycle_guard.GatewayLifecycleBlocked):
        lifecycle_guard.check_gateway_lifecycle("clean prompt", str(script))
