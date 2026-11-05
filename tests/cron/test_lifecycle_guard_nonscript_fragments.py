"""Content-derived non-script fragments must not exhaust the lifecycle guard's remote-read budget (#98801).

A path-shaped fragment at command position inside referenced-script text — ``</s:Header>``
tokenized out of an XML/HTML string literal in a ~1000-line Python CLI — does not resolve
locally, so every one of the >60 fragments charged the 64-remote-read budget, the walk
exhausted, and the guard failed closed: every absolute-path invocation of the script was
blocked while the bare name was never scanned. The fix gates content-derived bare-path
candidates on script plausibility (script suffix, shebang, or an explicit exec form:
``sh X`` / ``source X`` / ``./X``). The agent-typed depth-0 command, every exec/source/.sh
reference, the budget constants, and the fail-closed semantics are untouched.
"""

from __future__ import annotations

import cron.lifecycle_guard as lifecycle_guard

guard = lifecycle_guard.contains_gateway_lifecycle_command_or_referenced_script

# One more than the default _MAX_LIFECYCLE_SCAN_REMOTE_READS (64): on unfixed code each
# fragment charged a remote read, so anything above 64 forced the fail-closed verdict.
FRAGMENT_COUNT = 80


def _fragment_script_text() -> str:
    """A stdlib-Python CLI built from Exchange-SOAP-style XML literals, the #98801 shape.

    Each parenthesized quoted fragment tokenizes to a path-shaped token at command position
    (``(`` and ``)`` are control characters for the guard's shlex pass), exactly like the
    real mail CLI's ``</s:Header>`` fragments."""
    lines = [
        "#!/usr/bin/env python3",
        '"""Offline mail CLI: builds Exchange SOAP envelopes from reply-quote HTML."""',
        "",
    ]
    for i in range(FRAGMENT_COUNT):
        lines += [
            f"ENVELOPE_{i} = (",
            "    '</s:Header>'",
            "    '</s:Body></s:Envelope>'",
            ")",
        ]
    lines += [
        "",
        "def main() -> None:",
        "    print('checked')",
        "",
    ]
    return "\n".join(lines) + "\n"


# --- positive: the bug class no longer fails closed -------------------------------------------


def test_absolute_path_invocation_of_local_fragment_script_is_allowed(tmp_path):
    """The docs-recommended absolute form must not be punished: an extensionless local CLI whose
    text is full of XML fragments runs, and no remote read is spent on any fragment."""
    cli = tmp_path / "mail"
    cli.write_text(_fragment_script_text(), encoding="utf-8")
    remote_reads: list[str] = []

    def remote(path: str):
        remote_reads.append(path)
        return None

    assert guard(f"{cli} check", read_remote_script=remote, cwd=str(tmp_path)) is False
    assert remote_reads == []


def test_minimal_repro_remote_backend_is_not_blocked():
    """The #98801 minimal reproducer: the CLI exists only on the remote backend, so its own text
    arrives through ``read_remote_script``; the fragments inside it must not burn the budget.

    f(\"/opt/data/bin/mail check\", read_remote_script=rrs) blocked on unfixed code (True);
    the bare name never was (False) — the guard punished exactly the invoked form."""
    reads: list[str] = []

    def remote(path: str):
        reads.append(path)
        return _fragment_script_text()

    assert guard("/opt/data/bin/mail check", read_remote_script=remote) is False
    # Exactly one roundtrip — the CLI itself. The fragments charge nothing.
    assert reads == ["/opt/data/bin/mail"]


def test_markdown_table_inside_referenced_script_is_not_blocked(tmp_path):
    """A markdown table tokenized inside an executed script's text yields path-shaped
    command-position fragments too (``|`` splits segments); none is a script, none charges a
    remote read, and the walk no longer fails closed on them."""
    notes = tmp_path / "notes.sh"
    rows = "\n".join(f"| /opt/frag{i}/tool-{i}.md | step {i} |" for i in range(FRAGMENT_COUNT))
    notes.write_text(f"#!/bin/sh\n# runbook notes\n{rows}\n", encoding="utf-8")
    remote_reads: list[str] = []

    def remote(path: str):
        remote_reads.append(path)
        return None

    assert guard(f"bash {notes}", read_remote_script=remote, cwd=str(tmp_path)) is False
    assert remote_reads == []


def test_inert_heredoc_doc_chain_with_real_invocation_is_not_blocked(tmp_path):
    """The original #98801 documentation chain: a provably-inert heredoc body full of doc paths
    followed by a real absolute-path invocation stays allowed."""
    cli = tmp_path / "mail"
    cli.write_text(_fragment_script_text(), encoding="utf-8")
    rows = "\n".join(f"See /opt/docs/runbook-{i}.md step {i}" for i in range(FRAGMENT_COUNT))
    command = f"python3 - <<'PY'\nprint(open('notes').read())\n{rows}\nPY\n{cli} check"

    assert guard(command, cwd=str(tmp_path)) is False


# --- white-box: the budget is never charged for non-script candidates -------------------------


def test_nonscript_candidates_never_charge_walk_budget(monkeypatch, tmp_path):
    """White-box, in the style of test_lifecycle_guard_budget: with the remote-read budget set to
    0, a single charge anywhere in the walk fails every executed candidate closed. The fragment
    walk must finish with zero remote-read charges and exactly one path charge (the CLI itself);
    the fragments are gated out before either counter is touched."""
    monkeypatch.setattr(lifecycle_guard, "_MAX_LIFECYCLE_SCAN_REMOTE_READS", 0)
    path_charges: list[int] = []
    remote_charges: list[int] = []
    original_path = lifecycle_guard._LifecycleScanBudget.charge_path
    original_remote = lifecycle_guard._LifecycleScanBudget.charge_remote_read

    def spy_path(self):
        if original_path(self):
            path_charges.append(1)
        return True

    def spy_remote(self):
        remote_charges.append(1)
        return original_remote(self)

    monkeypatch.setattr(lifecycle_guard._LifecycleScanBudget, "charge_path", spy_path)
    monkeypatch.setattr(lifecycle_guard._LifecycleScanBudget, "charge_remote_read", spy_remote)

    cli = tmp_path / "mail"
    cli.write_text(_fragment_script_text(), encoding="utf-8")

    assert guard(f"{cli} check", read_remote_script=lambda p: None, cwd=str(tmp_path)) is False
    assert sum(path_charges) == 1
    assert remote_charges == []


# --- negative: the gate must not weaken the guard ---------------------------------------------


def test_executable_sh_with_gateway_stop_still_blocked(tmp_path):
    """A real .sh carrying a lifecycle command stays blocked through every reference form:
    explicit exec form, direct absolute-path invocation, and a bare .sh reference from INSIDE
    another script (depth 1, the exact shape the gate sits on)."""
    evil = tmp_path / "stop-gateway.sh"
    evil.write_text("#!/bin/sh\nhermes gateway stop\n", encoding="utf-8")

    assert guard(f"bash {evil}", cwd=str(tmp_path)) is True
    assert guard(str(evil), cwd=str(tmp_path)) is True

    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(f"#!/bin/sh\n{evil}\n", encoding="utf-8")
    assert guard(f"bash {wrapper}", cwd=str(tmp_path)) is True


def test_raw_lifecycle_string_in_typed_command_still_blocked():
    """The raw-text regex over the full command at every level is untouched: a literal lifecycle
    command anywhere in typed text blocks with no filesystem at all."""
    assert guard("echo pre && hermes gateway stop && echo post") is True
    assert guard("cat /opt/data/notes.md; hermes gateway stop") is True


def test_literal_lifecycle_text_inside_fragment_script_still_blocked(tmp_path):
    """Reading is not the only layer: when a scanned script's text carries a literal lifecycle
    command, it is blocked even when the script is otherwise full of inert XML fragments."""
    cli = tmp_path / "mail"
    cli.write_text(_fragment_script_text() + "RUN = 'hermes gateway restart'\n", encoding="utf-8")

    assert guard(f"{cli} check", cwd=str(tmp_path)) is True


def test_exec_and_source_forms_still_followed_at_depth(tmp_path):
    """The explicit exec forms must bypass the gate: an extensionless, shebangless payload is
    unreachable by the plausibility sniff, yet ``. X`` / ``sh X`` / ``./X`` references to it are
    still followed exactly as before and still block."""
    payload = tmp_path / "reload-helper"
    payload.write_text("hermes gateway restart\n", encoding="utf-8")

    dot_wrapper = tmp_path / "dot-wrapper.sh"
    dot_wrapper.write_text(f"#!/bin/sh\n. {payload}\n", encoding="utf-8")
    assert guard(f"bash {dot_wrapper}", cwd=str(tmp_path)) is True

    sh_wrapper = tmp_path / "sh-wrapper.sh"
    sh_wrapper.write_text(f"#!/bin/sh\nsh {payload}\n", encoding="utf-8")
    assert guard(f"bash {sh_wrapper}", cwd=str(tmp_path)) is True

    slash_wrapper = tmp_path / "slash-wrapper.sh"
    slash_wrapper.write_text("#!/bin/sh\n./reload-helper\n", encoding="utf-8")
    assert guard(f"bash {slash_wrapper}", cwd=str(tmp_path)) is True


def test_extensionless_local_script_with_shebang_still_followed(tmp_path):
    """The shebang branch of the plausibility gate: an extensionless executable with a shebang is
    still a script — still read, still scanned, still blocking."""
    tool = tmp_path / "gwtool"
    tool.write_text("#!/bin/sh\nhermes gateway restart\n", encoding="utf-8")
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(f"#!/bin/sh\n{tool}\n", encoding="utf-8")

    assert guard(f"bash {wrapper}", cwd=str(tmp_path)) is True


def test_remote_sh_with_gateway_stop_still_blocked():
    """Not charging remote reads for non-script candidates must not de-scan remote shell
    scripts: an exec-form .sh reached through a remote backend is still read and still blocks."""
    reads: list[str] = []

    def remote(path: str):
        reads.append(path)
        return "#!/bin/sh\nhermes gateway stop\n"

    assert guard("bash /remote/entry.sh", read_remote_script=remote) is True
    assert reads == ["/remote/entry.sh"]
