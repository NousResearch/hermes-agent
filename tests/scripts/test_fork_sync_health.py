"""Fork CI's sync-aware health gate (.github/scripts/fork_sync_health.py): which findings block
a change that brings in an upstream release, and which tags count as that release."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "fork_sync_health.py"
_spec = importlib.util.spec_from_file_location("fork_sync_health", _PATH)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _finding(path, rule="BLE001", scope="f", *, line=1, blocking=True, allowed_reason=None):
    return {"path": path, "rule": rule, "scope": scope, "line": line, "detail": "d",
            "blocking": blocking, "allowed_reason": allowed_reason}


def test_only_findings_new_against_both_sides_block():
    upstream_import = _finding("plugins/platforms/telegram/standalone.py")  # new vs fork main only
    fork_debt = _finding("plugins/platforms/discord/adapter.py", "CC", "_handle_message")  # vs release only
    regression = _finding("gateway/session.py", scope="resolve", line=10)  # both

    ours = gate.blocking_in_both(
        [upstream_import, regression],
        [fork_debt, _finding("gateway/session.py", scope="resolve", line=12)],
    )

    assert ours == [regression]


def test_allowed_or_advisory_findings_never_block():
    allowed = _finding("a.py", allowed_reason="reviewed")
    advisory = _finding("b.py", blocking=False)

    assert gate.blocking_in_both([allowed, advisory], [_finding("a.py"), _finding("b.py")]) == []


def test_only_plain_version_tags_are_releases_at_their_peeled_commit():
    listing = "\n".join([
        "aaa\trefs/tags/v0.23.1",
        "bbb\trefs/tags/v0.23.10",
        "ccc\trefs/tags/v0.23.10^{}",
        "ddd\trefs/tags/v0.23.3+canary.20261120T064609Z",
        "eee\trefs/tags/rc.4-v0.23.3",
        "fff\trefs/tags/abandoned-rc.1-v0.23.2",
        "ggg\trefs/tags/v0.9.1.1",
        "hhh\trefs/tags/v2026.9.24",
    ])

    assert gate.stable_releases(listing) == {"v0.23.1": "aaa", "v0.23.10": "ccc", "v2026.9.24": "hhh"}
