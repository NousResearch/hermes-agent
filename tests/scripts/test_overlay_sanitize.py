"""Tests for scripts/merge_tools/overlay_sanitize.py"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MERGE_TOOLS = REPO_ROOT / "scripts" / "merge_tools"
if str(MERGE_TOOLS) not in sys.path:
    sys.path.insert(0, str(MERGE_TOOLS))

from overlay_sanitize import load_overlay_sanitizers, sanitize_fork_overlay_text  # noqa: E402


def _has_git_blob(ref: str, path: str) -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{path}"],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def test_strategy_defines_toolsets_sanitizer():
    path = MERGE_TOOLS / "hermes-merge-conflict-strategies.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    sanitizers = load_overlay_sanitizers(payload)
    assert "toolsets.py" in sanitizers
    region = sanitizers["toolsets.py"]["replace_fork_region_with_upstream"]
    assert region["start_anchor"] == '"cronjob",'
    assert "Home Assistant" in region["end_anchor"]


def test_sanitize_drops_stale_send_message_region():
    upstream = '\n'.join(
        [
            '    "cronjob",',
            '    # Home Assistant smart home control (gated on HASS_TOKEN via check_fn)',
            '    "ha_list_entities",',
        ],
    ) + "\n"
    fork = '\n'.join(
        [
            '    "cronjob",',
            '    # Kanban coordination (runtime-gated by tools/kanban_tools.py check_fn)',
            '    "kanban_show",',
            '    # Cross-platform messaging (gated on gateway running via check_fn)',
            '    "send_message",',
            '    # Home Assistant smart home control (gated on HASS_TOKEN via check_fn)',
            '    "ha_list_entities",',
        ],
    ) + "\n"
    sanitizers = {
        "toolsets.py": {
            "replace_fork_region_with_upstream": {
                "start_anchor": '"cronjob",',
                "end_anchor": "# Home Assistant smart home control",
            },
        },
    }
    sanitized = sanitize_fork_overlay_text("toolsets.py", fork, upstream, sanitizers)
    assert '"send_message"' not in sanitized
    assert sanitized.count('"cronjob",') == 1
    assert sanitized.index('"cronjob",') < sanitized.index('"ha_list_entities"')


@pytest.mark.parametrize(
    ("fork_ref", "merge_base", "upstream_ref"),
    [
        (
            "6140cce1870a29dbdf78dca1dcba99428d5e99ae",
            "992b9223893453b3b1527b2ba728996ec81e83f2",
            "33b1d144590a211100f42aa911fd7f91ba031507",
        ),
    ],
)
def test_toolsets_three_way_overlay_clean_after_sanitize(
    fork_ref: str, merge_base: str, upstream_ref: str,
):
    from apply_three_way_overlay import three_way_merge  # noqa: E402

    missing_refs = [
        ref
        for ref in (fork_ref, merge_base, upstream_ref)
        if not _has_git_blob(ref, "toolsets.py")
    ]
    if missing_refs:
        pytest.skip(
            "historical toolsets.py merge fixtures are unavailable in this checkout"
        )

    path = MERGE_TOOLS / "hermes-merge-conflict-strategies.json"
    sanitizers = load_overlay_sanitizers(json.loads(path.read_text(encoding="utf-8")))
    code, merged = three_way_merge(
        "toolsets.py",
        merge_base,
        upstream_ref,
        fork_ref,
        sanitizers=sanitizers,
    )
    assert code != 2
    assert "<<<<<<<" not in merged
