from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agent import prior_work_first as pwf


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "agent@example.com")
    _git(repo, "config", "user.name", "Agent Test")
    (repo / "docs").mkdir()
    (repo / "agent").mkdir()
    (repo / "tests").mkdir()
    (repo / "AGENTS.md").write_text("contract v1\n", encoding="utf-8")
    (repo / "docs" / "design.md").write_text("sdd v1\n", encoding="utf-8")
    (repo / "agent" / "demo.py").write_text(
        "def stable():\n"
        "    return 1\n"
        "\n"
        "def helper():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    (repo / "tests" / "test_demo.py").write_text(
        "def test_demo():\n"
        "    assert True\n",
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        "[project]\n"
        "dependencies = [\"pytest==8.4.0\"]\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    return repo


def _packet(repo: Path, *, status: str = "accepted") -> dict[str, object]:
    git_state = pwf.collect_git_state(repo)
    return pwf.normalize_packet(
        {
            "PROGRAM_ID": pwf.PROGRAM_ID,
            "GOAL": "Finish the demo capability",
            "CAPABILITY_ID": "demo-capability",
            "PREFLIGHT_STATUS": status,
            "ACCEPTED_HEAD": git_state["HEAD"],
            "ACCEPTED_TREE": git_state["TREE"],
            "FIRST_UNPROVEN_BOUNDARY": "IMPLEMENT",
            "KNOWN_FILES": [
                {
                    "path": "agent/demo.py",
                    "digest": pwf.digest_path(repo / "agent" / "demo.py"),
                }
            ],
            "KNOWN_SYMBOLS": [
                {
                    "path": "agent/demo.py",
                    "symbol": "stable",
                    "digest": pwf.digest_symbol(repo / "agent" / "demo.py", "stable"),
                }
            ],
            "KNOWN_TESTS": [
                {
                    "path": "tests/test_demo.py",
                    "digest": pwf.digest_path(repo / "tests" / "test_demo.py"),
                }
            ],
            "KNOWN_DEPENDENCIES": [
                {
                    "name": "pytest",
                    "path": "pyproject.toml",
                    "digest": pwf.digest_path(repo / "pyproject.toml"),
                }
            ],
            "RELEVANT_CONTRACT_SLICES": [
                {
                    "path": "AGENTS.md",
                    "digest": pwf.digest_path(repo / "AGENTS.md"),
                }
            ],
            "RELEVANT_SDD_SLICES": [
                {
                    "path": "docs/design.md",
                    "digest": pwf.digest_path(repo / "docs" / "design.md"),
                }
            ],
            "INVALIDATE_IF": [
                {
                    "kind": "symbol_change",
                    "path": "agent/demo.py",
                    "symbol": "stable",
                    "expected_digest": pwf.digest_symbol(
                        repo / "agent" / "demo.py", "stable"
                    ),
                },
                {
                    "kind": "contract_change",
                    "path": "AGENTS.md",
                    "expected_digest": pwf.digest_path(repo / "AGENTS.md"),
                },
                {
                    "kind": "dependency_change",
                    "path": "pyproject.toml",
                    "expected_digest": pwf.digest_path(repo / "pyproject.toml"),
                },
                {
                    "kind": "test_semantics_change",
                    "path": "tests/test_demo.py",
                    "expected_digest": pwf.digest_path(repo / "tests" / "test_demo.py"),
                },
            ],
            "PRIMARY_WRITER": "writer-1",
            "PARALLEL_SCOUTS": [
                {"name": "scout-1", "mode": "read_only", "read_only": True}
            ],
            "MESSAGES": [{"role": "user", "content": "must be stripped"}],
        }
    )


def test_compact_packet_drops_transcripts():
    packet = pwf.normalize_packet(
        {
            "PROGRAM_ID": pwf.PROGRAM_ID,
            "GOAL": "Goal",
            "CAPABILITY_ID": "cap",
            "MESSAGES": [{"role": "user"}],
            "TRANSCRIPT": "nope",
            "CONVERSATION_HISTORY": ["nope"],
        }
    )

    assert "MESSAGES" not in packet
    assert "TRANSCRIPT" not in packet
    assert "CONVERSATION_HISTORY" not in packet
    assert packet["CAPABILITY_ID"] == "cap"


def test_accepted_preflight_with_no_invalidator_skips_broad_rediscovery(tmp_path):
    repo = _init_repo(tmp_path)
    packet = _packet(repo)
    state = pwf.capture_repo_state(repo, packet)

    decision = pwf.evaluate_preflight(
        packet,
        state,
        attempted_full_rediscovery=True,
    )

    assert decision["PREFLIGHT_STATUS"] == "accepted_reuse"
    assert decision["INVALIDATOR_COUNT"] == 0
    assert decision["RESUME_FROM_FIRST_UNPROVEN_BOUNDARY"] is True
    assert decision["FULL_DISCOVERY_FORBIDDEN"] is True
    assert decision["FULL_PREFLIGHT_REEXECUTION_FORBIDDEN"] is True
    assert decision["REPLAN_FORBIDDEN"] is True
    assert decision["REDUNDANT_PREFLIGHT_ATTEMPT"] is True
    assert decision["PREFLIGHT_REUSE_REQUIRED"] is True


def test_relevant_symbol_change_refreshes_only_affected_slice(tmp_path):
    repo = _init_repo(tmp_path)
    packet = _packet(repo)
    (repo / "agent" / "demo.py").write_text(
        "def stable():\n"
        "    return 2\n"
        "\n"
        "def helper():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    state = pwf.capture_repo_state(repo, packet)

    decision = pwf.evaluate_preflight(packet, state)

    assert decision["PREFLIGHT_STATUS"] == "targeted_refresh_required"
    assert decision["INVALIDATOR_COUNT"] == 1
    assert decision["FULL_DISCOVERY_FORBIDDEN"] is True
    assert decision["REFRESH_SCOPE"] == [
        {
            "kind": "symbol_change",
            "scope": "targeted",
            "path": "agent/demo.py",
            "symbol": "stable",
        }
    ]


def test_discovery_finds_handoff_worktree_draft_and_receipt_before_new_plan(tmp_path):
    repo = _init_repo(tmp_path)
    packet = _packet(repo)
    cap_dir = pwf.canonical_capability_dir(repo, "demo-capability")
    (cap_dir / "handoffs").mkdir(parents=True)
    (cap_dir / "receipts").mkdir()
    (cap_dir / "drafts").mkdir()
    (cap_dir / "rag").mkdir()
    (cap_dir / "handoffs" / "accepted.md").write_text("handoff\n", encoding="utf-8")
    (cap_dir / "receipts" / "accepted.json").write_text("{}", encoding="utf-8")
    (cap_dir / "drafts" / "draft.txt").write_text("draft\n", encoding="utf-8")
    (cap_dir / "rag" / "impact.json").write_text("{}", encoding="utf-8")
    (cap_dir / pwf.PACKET_FILENAME).write_text(json.dumps(packet, indent=2), encoding="utf-8")
    worktree_path = repo / ".worktrees" / "resume-demo"
    worktree_path.parent.mkdir()
    _git(repo, "worktree", "add", str(worktree_path), "-b", "feat/resume-demo", "HEAD")

    discovery = pwf.discover_prior_work(repo, "demo-capability")

    assert discovery["SEARCH_MODE"] == "targeted_read_only"
    assert discovery["FULL_HISTORY_LOADED"] is False
    assert discovery["ARTIFACTS"]["preflight_packets"] == [
        ".task-state/prior-work-first/demo-capability/active.packet.json"
    ]
    assert discovery["ARTIFACTS"]["handoffs"] == [
        ".task-state/prior-work-first/demo-capability/handoffs/accepted.md"
    ]
    assert discovery["ARTIFACTS"]["receipts"] == [
        ".task-state/prior-work-first/demo-capability/receipts/accepted.json"
    ]
    assert discovery["ARTIFACTS"]["draft_builds"] == [
        ".task-state/prior-work-first/demo-capability/drafts/draft.txt"
    ]
    assert discovery["ARTIFACTS"]["rag"] == [
        ".task-state/prior-work-first/demo-capability/rag/impact.json"
    ]
    assert any(item["branch"] == "feat/resume-demo" for item in discovery["KNOWN_WORKTREES"])
    assert any(item["branch"] == "feat/resume-demo" for item in discovery["KNOWN_BRANCHES"])


def test_rag_conflict_loses_to_source_and_tests():
    selected_impl = pwf.resolve_authority(
        [
            {"kind": "rag", "value": "stale"},
            {"kind": "source", "value": "live source"},
            {"kind": "tests", "value": "live tests"},
        ],
        intent="implementation",
    )
    selected_verify = pwf.resolve_authority(
        [
            {"kind": "rag", "value": "stale"},
            {"kind": "tests", "value": "live tests"},
        ],
        intent="verification",
    )

    assert selected_impl == {"kind": "source", "value": "live source"}
    assert selected_verify == {"kind": "tests", "value": "live tests"}


def test_contract_change_invalidates_only_contract_slice(tmp_path):
    repo = _init_repo(tmp_path)
    packet = _packet(repo)
    (repo / "AGENTS.md").write_text("contract v2\n", encoding="utf-8")
    state = pwf.capture_repo_state(repo, packet)

    decision = pwf.evaluate_preflight(packet, state)

    assert decision["PREFLIGHT_STATUS"] == "targeted_refresh_required"
    assert decision["INVALIDATOR_COUNT"] == 1
    assert decision["REFRESH_SCOPE"] == [
        {
            "kind": "contract_change",
            "scope": "targeted",
            "path": "AGENTS.md",
        }
    ]


def test_unrelated_head_change_preserves_validity(tmp_path):
    repo = _init_repo(tmp_path)
    packet = _packet(repo)
    (repo / "README.md").write_text("unrelated\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "unrelated change")
    state = pwf.capture_repo_state(repo, packet)

    decision = pwf.evaluate_preflight(packet, state)

    assert state["HEAD"] != packet["ACCEPTED_HEAD"]
    assert state["TREE"] != packet["ACCEPTED_TREE"]
    assert decision["INVALIDATOR_COUNT"] == 0
    assert decision["PREFLIGHT_STATUS"] == "accepted_reuse"


def test_parallel_scouts_stay_read_only_and_one_writer_is_enforced():
    packet = pwf.normalize_packet(
        {
            "PROGRAM_ID": pwf.PROGRAM_ID,
            "GOAL": "Goal",
            "CAPABILITY_ID": "cap",
            "PRIMARY_WRITER": ["writer-a", "writer-b"],
            "PARALLEL_SCOUTS": [
                {"name": "scout-a", "mode": "write", "read_only": False}
            ],
        }
    )

    result = pwf.validate_writer_invariants(packet)

    assert result["ok"] is False
    assert result["primary_writer_count"] == 2
    assert "multiple canonical writers declared" in result["violations"]
    assert any("parallel scout not read-only" in msg for msg in result["violations"])


def test_prior_capability_pass_is_not_reopened_without_invalidator(tmp_path):
    repo = _init_repo(tmp_path)
    packet = _packet(repo, status="verified")
    state = pwf.capture_repo_state(repo, packet)

    decision = pwf.evaluate_preflight(packet, state)

    assert decision["PREFLIGHT_REUSE_REQUIRED"] is True
    assert decision["REOPEN_REQUIRED"] is False
    assert decision["INVALIDATOR_COUNT"] == 0


def test_new_agent_without_chat_history_reconstructs_resume_from_repo_artifacts(tmp_path):
    repo = _init_repo(tmp_path)
    packet = _packet(repo)
    cap_dir = pwf.canonical_capability_dir(repo, "demo-capability")
    (cap_dir / "handoffs").mkdir(parents=True)
    (cap_dir / "drafts").mkdir()
    (cap_dir / "receipts").mkdir()
    (cap_dir / "handoffs" / "accepted.md").write_text("handoff\n", encoding="utf-8")
    (cap_dir / "drafts" / "draft.txt").write_text("draft\n", encoding="utf-8")
    (cap_dir / "receipts" / "accepted.json").write_text("{}", encoding="utf-8")
    packet_path = cap_dir / pwf.PACKET_FILENAME
    packet_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prior_work_first.py",
            "resume",
            "--repo-root",
            str(repo),
            "--packet",
            str(packet_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = json.loads(result.stdout)

    assert payload["GOAL"] == "Finish the demo capability"
    assert payload["CAPABILITY_ID"] == "demo-capability"
    assert payload["FIRST_UNPROVEN_BOUNDARY"] == "IMPLEMENT"
    assert payload["PRIOR_WORK_LOCATE"]["ARTIFACTS"]["handoffs"]
    assert payload["PRIOR_WORK_LOCATE"]["ARTIFACTS"]["draft_builds"]
    assert payload["PRIOR_WORK_LOCATE"]["ARTIFACTS"]["receipts"]
    assert payload["INVALIDATOR_CHECK"]["PREFLIGHT_STATUS"] == "accepted_reuse"
    assert payload["NEXT_ACTION"] == "RESUME_FROM_FIRST_UNPROVEN_BOUNDARY"
