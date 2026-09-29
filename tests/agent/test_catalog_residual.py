"""Pure catalog residual extraction — no LLM, no invention, last-wins, budget."""

from unittest.mock import patch

from agent.catalog_residual import (
    CATALOG_BUDGET_CHARS,
    CATALOG_HEADING,
    CATALOG_RECEIPT_HEADING,
    DEFAULT_COMPRESSION_MODE,
    HYBRID_INDEX_HEADING,
    LEAN_ANCHOR_HEADING,
    append_hybrid_handle_index,
    build_catalog_residual,
    build_hybrid_handle_index,
    extract_catalog_items,
    merge_handles_into_anchor_index,
    normalize_compression_mode,
)

SECRET = "sk-proj-" + ("a" * 40)
OAUTH_URL = (
    "https://localhost/callback?code=opaque-code-123"
    "&access_token=opaque-token-456&state=keep"
)


def _msgs():
    return [
        {
            "role": "user",
            "content": "Please edit /tmp/project/app.py and check issue #4242",
        },
        {
            "role": "assistant",
            "content": "Inspecting the file.",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"/tmp/project/app.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "def main():\n    return 1\n" + ("x" * 80),
        },
        {
            "role": "user",
            "content": "Also fetch https://example.com/docs and use commit abcdef123",
        },
        {
            "role": "assistant",
            "content": "Fetching docs.",
            "tool_calls": [
                {
                    "id": "c2",
                    "function": {
                        "name": "web_extract",
                        "arguments": '{"url":"https://example.com/docs"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "c2",
            "name": "web_extract",
            "content": "Docs ok",
        },
        {
            "role": "user",
            "content": "Please edit /tmp/project/app.py again — last write wins",
        },
    ]


def test_normalize_mode_default_and_invalid():
    assert normalize_compression_mode(None) == DEFAULT_COMPRESSION_MODE
    assert normalize_compression_mode("") == "standard"
    assert normalize_compression_mode("CATALOG") == "catalog"
    assert normalize_compression_mode("hybrid") == "hybrid"
    assert normalize_compression_mode("banana") == "standard"
    assert normalize_compression_mode(1) == "standard"


def test_extract_is_extractive_no_invention():
    items = extract_catalog_items(_msgs())
    catalog = "\n".join(
        list(items["files"].values())
        + list(items["tools"].values())
        + list(items["identifiers"].values())
        + list(items["topics"].values())
        + list(items["story"].values())
    )
    invented = "quantum entanglement drive"
    assert invented not in catalog
    assert "/tmp/project/app.py" in items["files"].values()
    assert "read_file" in items["tools"].values()
    assert "web_extract" in items["tools"].values()
    assert any("example.com/docs" in value for value in items["identifiers"].values())
    assert any("#4242" in value for value in items["identifiers"].values())
    assert any("abcdef123" in value for value in items["identifiers"].values())
    assert any("last write wins" in value for value in items["story"].values())


def test_last_wins_file_and_story():
    messages = [
        {"role": "user", "content": "Use /tmp/old.py for the first pass"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"/tmp/conflict.py"}',
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c2",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"/tmp/conflict.py"}',
                    },
                }
            ],
        },
        {"role": "user", "content": "Switch to /tmp/new.py instead"},
    ]
    # Same normalized path written twice — last args win (same path, same value).
    # Different story asks: newest unique ask is kept.
    items = extract_catalog_items(messages)
    assert "/tmp/new.py" in items["files"].values()
    assert "/tmp/conflict.py" in items["files"].values()
    story = " ".join(items["story"].values())
    assert "Switch to /tmp/new.py" in story
    # Oldest ask may drop once the story window fills; last-wins keeps newest.
    assert list(items["story"].values())[-1].startswith("Switch to")


def test_last_wins_moves_updated_stem_story_and_path():
    """Updating a duplicate handle must move it to the newest recency slot."""
    messages = [
        {
            "role": "user",
            "content": "Ship the release today. Use the staging checklist.",
        },
        {"role": "user", "content": "Beta request about the docs site."},
        {"role": "user", "content": "Gamma request about the test suite."},
        {"role": "user", "content": "Delta request about the lint rules."},
        {
            "role": "user",
            "content": "Ship the release today. Use the production checklist.",
        },
    ]
    items = extract_catalog_items(messages)
    stories = list(items["story"].values())
    story_text = " ".join(stories)
    assert "production checklist" in story_text
    assert "staging checklist" not in story_text
    assert not any("Beta request" in value for value in stories)
    assert stories[-1].startswith("Ship the release today")

    path_messages = [
        {"role": "user", "content": f"Inspect /tmp/old_{idx:03d}.py"}
        for idx in range(24)
    ]
    path_messages.append({
        "role": "user",
        "content": "Inspect /tmp/old_000.py with the newest note",
    })
    path_messages.append({
        "role": "user",
        "content": "Finally inspect /tmp/newest_win.py",
    })
    files = list(extract_catalog_items(path_messages)["files"].values())
    assert "/tmp/newest_win.py" in files
    assert files.index("/tmp/old_000.py") > files.index("/tmp/old_001.py")
    assert files.index("/tmp/newest_win.py") > files.index("/tmp/old_001.py")

    body = build_catalog_residual(path_messages, budget=360)
    files_section = body.split("## Files", 1)[-1].split("## ", 1)[0]
    assert "/tmp/newest_win.py" in files_section
    assert "/tmp/old_001.py" not in files_section


def test_hard_char_cap_and_receipt():
    messages = [
        {
            "role": "user",
            "content": f"Please inspect /tmp/file_{idx:03d}.py for ticket TICK-{idx:04d}",
        }
        for idx in range(80)
    ]
    body = build_catalog_residual(messages, budget=800)
    assert len(body) <= 800
    assert CATALOG_HEADING in body
    assert CATALOG_RECEIPT_HEADING in body
    receipt_idx = body.rfind(CATALOG_RECEIPT_HEADING)
    assert receipt_idx != -1
    assert "kept:" in body[receipt_idx:]
    assert "dropped:" in body[receipt_idx:]
    assert "over budget" in body[receipt_idx:]
    assert body[receipt_idx:].count(CATALOG_RECEIPT_HEADING) == 1


def test_secret_redaction_in_catalog_and_index():
    messages = [
        {
            "role": "user",
            "content": f"Store token {SECRET} and visit {OAUTH_URL} in /tmp/secret.py",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "read_file",
                        "arguments": f'{{"path":"/tmp/secret.py","token":"{SECRET}"}}',
                    },
                }
            ],
        },
    ]
    with patch("agent.redact._REDACT_ENABLED", False):
        body = build_catalog_residual(messages)
        index = build_hybrid_handle_index(messages)
    assert SECRET not in body
    assert SECRET not in index
    assert "sk-proj-" not in body
    assert "code=opaque-code-123" not in body
    assert "access_token=opaque-token-456" not in body
    assert "/tmp/secret.py" in body


def test_prior_residual_reingest_is_stable_and_last_wins():
    prior = build_catalog_residual([
        {
            "role": "user",
            "content": "First pass on /tmp/old.py and https://old.example/a",
        }
    ])
    newer = [
        {
            "role": "user",
            "content": "Now use /tmp/old.py for the rewrite and https://new.example/b",
        },
    ]
    first = build_catalog_residual(newer, previous_residual=prior)
    second = build_catalog_residual(newer, previous_residual=first)
    # Re-running with the prior residual as input must not invent new handles
    # and must keep last-wins (new URL present).
    assert "https://new.example/b" in second
    assert "quantum" not in second.lower()
    # Stability: a second reingest of the same residual does not grow sections.
    assert second.count("/tmp/old.py") == first.count("/tmp/old.py")


def test_hybrid_index_is_compact_unique_handles():
    index = build_hybrid_handle_index(_msgs())
    assert index.startswith(HYBRID_INDEX_HEADING)
    assert "files:" in index
    assert "tools:" in index
    assert "ids:" in index
    assert CATALOG_HEADING not in index
    assert "/tmp/project/app.py" in index
    assert "read_file" in index


def test_append_hybrid_index_is_idempotent():
    summary = "standard summary body"
    index = build_hybrid_handle_index(_msgs())
    once = append_hybrid_handle_index(summary, index)
    twice = append_hybrid_handle_index(once, index)
    assert once.count(HYBRID_INDEX_HEADING) == 1
    assert twice == once


def test_append_hybrid_index_replaces_echoed_heading():
    echoed = "standard summary body\n\n## Unique handles\nfiles: /tmp/stale.py"
    index = "## Unique handles\nfiles: /tmp/fresh.py"
    replaced = append_hybrid_handle_index(echoed, index)
    assert replaced.count(HYBRID_INDEX_HEADING) == 1
    assert "/tmp/fresh.py" in replaced
    assert "/tmp/stale.py" not in replaced


def test_catalog_budget_default_is_hard_capped():
    messages = [
        {"role": "user", "content": f"/very/long/path/segment_{i}/file.py " * 20}
        for i in range(40)
    ]
    body = build_catalog_residual(messages)
    assert len(body) <= CATALOG_BUDGET_CHARS
    assert "kept:" in body
    assert "dropped:" in body


def test_catalog_excludes_synthetic_user_turns_from_story_and_topics():
    from agent.context_compressor import COMPRESSION_CONTINUATION_USER_CONTENT
    from tools.todo_tool import TODO_INJECTION_HEADER

    real_ask = "Please ship the billing invoice exporter"
    messages = [
        {
            "role": "user",
            "content": f"{TODO_INJECTION_HEADER}\n- [ ] preserve this todo",
        },
        {
            "role": "user",
            "content": "[System: Your previous response was truncated] keep going",
        },
        {
            "role": "user",
            "content": "[IMPORTANT: Background process 99] still running",
        },
        {"role": "user", "content": "[Planning state preserved] old plan"},
        {"role": "user", "content": "[ASYNC DELEGATION] worker finished"},
        {"role": "user", "content": "Cronjob Response: nightly batch done"},
        {
            "role": "user",
            "content": "Looks like a real ask about invoices",
            "_todo_snapshot_synthetic": True,
        },
        {
            "role": "user",
            "content": "Handoff that must not become a topic",
            "_compressed_summary": True,
        },
        {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT},
        {"role": "user", "content": real_ask},
    ]
    items = extract_catalog_items(messages)
    story = " ".join(items["story"].values())
    topics = " ".join(items["topics"].values())
    assert "billing invoice exporter" in story
    assert "billing invoice exporter" in topics
    for banned in (
        "preserve this todo",
        "truncated",
        "Background process",
        "old plan",
        "worker finished",
        "nightly batch",
        "Looks like a real ask",
        "Handoff that must not become a topic",
        "Continue from the compressed",
    ):
        assert banned not in story
        assert banned not in topics


def test_reingest_parses_lean_anchor_index():
    prior = (
        f"{LEAN_ANCHOR_HEADING}\n"
        "files: /tmp/first-only.py(x2), /tmp/keep.py\n"
        "urls: https://old.example/a"
    )
    items = extract_catalog_items(
        [{"role": "user", "content": "Now use /tmp/second-pass.py"}],
        previous_residual=prior,
    )
    assert "/tmp/first-only.py" in items["files"].values()
    assert "/tmp/second-pass.py" in items["files"].values()
    assert any("old.example/a" in value for value in items["identifiers"].values())


def test_sha_identifiers_require_a_digit():
    messages = [
        {
            "role": "user",
            "content": (
                "Words acceded and defaced are not SHAs; "
                "use abcdef1 and 973d93f"
            ),
        }
    ]
    items = extract_catalog_items(messages)
    identifiers = " ".join(items["identifiers"].values())
    assert "acceded" not in identifiers
    assert "defaced" not in identifiers
    assert "abcdef1" in identifiers
    assert "973d93f" in identifiers


def test_merge_handles_into_anchor_index_keeps_first_window_files():
    summary = (
        "## Goal\nContinue.\n\n"
        f"{LEAN_ANCHOR_HEADING}\n"
        "files: /tmp/second-pass.py\n"
        "(Exact identifiers from the compacted region.)"
    )
    merged = merge_handles_into_anchor_index(
        summary,
        {
            "files": {
                "/tmp/first-only.py": "/tmp/first-only.py",
                "/tmp/second-pass.py": "/tmp/second-pass.py",
            }
        },
    )
    assert merged.count(LEAN_ANCHOR_HEADING) == 1
    assert "/tmp/first-only.py" in merged
    assert "/tmp/second-pass.py" in merged
    assert merged.count("files:") == 1
