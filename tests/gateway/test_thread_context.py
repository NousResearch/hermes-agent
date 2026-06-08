from pathlib import Path

from gateway.thread_context import (
    ThreadContextConfig,
    build_thread_context_block,
    resolve_thread_context_path,
)


def test_resolve_thread_context_path_uses_stable_ids_not_chat_name(tmp_path: Path):
    cfg = ThreadContextConfig(enabled=True, root=tmp_path)

    path = resolve_thread_context_path(
        platform="telegram",
        chat_id="-100123456",
        thread_id="1520",
        chat_name="lgd & iKun renamed later",
        config=cfg,
    )

    assert path == tmp_path / "telegram" / "-100123456" / "thread-1520.md"


def test_build_thread_context_block_loads_generic_thread_summary(tmp_path: Path):
    summary = tmp_path / "discord" / "987654" / "thread-112233.md"
    summary.parent.mkdir(parents=True)
    summary.write_text("Project summary\n\nDecision: keep it compact.", encoding="utf-8")

    block = build_thread_context_block(
        platform="discord",
        chat_id="987654",
        thread_id="112233",
        config=ThreadContextConfig(enabled=True, root=tmp_path),
    )

    assert block is not None
    assert block.startswith("## Thread Context")
    assert "local thread summary" in block
    assert str(summary) not in block
    assert "Project summary" in block
    assert "Decision: keep it compact." in block


def test_build_thread_context_block_is_noop_when_disabled(tmp_path: Path):
    summary = tmp_path / "telegram" / "123" / "thread-2.md"
    summary.parent.mkdir(parents=True)
    summary.write_text("Should not load", encoding="utf-8")

    assert (
        build_thread_context_block(
            platform="telegram",
            chat_id="123",
            thread_id="2",
            config=ThreadContextConfig(enabled=False, root=tmp_path),
        )
        is None
    )


def test_build_thread_context_block_sanitizes_path_segments(tmp_path: Path):
    cfg = ThreadContextConfig(enabled=True, root=tmp_path)

    path = resolve_thread_context_path(
        platform="telegram/../../evil",
        chat_id="../secret",
        thread_id="../../escape",
        config=cfg,
    )

    assert path is not None
    assert path.is_relative_to(tmp_path)
    assert ".." not in path.relative_to(tmp_path).parts
    assert path.name == "thread-escape.md"


def test_build_thread_context_block_truncates_long_summary(tmp_path: Path):
    summary = tmp_path / "telegram" / "123" / "thread-9.md"
    summary.parent.mkdir(parents=True)
    summary.write_text("A" * 20, encoding="utf-8")

    block = build_thread_context_block(
        platform="telegram",
        chat_id="123",
        thread_id="9",
        config=ThreadContextConfig(enabled=True, root=tmp_path, max_chars=10),
    )

    assert block is not None
    assert "AAAAAAAAAA" in block
    assert "truncated" in block.lower()
