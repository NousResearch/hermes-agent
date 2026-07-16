"""Fail-closed runtime configuration tests."""

from __future__ import annotations

import asyncio

import pytest

from plugins.research_protocol.planner_tools import (
    check_planner_approvals,
    check_planner_artifacts,
    check_planner_context,
    configure_planner_runtime,
)
from plugins.research_protocol.runtime import (
    RuntimeConfigurationError,
    build_runtime_bundle,
)


def test_runtime_rejects_missing_or_relative_artifact_root(tmp_path):
    with pytest.raises(RuntimeConfigurationError, match="absolute artifact_root"):
        build_runtime_bundle({}, environ={})
    with pytest.raises(RuntimeConfigurationError, match="absolute artifact_root"):
        build_runtime_bundle({"artifact_root": "relative/artifacts"}, environ={})


def test_artifact_only_runtime_does_not_enable_database_tools(tmp_path):
    bundle = build_runtime_bundle(
        {"artifact_root": str(tmp_path / "artifacts")},
        environ={},
    )
    configure_planner_runtime(
        bundle.handlers,
        artifact_available=bundle.artifact_available,
        context_available=bundle.context_available,
        approval_available=bundle.approval_available,
    )
    try:
        assert check_planner_artifacts() is True
        assert check_planner_context() is False
        assert check_planner_approvals() is False
        result = asyncio.run(
            bundle.handlers.context_read({
                "query_id": "approval_status.v1",
                "parameters": {"approval_id": "approval-001"},
            })
        )
        assert result == {"ok": False, "error": "planner database runtime unavailable"}
    finally:
        configure_planner_runtime(None)


def test_database_runtime_uses_distinct_lazy_reader_and_writer_pools(tmp_path):
    bundle = build_runtime_bundle(
        {
            "artifact_root": str(tmp_path / "artifacts"),
            "database": {
                "max_rows": 7,
                "max_bytes": 4096,
                "timeout_seconds": 2.0,
            },
        },
        environ={
            "RESEARCH_PROTOCOL_READER_DATABASE_URL": (
                "postgresql://reader:not-contacted@invalid/read"
            ),
            "RESEARCH_PROTOCOL_WRITER_DATABASE_URL": (
                "postgresql://writer:not-contacted@invalid/write"
            ),
        },
    )

    assert bundle.artifact_available is True
    assert bundle.context_available is True
    assert bundle.approval_available is True
    assert bundle.reader_pool is not None
    assert bundle.writer_pool is not None
    assert bundle.reader_pool is not bundle.writer_pool
    assert bundle.reader_pool.created is False
    assert bundle.writer_pool.created is False


def test_database_runtime_keeps_reader_and_writer_authority_independent(tmp_path):
    reader_only = build_runtime_bundle(
        {"artifact_root": str(tmp_path / "reader-artifacts")},
        environ={
            "RESEARCH_PROTOCOL_READER_DATABASE_URL": (
                "postgresql://reader:not-contacted@invalid/read"
            )
        },
    )
    assert reader_only.context_available is True
    assert reader_only.approval_available is False

    writer_only = build_runtime_bundle(
        {"artifact_root": str(tmp_path / "writer-artifacts")},
        environ={
            "RESEARCH_PROTOCOL_WRITER_DATABASE_URL": (
                "postgresql://writer:not-contacted@invalid/write"
            )
        },
    )
    assert writer_only.context_available is False
    assert writer_only.approval_available is True


def test_database_runtime_rejects_shared_reader_writer_dsn(tmp_path):
    dsn = "postgresql://combined:not-contacted@invalid/database"
    with pytest.raises(RuntimeConfigurationError, match="must be distinct"):
        build_runtime_bundle(
            {"artifact_root": str(tmp_path / "artifacts")},
            environ={
                "RESEARCH_PROTOCOL_READER_DATABASE_URL": dsn,
                "RESEARCH_PROTOCOL_WRITER_DATABASE_URL": dsn,
            },
        )


def test_runtime_rejects_out_of_range_database_caps(tmp_path):
    with pytest.raises(RuntimeConfigurationError, match="max_rows"):
        build_runtime_bundle(
            {
                "artifact_root": str(tmp_path / "artifacts"),
                "database": {"max_rows": 0},
            },
            environ={
                "RESEARCH_PROTOCOL_READER_DATABASE_URL": "postgresql://example/read"
            },
        )
