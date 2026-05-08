"""LanceDB memory provider — schemas + tool dispatch（默认用伪造向量避免拉模型）。

未安装 ``lancedb`` / ``pyarrow`` 时相关用例跳过。

本地完整验收（会触发 sentence-transformers 首次下载）::

    LANCEDB_E2E=1 scripts/run_tests.sh tests/plugins/memory/test_lancedb_memory_plugin.py -s -q --tb=short -n 0
"""

from __future__ import annotations

import json
import os

import pytest

EXPECTED_TOOL_NAMES = frozenset(
    {"lancedb_profile", "lancedb_search", "lancedb_conclude", "lancedb_remove"}
)


@pytest.fixture
def lancedb_provider(tmp_path, monkeypatch):
    pytest.importorskip("lancedb")
    pytest.importorskip("pyarrow")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LANCEDB_RERANK", "false")

    from plugins.memory.lancedb import LanceDBMemoryProvider

    assert LanceDBMemoryProvider().is_available() is True

    p = LanceDBMemoryProvider()
    p.initialize(
        "session-lancedb-pytest",
        hermes_home=str(tmp_path),
        user_id="u_pytest",
        agent_identity="agent_pytest",
        platform="cli",
    )

    # 伪造向量嵌入，便于 CI / 离线跑通 LanceDB（真实模型见 test_lancedb_integration_real_embedding）
    def _fake_embed(self, texts):  # noqa: ARG001
        d = getattr(self, "_vector_dim", 384) or 384
        v = [(1.0 / d ** 0.5)] * d
        return [list(v) for _ in texts]

    monkeypatch.setattr(
        LanceDBMemoryProvider,
        "_embed",
        _fake_embed,
    )
    yield p


def test_four_tool_schemas(lancedb_provider):
    schemas = lancedb_provider.get_tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == EXPECTED_TOOL_NAMES
    assert len(schemas) == 4
    for s in schemas:
        assert s.get("parameters", {}).get("type") == "object"
        assert "description" in s


def test_lancedb_profile_and_conclude_and_search(lancedb_provider):
    p = lancedb_provider
    raw_p0 = p.handle_tool_call("lancedb_profile", {})
    r0 = json.loads(raw_p0)
    assert "result" in r0
    assert "No memories stored yet" in r0.get("result", "")

    raw_c = p.handle_tool_call(
        "lancedb_conclude",
        {"conclusion": "用户偏好深色主题，常用 Python。", "scope": "user"},
    )
    assert json.loads(raw_c).get("result") == "Fact stored."

    raw_s = p.handle_tool_call(
        "lancedb_search",
        {"query": "深色 主题 Python", "top_k": 5, "rerank": False},
    )
    data = json.loads(raw_s)
    assert data["count"] >= 1
    mid = data["results"][0]["id"]
    assert data["results"][0]["content"]

    raw_p1 = p.handle_tool_call("lancedb_profile", {})
    r1 = json.loads(raw_p1)
    assert r1.get("count", 0) >= 1
    assert "深色" in r1.get("result", "")

    raw_rm = p.handle_tool_call("lancedb_remove", {"memory_id": mid})
    assert json.loads(raw_rm).get("result") == "Memory removed."


@pytest.mark.skipif(
    os.environ.get("LANCEDB_E2E") != "1",
    reason="完整验收请加 LANCEDB_E2E=1（首次会下载向量模型）",
)
def test_lancedb_integration_real_embedding(tmp_path, monkeypatch):
    """本地验收：加载真实 sentence-transformers（首次运行会下载模型）。"""
    pytest.importorskip("lancedb")
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LANCEDB_RERANK", "false")

    from plugins.memory.lancedb import LanceDBMemoryProvider

    p = LanceDBMemoryProvider()
    assert p.is_available()
    p.initialize(
        "session-real-embed",
        hermes_home=str(tmp_path),
        user_id="u_real",
        agent_identity="hermes-test",
        platform="cli",
    )

    print("=== SCHEMAS ===")
    schemas = p.get_tool_schemas()
    for sch in schemas:
        print(json.dumps(sch, ensure_ascii=False))
    print()

    print("=== lancedb_profile (empty expected) ===")
    print(p.handle_tool_call("lancedb_profile", {}))
    print()

    print("=== lancedb_conclude ===")
    print(
        p.handle_tool_call(
            "lancedb_conclude",
            {"conclusion": "集成测试写入：深色主题偏好。", "scope": "user"},
        )
    )
    print()

    print("=== lancedb_search ===")
    print(p.handle_tool_call("lancedb_search", {"query": "深色", "rerank": False, "top_k": 8}))
    print()

    print("=== lancedb_profile ===")
    print(p.handle_tool_call("lancedb_profile", {}))
    print()

    data_s = json.loads(
        p.handle_tool_call(
            "lancedb_search", {"query": "深色", "rerank": False, "top_k": 8}
        )
    )
    assert data_s["count"] >= 1
