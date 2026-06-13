"""Tests for plugins/memory/openviking/__init__.py — URI normalization and payload handling."""

import json

from plugins.memory.openviking import OpenVikingMemoryProvider


class FakeVikingClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, path, params=None, **kwargs):
        self.calls.append((path, params or {}))
        response = self.responses[(path, tuple(sorted((params or {}).items())))]
        if isinstance(response, Exception):
            raise response
        return response


class TestOpenVikingSummaryUriNormalization:
    def test_normalize_summary_uri_maps_pseudo_files_to_parent_directory(self):
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/.overview.md") == "viking://user/hermes"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://resources/.abstract.md") == "viking://resources"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://") == "viking://"
        assert OpenVikingMemoryProvider._normalize_summary_uri("viking://user/hermes/memories/profile.md") == "viking://user/hermes/memories/profile.md"


class TestOpenVikingRead:
    def test_overview_read_normalizes_uri_and_unwraps_result(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): {"result": {"content": "overview text"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"}))

        assert result["uri"] == "viking://user/hermes/.overview.md"
        assert result["resolved_uri"] == "viking://user/hermes"
        assert result["level"] == "overview"
        assert result["content"] == "overview text"
        assert provider._client.calls == [(
            "/api/v1/content/overview",
            {"uri": "viking://user/hermes"},
        )]

    def test_full_read_keeps_original_uri(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/read",
                    (("uri", "viking://user/hermes/memories/profile.md"),),
                ): {"result": "full text"},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/memories/profile.md", "level": "full"}))

        assert result["uri"] == "viking://user/hermes/memories/profile.md"
        assert result["resolved_uri"] == "viking://user/hermes/memories/profile.md"
        assert result["level"] == "full"
        assert result["content"] == "full text"
        assert provider._client.calls == [(
            "/api/v1/content/read",
            {"uri": "viking://user/hermes/memories/profile.md"},
        )]

    def test_overview_file_uri_routes_straight_to_content_read_via_stat_probe(self):
        """Pre-check via fs/stat: file URIs skip the directory-only endpoint entirely."""
        provider = OpenVikingMemoryProvider()
        file_uri = "viking://user/hermes/memories/entities/mem_abc.md"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", file_uri),),
                ): {"result": {"isDir": False}},
                (
                    "/api/v1/content/read",
                    (("uri", file_uri),),
                ): {"result": {"content": "full content"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": file_uri, "level": "overview"}))

        assert result["uri"] == file_uri
        assert result["resolved_uri"] == file_uri
        assert result["level"] == "overview"
        assert result["fallback"] == "content/read"
        assert result["content"] == "full content"
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": file_uri}),
            ("/api/v1/content/read", {"uri": file_uri}),
        ]

    def test_overview_dir_uri_skips_stat_when_pseudo_summary(self):
        """Pseudo-URI path already resolves to dir, so no stat probe needed."""
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): {"result": "overview"},
            }
        )

        result = json.loads(provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"}))

        assert result["content"] == "overview"
        # No fs/stat call — normalization already determined it's a directory.
        assert provider._client.calls == [
            ("/api/v1/content/overview", {"uri": "viking://user/hermes"}),
        ]

    def test_overview_directory_uri_uses_stat_probe_then_overview(self):
        """Non-pseudo directory URI: stat → isDir=True → summary endpoint."""
        provider = OpenVikingMemoryProvider()
        dir_uri = "viking://user/hermes/memories"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", dir_uri),),
                ): {"result": {"isDir": True}},
                (
                    "/api/v1/content/overview",
                    (("uri", dir_uri),),
                ): {"result": "dir overview"},
            }
        )

        result = json.loads(provider._tool_read({"uri": dir_uri, "level": "overview"}))

        assert result["content"] == "dir overview"
        assert "fallback" not in result
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": dir_uri}),
            ("/api/v1/content/overview", {"uri": dir_uri}),
        ]

    def test_overview_file_uri_falls_back_via_exception_when_stat_indeterminate(self):
        """If fs/stat raises or returns unknown shape, legacy exception fallback still kicks in."""
        provider = OpenVikingMemoryProvider()
        file_uri = "viking://user/hermes/memories/entities/mem_abc.md"
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/stat",
                    (("uri", file_uri),),
                ): RuntimeError("stat unavailable"),
                (
                    "/api/v1/content/overview",
                    (("uri", file_uri),),
                ): RuntimeError("500 Internal Server Error"),
                (
                    "/api/v1/content/read",
                    (("uri", file_uri),),
                ): {"result": {"content": "fallback full content"}},
            }
        )

        result = json.loads(provider._tool_read({"uri": file_uri, "level": "overview"}))

        assert result["uri"] == file_uri
        assert result["level"] == "overview"
        assert result["fallback"] == "content/read"
        assert result["content"] == "fallback full content"
        assert provider._client.calls == [
            ("/api/v1/fs/stat", {"uri": file_uri}),
            ("/api/v1/content/overview", {"uri": file_uri}),
            ("/api/v1/content/read", {"uri": file_uri}),
        ]

    def test_summary_uri_error_does_not_fallback_and_raises(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/content/overview",
                    (("uri", "viking://user/hermes"),),
                ): RuntimeError("500 Internal Server Error"),
            }
        )

        try:
            provider._tool_read({"uri": "viking://user/hermes/.overview.md", "level": "overview"})
            assert False, "Expected summary endpoint error to be raised"
        except RuntimeError:
            pass

        assert provider._client.calls == [
            ("/api/v1/content/overview", {"uri": "viking://user/hermes"}),
        ]


class TestOpenVikingBrowse:
    def test_list_browse_unwraps_and_normalizes_entry_shapes(self):
        provider = OpenVikingMemoryProvider()
        provider._client = FakeVikingClient(
            {
                (
                    "/api/v1/fs/ls",
                    (("uri", "viking://user/hermes"),),
                ): {
                    "result": {
                        "entries": [
                            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir"},
                            {"rel_path": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "isDir": False, "abstract": "Profile"},
                        ]
                    }
                },
            }
        )

        result = json.loads(provider._tool_browse({"action": "list", "path": "viking://user/hermes"}))

        assert result["path"] == "viking://user/hermes"
        assert result["entries"] == [
            {"name": "memories", "uri": "viking://user/hermes/memories", "type": "dir", "abstract": ""},
            {"name": "profile.md", "uri": "viking://user/hermes/memories/profile.md", "type": "file", "abstract": "Profile"},
        ]
        assert provider._client.calls == [(
            "/api/v1/fs/ls",
            {"uri": "viking://user/hermes"},
        )]


class TestOpenVikingMemoryUriBuilder:
    """Regression tests for _build_memory_uri — fixes #36969.

    Before the fix the URI omitted /agent/{agent}/, causing all agents
    under the same user to share the same memory namespace.
    """

    def _make_provider(self, user="alice", agent="coder"):
        p = OpenVikingMemoryProvider.__new__(OpenVikingMemoryProvider)
        p._user = user
        p._agent = agent
        return p

    def test_uri_layout_includes_agent_segment(self):
        """URI must contain /agent/{agent}/ between user and memories."""
        p = self._make_provider(user="alice", agent="coder")
        uri = p._build_memory_uri("preferences")
        assert uri.startswith("viking://user/alice/agent/coder/memories/preferences/mem_")
        assert uri.endswith(".md")

    def test_uri_uses_configured_agent_not_default(self):
        """_agent value must be interpolated — not hardcoded to 'hermes'."""
        p = self._make_provider(user="alice", agent="research-bot")
        uri = p._build_memory_uri("entities")
        assert "/agent/research-bot/" in uri
        assert "/agent/hermes/" not in uri

    def test_uri_slug_is_twelve_hex_chars_and_unique(self):
        """Slug must be 12 hex chars and differ between calls."""
        import re
        p = self._make_provider()
        uri1 = p._build_memory_uri("preferences")
        uri2 = p._build_memory_uri("preferences")
        slug1 = uri1.split("/mem_")[1].replace(".md", "")
        slug2 = uri2.split("/mem_")[1].replace(".md", "")
        assert re.fullmatch(r"[0-9a-f]{12}", slug1)
        assert re.fullmatch(r"[0-9a-f]{12}", slug2)
        assert slug1 != slug2

    def test_uri_subdir_placed_correctly_for_all_categories(self):
        """All five category subdirs must appear between memories/ and slug."""
        p = self._make_provider(user="u", agent="a")
        subdirs = ["preferences", "entities", "events", "cases", "patterns"]
        for subdir in subdirs:
            uri = p._build_memory_uri(subdir)
            assert f"/memories/{subdir}/mem_" in uri, (
                f"subdir '{subdir}' not placed correctly in URI: {uri}"
            )
