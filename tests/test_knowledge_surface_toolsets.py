from toolsets import (
    get_toolset,
    get_toolset_contract,
    get_toolset_info,
    get_toolset_names,
    resolve_toolset,
    validate_toolset,
)


class TestKnowledgeSurfaceToolsets:
    def test_canonical_knowledge_surface_toolsets_exist(self):
        canonical_names = {
            "repo-code-knowledge",
            "web-research-knowledge",
            "document-pdf-diagram-intelligence",
        }

        for name in canonical_names:
            toolset = get_toolset(name)
            assert toolset is not None
            assert validate_toolset(name)
            assert name in get_toolset_names()

    def test_static_aliases_validate_and_resolve_to_canonical_toolsets(self):
        alias_to_canonical = {
            "repo_code_knowledge": "repo-code-knowledge",
            "repo-knowledge": "repo-code-knowledge",
            "web_research_knowledge": "web-research-knowledge",
            "research-knowledge": "web-research-knowledge",
            "document_intelligence": "document-pdf-diagram-intelligence",
            "document-intelligence": "document-pdf-diagram-intelligence",
        }

        for alias, canonical in alias_to_canonical.items():
            assert validate_toolset(alias)
            alias_toolset = get_toolset(alias)
            canonical_toolset = get_toolset(canonical)
            assert alias_toolset is not None
            assert canonical_toolset is not None
            assert alias_toolset["canonical_name"] == canonical
            assert resolve_toolset(alias) == resolve_toolset(canonical)

    def test_aliases_are_not_reported_as_canonical_toolset_names(self):
        toolset_names = set(get_toolset_names())

        assert "repo-code-knowledge" in toolset_names
        assert "web-research-knowledge" in toolset_names
        assert "document-pdf-diagram-intelligence" in toolset_names

        assert "repo_code_knowledge" not in toolset_names
        assert "web_research_knowledge" not in toolset_names
        assert "document_intelligence" not in toolset_names

    def test_repo_code_knowledge_surface_uses_builtin_file_ast_and_lsp_primitives(self):
        tools = set(resolve_toolset("repo-code-knowledge"))

        assert tools == {
            "read_file",
            "search_files",
            "ast_list_defs",
            "ast_find_nodes",
            "lsp_document_symbols",
            "lsp_definition",
            "lsp_diagnostics",
        }

    def test_web_research_knowledge_surface_uses_current_builtin_web_primitives(self):
        tools = set(resolve_toolset("web-research-knowledge"))

        assert tools == {"web_search", "web_extract"}
        assert "web_crawl" not in tools

    def test_document_intelligence_surface_uses_builtin_browser_and_vision_primitives(self):
        tools = set(resolve_toolset("document-pdf-diagram-intelligence"))

        assert tools == {
            "browser_navigate",
            "browser_snapshot",
            "browser_click",
            "browser_scroll",
            "browser_back",
            "browser_press",
            "browser_get_images",
            "browser_vision",
            "browser_console",
            "vision_analyze",
        }

    def test_canonical_descriptions_use_honest_builtin_wording(self):
        repo_desc = get_toolset("repo-code-knowledge")["description"]
        web_desc = get_toolset("web-research-knowledge")["description"]
        doc_desc = get_toolset("document-pdf-diagram-intelligence")["description"]
        alias_desc = get_toolset("document_intelligence")["description"]

        assert "Canonical built-in repo/code knowledge surface" in repo_desc
        assert "AST structure lookup" in repo_desc

        assert "Canonical built-in web/research knowledge surface" in web_desc
        assert "not an additive MCP crawl stack" in web_desc

        assert "Canonical built-in document/PDF/diagram intelligence surface" in doc_desc
        assert "not a standalone PDF parser or diagram editor" in doc_desc
        assert "Alias for canonical toolset 'document-pdf-diagram-intelligence'." in alias_desc

    def test_builtin_knowledge_surface_contract_marks_builtin_not_additive(self):
        contract = get_toolset_contract("repo_code_knowledge")
        info = get_toolset_info("repo_code_knowledge")

        assert contract == {
            "name": "repo_code_knowledge",
            "canonical_name": "repo-code-knowledge",
            "source": "builtin",
            "is_builtin": True,
            "is_additive": False,
            "is_alias": True,
            "aliases": ["repo_code_knowledge", "repo-knowledge"],
            "is_canonical_knowledge_surface": True,
            "boundary_note": (
                "Canonical built-in control-plane surface; downstream wrappers, propagation, "
                "and additive MCP/plugin productization stay separate."
            ),
            "tools": [
                "ast_find_nodes",
                "ast_list_defs",
                "lsp_definition",
                "lsp_diagnostics",
                "lsp_document_symbols",
                "read_file",
                "search_files",
            ],
        }
        assert info["canonical_name"] == "repo-code-knowledge"
        assert info["source"] == "builtin"
        assert info["is_builtin"] is True
        assert info["is_additive"] is False
        assert info["is_alias"] is True
        assert info["is_canonical_knowledge_surface"] is True
        assert info["boundary_note"] == (
            "Canonical built-in control-plane surface; downstream wrappers, propagation, "
            "and additive MCP/plugin productization stay separate."
        )

    def test_registry_managed_toolsets_are_classified_as_additive(self):
        from tools.registry import registry

        schema = {
            "name": "mcp_test_surface_lookup",
            "description": "Lookup test surface",
            "parameters": {"type": "object", "properties": {}},
        }

        registry.register(
            name="mcp_test_surface_lookup",
            toolset="mcp-test-surface",
            schema=schema,
            handler=lambda args: "{}",
            description="Lookup test surface",
        )
        registry.register_toolset_alias("test-surface", "mcp-test-surface")
        try:
            contract = get_toolset_contract("test-surface")
            assert contract == {
                "name": "test-surface",
                "canonical_name": "mcp-test-surface",
                "source": "mcp",
                "is_builtin": False,
                "is_additive": True,
                "is_alias": True,
                "aliases": ["test-surface"],
                "boundary_note": "Additive registry surface; not a canonical built-in knowledge surface.",
                "tools": ["mcp_test_surface_lookup"],
            }
        finally:
            registry.deregister("mcp_test_surface_lookup")
