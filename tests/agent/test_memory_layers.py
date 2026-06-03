import json

from agent.memory_layers import (
    CavemanCompressor,
    LayeredMemoryRouter,
    MemPalaceAdapter,
    MemoryDestination,
    MemoryKind,
)


def test_router_keeps_preferences_in_curated_user_memory():
    decision = LayeredMemoryRouter().route(
        "User prefers concise replies and dislikes noisy cron jobs."
    )

    assert decision.kind == MemoryKind.USER_PREFERENCE
    assert decision.primary == MemoryDestination.CURATED_USER
    assert decision.action == "remember"
    assert "preference" in decision.reason.lower()


def test_router_sends_workflows_to_skills_not_curated_memory():
    decision = LayeredMemoryRouter().route(
        "When filing classified recipes, extract ingredients, tags, and save canonical source links to Obsidian."
    )

    assert decision.kind == MemoryKind.PROCEDURE
    assert decision.primary == MemoryDestination.SKILL
    assert decision.action == "create_or_update_skill"


def test_router_sends_artifacts_to_domain_store_with_canonical_pointer():
    decision = LayeredMemoryRouter().route(
        "File this recipe https://example.com/cookies under breakfast in the Obsidian recipe vault."
    )

    assert decision.kind == MemoryKind.ARTIFACT
    assert decision.primary == MemoryDestination.DOMAIN_STORE
    assert "obsidian" in decision.domain.lower()
    assert decision.action == "store_canonical_artifact"


def test_router_refuses_stale_task_progress_for_memory():
    decision = LayeredMemoryRouter().route("PR #123 merged and phase 2 is complete")

    assert decision.kind == MemoryKind.EPHEMERAL_PROGRESS
    assert decision.primary == MemoryDestination.NONE
    assert decision.action == "skip"


def test_caveman_compressor_removes_ai_padding_but_keeps_facts():
    text = """
    Great question! I'd be happy to help. It is important to note that the project uses pytest with xdist.
    In conclusion, we should run the focused tests before finalizing.
    """

    compressed = CavemanCompressor().compress(text)

    assert "Great question" not in compressed
    assert "I'd be happy" not in compressed
    assert "project uses pytest with xdist" in compressed
    assert "run the focused tests" in compressed
    assert len(compressed) < len(text)


def test_mempalace_adapter_builds_scoped_cli_command_without_running_network():
    adapter = MemPalaceAdapter(binary="mempalace", scope="profile:default/channel:personal", enabled=True)

    cmd = adapter.build_search_command("recipe filing workflow", limit=4)

    assert cmd[:3] == ["mempalace", "search", "recipe filing workflow"]
    assert "--limit" in cmd and "4" in cmd
    assert "--scope" in cmd and "profile:default/channel:personal" in cmd


def test_mempalace_adapter_disabled_returns_empty_prefetch():
    adapter = MemPalaceAdapter(enabled=False)

    assert adapter.prefetch("anything") == ""


def test_layered_memory_tool_routes_and_compresses_json():
    from plugins.memory.layered import LayeredMemoryProvider

    provider = LayeredMemoryProvider(config={"mempalace_enabled": False})
    provider.initialize(session_id="s1", platform="discord", agent_identity="default", chat_name="Recipes")
    raw = provider.handle_tool_call(
        "memory_route",
        {"content": "Great question! User prefers recipe notes in Obsidian with canonical source links."},
    )
    payload = json.loads(raw)

    assert payload["success"] is True
    assert payload["decision"]["primary"] == "curated_user"
    assert "Great question" not in payload["compressed_content"]
