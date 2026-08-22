from pathlib import Path

from gateway.config import HomeChannel, Platform
from gateway.session import SessionContext, SessionSource, build_session_context_prompt
from gateway.workspace_registry import WorkspaceBinding, WorkspaceRegistry


def _write_registry(tmp_path: Path) -> Path:
    path = tmp_path / "workspaces.yaml"
    path.write_text(
        """
workspaces:
  example:
    name: Example Project
    repo_path: /srv/example
    canonical_repo_url: https://example.org/example/project.git
    default_branch: main
    channels:
      - platform: matrix
        room_id: "!room:example.org"
        response_policy: free_response
      - platform: discord
        scope_id: "server-1"
        channel_id: "channel-1"
        thread_id: "thread-1"
        response_policy: mention_only
      - platform: slack
        workspace_id: "T123"
        channel_id: "C456"
""".strip(),
        encoding="utf-8",
    )
    return path


def test_registry_resolves_matrix_room_binding(tmp_path):
    registry = WorkspaceRegistry(_write_registry(tmp_path))
    source = SessionSource(
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        chat_name="Project - Example",
        chat_type="group",
    )

    assert registry.resolve_source(source) == WorkspaceBinding(
        slug="example",
        name="Example Project",
        repo_path="/srv/example",
        canonical_repo_url="https://example.org/example/project.git",
        default_branch="main",
        response_policy="free_response",
        source=str(registry.config_path),
    )


def test_registry_uses_canonical_scope_id_for_thread_binding(tmp_path):
    registry = WorkspaceRegistry(_write_registry(tmp_path))
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        thread_id="thread-1",
        scope_id="server-1",
        chat_type="thread",
    )

    binding = registry.resolve_source(source)

    assert binding is not None
    assert binding.slug == "example"
    assert binding.response_policy == "mention_only"


def test_registry_rejects_wrong_thread_and_scope(tmp_path):
    registry = WorkspaceRegistry(_write_registry(tmp_path))

    wrong_thread = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        thread_id="other-thread",
        scope_id="server-1",
        chat_type="thread",
    )
    wrong_scope = SessionSource(
        platform=Platform.DISCORD,
        chat_id="channel-1",
        thread_id="thread-1",
        scope_id="other-server",
        chat_type="thread",
    )

    assert registry.resolve_source(wrong_thread) is None
    assert registry.resolve_source(wrong_scope) is None


def test_registry_resolves_legacy_workspace_scope_alias(tmp_path):
    registry = WorkspaceRegistry(_write_registry(tmp_path))
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="C456",
        scope_id="T123",
        chat_type="channel",
    )

    binding = registry.resolve_source(source)

    assert binding is not None
    assert binding.slug == "example"


def test_prompt_includes_authoritative_project_binding():
    context = SessionContext(
        source=SessionSource(
            platform=Platform.MATRIX,
            chat_id="!room:example.org",
            chat_name="Project - Example",
            chat_type="group",
        ),
        connected_platforms=[Platform.MATRIX],
        home_channels={
            Platform.MATRIX: HomeChannel(
                platform=Platform.MATRIX,
                chat_id="!home:example.org",
                name="Home",
            )
        },
        workspace_binding=WorkspaceBinding(
            slug="example",
            name="Example Project",
            repo_path="/srv/example",
            canonical_repo_url="https://example.org/example/project.git",
            default_branch="main",
            response_policy="free_response",
            source="/tmp/workspaces.yaml",
        ),
    )

    prompt = build_session_context_prompt(context)

    assert "**Current Project Binding:**" in prompt
    assert "Example Project (`example`)" in prompt
    assert "Repo path: `/srv/example`" in prompt
    assert "authoritative for side-effecting project work" in prompt
    assert "session-local operational metadata" in prompt
    assert "do not log, quote, screenshot" in prompt


def test_prompt_marks_missing_project_binding_for_gateway_rooms():
    context = SessionContext(
        source=SessionSource(
            platform=Platform.MATRIX,
            chat_id="!unknown:example.org",
            chat_type="group",
        ),
        connected_platforms=[Platform.MATRIX],
        home_channels={},
    )

    prompt = build_session_context_prompt(context)

    assert "No authoritative project binding was found" in prompt
    assert "continue read-only" in prompt
