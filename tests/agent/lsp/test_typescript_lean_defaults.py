"""typescript-language-server starts without the syntax-only tsserver or typingsInstaller, and a
user's ``initialization_options`` still override those defaults."""

from agent.lsp.servers import SERVERS, ServerContext


def _spawn(tmp_path, overrides=None):
    binary = tmp_path / "typescript-language-server"
    binary.write_text("")
    ts = next(s for s in SERVERS if s.server_id == "typescript")
    ctx = ServerContext(workspace_root=str(tmp_path), install_strategy="off", trusted=True,
                        binary_overrides={"typescript": [str(binary)]},
                        init_overrides={"typescript": overrides} if overrides else {})
    return ts.build_spawn(str(tmp_path), ctx).initialization_options


def test_typescript_starts_semantic_server_only(tmp_path):
    init = _spawn(tmp_path)
    assert init["tsserver"]["useSyntaxServer"] == "never"
    assert init["disableAutomaticTypingAcquisition"] is True


def test_user_initialization_options_win(tmp_path):
    init = _spawn(tmp_path, {"tsserver": {"useSyntaxServer": "auto"}, "disableAutomaticTypingAcquisition": False})
    assert init["tsserver"] == {"useSyntaxServer": "auto"}
    assert init["disableAutomaticTypingAcquisition"] is False
