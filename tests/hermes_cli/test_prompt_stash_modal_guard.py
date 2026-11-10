"""Ctrl+S never pops a stashed draft into a composer that a modal prompt is about to overwrite."""


def test_ctrl_s_is_inert_while_the_command_palette_is_open(monkeypatch):
    from cli import HermesCLI

    monkeypatch.setenv('HERMES_DEFER_AGENT_STARTUP', '1')
    cli = HermesCLI(model='fixture', provider='openai-compat', api_key='fixture',
                    base_url='http://127.0.0.1:1/v1')
    cli._tui_init_run_state()
    cli._input_area = cli._tui_build_input_area()
    bindings = cli._tui_build_key_bindings()
    cli._prompt_stash.stash('a long draft')
    cli._open_command_palette()
    try:
        assert cli._command_palette_state
        assert not any(b.filter() for b in bindings.get_bindings_for_keys(('c-s',)))
    finally:
        cli._close_command_palette()
    assert len(cli._prompt_stash) == 1
    assert any(b.filter() for b in bindings.get_bindings_for_keys(('c-s',)))
