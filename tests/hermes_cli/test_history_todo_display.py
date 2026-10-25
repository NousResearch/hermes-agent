import copy
import pytest
from hermes_cli.web_routers.sessions import _project_for_display
from tui_gateway.server import _history_to_messages
from tools.todo_tool import TODO_INJECTION_HEADER as HEADER


@pytest.mark.parametrize('shape', ['string', 'parts', 'dict', 'strings'])
@pytest.mark.parametrize('merged', [False, True])
@pytest.mark.parametrize('inline', [False, True])
def test_todo_projection_parity(shape, merged, inline):
    snapshot = HEADER + '\n- [ ] t1. Check result'
    prefix = 'look at this screenshot'
    if shape == 'string':
        content = prefix + '\n\n' + snapshot if merged else snapshot
    elif shape == 'dict':
        content = {'type': 'text', 'text': prefix + '\n\n' + snapshot if merged else snapshot}
    elif shape == 'strings':
        content = [prefix, snapshot] if merged else [snapshot]
    else:
        content = ([{'type': 'text', 'text': prefix}, {'type': 'image_url', 'image_url': {'url': 'https://example.com/x.png'}}] if merged else []) + [{'type': 'text', 'text': '\n\n' + snapshot}]
    row = {'role': 'user', 'content': content}
    original = copy.deepcopy(row)
    rest = _project_for_display([row], inline_images=inline)
    gateway = _history_to_messages([row], image_urls=inline)
    if merged:
        assert HEADER not in str(rest)
        assert HEADER not in str(gateway)
        assert prefix in str(rest) and prefix in str(gateway)
    else:
        assert rest[0]['display_kind'] == 'hidden'
        assert gateway == []
    assert row == original


@pytest.mark.parametrize('text', ['what does ' + HEADER + ' mean? and tell me more', HEADER + '\nand then I said something'])
def test_header_quote_not_a_snapshot(text):
    row = {'role': 'user', 'content': text}
    assert _project_for_display([row])[0]['content'] == text
    assert _history_to_messages([row])[0]['text'] == text
