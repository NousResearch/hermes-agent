"""Activity remains readable and distinct when wrapping or disabling color."""
import pytest
from rich.text import Text

from hermes_cli.cli_tool_activity import render_tool_activity


@pytest.mark.parametrize('prefix', ['┊', '╎', '┃'])
def test_activity_wraps_with_rail_preserves_content_and_marks_failure(monkeypatch, prefix):
    monkeypatch.setenv('NO_COLOR', '1')
    from hermes_cli.skin_engine import SkinConfig
    monkeypatch.setattr('hermes_cli.skin_engine.get_active_skin',
                        lambda: SkinConfig(name='fixture', tool_prefix=prefix))
    detail = 'exec_command /a/long/path/to/my/note.md  0.5s'
    for width in (12, 40, 80):
        rendered = render_tool_activity(prefix + ' ⚡ ' + detail, width, failed=True)
        assert '\x1b' not in rendered
        assert rendered.endswith('\n')
        rows = rendered.splitlines()
        assert all(row.startswith('  │ ') for row in rows)
        assert all(Text.from_ansi(row).cell_len <= width for row in rows)
        joined = ''.join(row.removeprefix('  │ ').replace(' ', '') for row in rows)
        assert 'Failed·' + detail.replace(' ', '') == joined
