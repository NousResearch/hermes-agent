"""Regression coverage for CLI delivery after transform_llm_output streaming."""

from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console
from rich.panel import Panel

from cli import _post_stream_transform_output


def _render_streamed_suffix(monkeypatch, suffix, hook_results):
    import cli as cli_module
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    from hermes_cli import lifecycle

    hook_calls = []
    prose = []
    renderables = []

    def invoke_hook(hook_name, **kwargs):
        hook_calls.append((hook_name, kwargs))
        return list(hook_results)

    monkeypatch.setattr(lifecycle, "invoke_hook", invoke_hook)
    monkeypatch.setattr(cli_module, "_cprint", prose.append)
    monkeypatch.setattr(
        cli_module,
        "ChatConsole",
        lambda: SimpleNamespace(print=renderables.append),
    )

    shell = SimpleNamespace(
        _last_turn_interrupted=False,
        _stream_started=True,
        _stream_box_opened=True,
        _streamed_text_this_turn="original answer",
        final_response_markdown="render",
    )
    response = f"original answer\n\n{suffix}"
    turn = SimpleNamespace(
        use_streaming_tts=False,
        box_opened=False,
        result={
            "completed": True,
            "final_response": response,
            "response_transformed": True,
            "pre_transform_response": "original answer",
        },
    )

    CLIChatTurnMixin._chat_print_response_panel(shell, turn, response)
    return hook_calls, prose, renderables


def test_streamed_transform_prints_only_appended_suffix():
    output = _post_stream_transform_output(
        "original answer\n\n[plugin appended this]",
        {
            "response_transformed": True,
            "pre_transform_response": "original answer",
        },
    )

    assert output == "\n\n[plugin appended this]"


def test_streamed_transform_prints_full_replacement_instead_of_dropping_it():
    output = _post_stream_transform_output(
        "XYZ",
        {
            "response_transformed": True,
            "pre_transform_response": "abc",
        },
    )

    assert output.endswith("\nXYZ")
    assert "abc" not in output


def test_untransformed_stream_has_no_post_stream_output():
    assert _post_stream_transform_output("original answer", {}) == ""


def test_streamed_transform_renders_claimed_directive_as_rich_component(monkeypatch):
    marker = '::gravity-ad{id="ad_123"}'
    component = Panel("Sponsored result")

    hook_calls, prose, renderables = _render_streamed_suffix(
        monkeypatch, marker, [component]
    )

    assert renderables == [component]
    assert marker not in "".join(prose)
    assert hook_calls == [
        (
            "render_cli_transcript_directive",
            {
                "name": "gravity-ad",
                "attrs": {"id": "ad_123"},
                "source": marker,
                "platform": "cli",
            },
        )
    ]


def test_streamed_transform_preserves_unclaimed_directive(monkeypatch):
    marker = "::notice{id='n_123'}"

    hook_calls, prose, renderables = _render_streamed_suffix(monkeypatch, marker, [])

    assert hook_calls[0][1]["attrs"] == {"id": "n_123"}
    assert marker in "".join(prose)
    assert renderables == []


def test_unstreamed_response_renders_claimed_directive_as_component(monkeypatch):
    import cli as cli_module
    from hermes_cli import lifecycle
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

    marker = '::notice{id="n_123"}'
    component = Panel("Notice")
    renderables = []
    monkeypatch.setattr(
        lifecycle,
        "invoke_hook",
        lambda hook_name, **kwargs: [component],
    )
    monkeypatch.setattr(
        cli_module,
        "ChatConsole",
        lambda: SimpleNamespace(print=renderables.append),
    )

    shell = SimpleNamespace(
        _last_turn_interrupted=False,
        _stream_started=False,
        _stream_box_opened=False,
        _streamed_text_this_turn="",
        final_response_markdown="render",
        _scrollback_box_width=lambda: 74,
    )
    response = f"Intro copy.\n\n{marker}"
    turn = SimpleNamespace(
        use_streaming_tts=False,
        box_opened=False,
        result={"completed": True, "final_response": response},
    )

    CLIChatTurnMixin._chat_print_response_panel(shell, turn, response)

    output = StringIO()
    console = Console(file=output, width=80, force_terminal=False, color_system=None)
    for renderable in renderables:
        console.print(renderable)
    assert renderables[-1] is component
    assert "Intro copy." in output.getvalue()
    assert marker not in output.getvalue()


def test_cli_transcript_directive_hook_is_registered():
    from hermes_cli.plugins import VALID_HOOKS

    assert "render_cli_transcript_directive" in VALID_HOOKS


@pytest.mark.parametrize(
    "marker",
    [
        "::gravity-ad{id=ad_123}",
        '::Gravity-ad{id="ad_123"}',
        'prefix ::gravity-ad{id="ad_123"}',
    ],
)
def test_streamed_transform_preserves_malformed_or_nonparagraph_directive(
    monkeypatch, marker
):
    hook_calls, prose, renderables = _render_streamed_suffix(
        monkeypatch, marker, [Panel("must not render")]
    )

    assert hook_calls == []
    assert marker in "".join(prose)
    assert renderables == []
