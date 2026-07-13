import time
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import cli as cli_mod
from cli import HermesCLI
from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
)


def _make_cli(model: str = "anthropic/claude-sonnet-4-20250514"):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = model
    cli_obj.session_start = datetime.now() - timedelta(minutes=14, seconds=32)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    return cli_obj


def _attach_agent(
    cli_obj,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    api_calls: int,
    context_tokens: int,
    context_length: int,
    compressions: int = 0,
):
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model,
        provider="anthropic" if cli_obj.model.startswith("anthropic/") else None,
        base_url="",
        session_input_tokens=input_tokens if input_tokens is not None else prompt_tokens,
        session_output_tokens=output_tokens if output_tokens is not None else completion_tokens,
        session_cache_read_tokens=cache_read_tokens,
        session_cache_write_tokens=cache_write_tokens,
        session_prompt_tokens=prompt_tokens,
        session_completion_tokens=completion_tokens,
        session_total_tokens=total_tokens,
        session_api_calls=api_calls,
        get_rate_limit_state=lambda: None,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=context_tokens,
            context_length=context_length,
            compression_count=compressions,
        ),
    )
    return cli_obj


_STATUS_BAR_WIDTHS = (51, 52, 75, 76)


def _plugin_status_bar_snapshot():
    """Return a deterministic snapshot that exercises all three text tiers."""
    return {
        "model_short": "m",
        "context_percent": 25,
        "context_length": 200_000,
        "context_tokens": 50_000,
        "duration": "9m",
        "compressions": 0,
        "active_background_tasks": 0,
        "active_background_processes": 0,
        "active_background_subagents": 0,
        "prompt_elapsed": "⏲ 3s",
        "idle_since": "",
    }


def _plugin_status_bar_baseline(width):
    if width < 52:
        return "⚕ m · 9m"
    if width < 76:
        return "⚕ m · 25% · 9m"
    return "⚕ m │ 50K/200K │ 25% │ 9m │ ⏲ 3s"


def _assert_status_bar_hook_call(hook, snapshot):
    assert hook.call_count == 1
    assert hook.call_args.args == ("on_status_bar_render",)
    assert set(hook.call_args.kwargs) == {"snapshot"}
    assert hook.call_args.kwargs["snapshot"] == snapshot
    assert hook.call_args.kwargs["snapshot"] is not snapshot


class _BrokenTruthValue:
    def __bool__(self):
        raise RuntimeError("truth-value failure")


class _BrokenStringValue:
    def __bool__(self):
        return True

    def __str__(self):
        raise RuntimeError("stringification failure")


class TestStatusBarPluginValueHelper:
    def test_invokes_once_and_normalizes_ordered_truthy_values(self):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[None, "", 0, False, "ready", 7, {"state": "ok"}],
        ) as hook:
            values = cli_obj._get_status_bar_plugin_values(snapshot)

        assert values == ["ready", "7", "{'state': 'ok'}"]
        _assert_status_bar_hook_call(hook, snapshot)

    def test_sanitizes_control_characters_to_keep_footer_on_one_line(self):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        with patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[" ready\nnow\r\x1b[31m\t "],
        ) as hook:
            values = cli_obj._get_status_bar_plugin_values(snapshot)

        assert values == ["ready now  [31m"]
        _assert_status_bar_hook_call(hook, snapshot)

    def test_plugin_cannot_mutate_renderer_snapshot(self):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()

        def mutate_snapshot(*_args, **kwargs):
            kwargs["snapshot"].clear()
            return ["ready"]

        with patch("hermes_cli.plugins.invoke_hook", side_effect=mutate_snapshot):
            assert cli_obj._get_status_bar_plugin_values(snapshot) == ["ready"]

        assert snapshot == _plugin_status_bar_snapshot()

    @pytest.mark.parametrize(
        "unsupported",
        [
            (),
            (value for value in ("generated",)),
            "string aggregate",
            None,
        ],
        ids=("tuple", "generator", "string", "none"),
    )
    def test_rejects_non_list_aggregates(self, unsupported):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        with patch("hermes_cli.plugins.invoke_hook", return_value=unsupported) as hook:
            assert cli_obj._get_status_bar_plugin_values(snapshot) == []

        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize("broken", [_BrokenTruthValue(), _BrokenStringValue()])
    def test_preserves_healthy_values_when_an_element_is_broken(self, broken):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        with patch(
            "hermes_cli.plugins.invoke_hook", return_value=["partial", broken]
        ) as hook:
            assert cli_obj._get_status_bar_plugin_values(snapshot) == ["partial"]

        _assert_status_bar_hook_call(hook, snapshot)

    def test_caches_hook_values_during_repaint_window(self):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        with patch("hermes_cli.plugins.invoke_hook", return_value=["ready"]) as hook, patch(
            "cli.time.monotonic", side_effect=[10.0, 10.5, 11.0]
        ):
            assert cli_obj._get_status_bar_plugin_values(snapshot) == ["ready"]
            assert cli_obj._get_status_bar_plugin_values(snapshot) == ["ready"]
            assert cli_obj._get_status_bar_plugin_values(snapshot) == ["ready"]

        assert hook.call_count == 2


class TestStatusBarPluginTierIntegration:
    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS)
    def test_appends_ordered_truthy_values_with_tier_separator(self, width):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        separator = " · " if width < 76 else " │ "
        expected = separator.join(
            [_plugin_status_bar_baseline(width), "ready", "7"]
        )

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[None, "", 0, False, "ready", 7],
        ) as hook:
            text = cli_obj._build_status_bar_text(width=width)

        assert text == expected
        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS)
    @pytest.mark.parametrize(
        "result_factory",
        [
            lambda: [],
            lambda: [None, "", 0, False],
            lambda: ("tuple",),
            lambda: (value for value in ("generated",)),
            lambda: "string aggregate",
            lambda: None,
        ],
        ids=("empty-list", "falsey-list", "tuple", "generator", "string", "none"),
    )
    def test_omission_forms_preserve_exact_baseline_without_trailing_separator(
        self, width, result_factory
    ):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        separator = " · " if width < 76 else " │ "

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook", return_value=result_factory()
        ) as hook:
            text = cli_obj._build_status_bar_text(width=width)

        assert text == _plugin_status_bar_baseline(width)
        assert not text.endswith(separator)
        _assert_status_bar_hook_call(hook, snapshot)


class TestStatusBarPluginFailuresAndOverflow:
    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS)
    def test_invocation_failure_preserves_exact_baseline(self, width):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook", side_effect=RuntimeError("plugin failure")
        ) as hook:
            text = cli_obj._build_status_bar_text(width=width)

        assert text == _plugin_status_bar_baseline(width)
        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS)
    @pytest.mark.parametrize("broken", [_BrokenTruthValue(), _BrokenStringValue()])
    def test_element_failure_preserves_exact_baseline(self, width, broken):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook", return_value=["partial", broken]
        ) as hook:
            text = cli_obj._build_status_bar_text(width=width)

        separator = " · " if width < 76 else " │ "
        assert text == _plugin_status_bar_baseline(width) + separator + "partial"
        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS)
    def test_lookup_failure_preserves_exact_baseline(self, width, monkeypatch):
        import hermes_cli.plugins as plugins

        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        monkeypatch.delattr(plugins, "invoke_hook")

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(cli_obj, "_is_session_yolo_active", return_value=False):
            text = cli_obj._build_status_bar_text(width=width)

        assert text == _plugin_status_bar_baseline(width)

    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS)
    def test_plugin_values_are_appended_before_display_width_trimming(self, width):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        separator = " · " if width < 76 else " │ "
        untrimmed = separator.join(
            [_plugin_status_bar_baseline(width), "x" * 200]
        )
        expected = cli_obj._trim_status_bar_text(untrimmed, width)

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook", return_value=["x" * 200]
        ) as hook:
            text = cli_obj._build_status_bar_text(width=width)

        assert text == expected
        assert text.endswith("...")
        assert cli_obj._status_bar_display_width(text) <= width
        _assert_status_bar_hook_call(hook, snapshot)


_RENDERERS = ("text", "fragments")


def _render_status_bar(cli_obj, renderer, width):
    if renderer == "text":
        return cli_obj._build_status_bar_text(width=width)
    cli_obj._status_bar_visible = True
    with patch.object(cli_obj, "_get_tui_terminal_width", return_value=width):
        return cli_obj._get_status_bar_fragments()


def _rendered_status_bar_text(rendered):
    if isinstance(rendered, str):
        return rendered
    return "".join(text for _, text in rendered)


def _plugin_status_bar_fragment_baseline(width):
    if width < 52:
        return [
            ("class:status-bar", " ⚕ "),
            ("class:status-bar-strong", "m"),
            ("class:status-bar-dim", " · "),
            ("class:status-bar-dim", "9m"),
            ("class:status-bar", " "),
        ]
    if width < 76:
        return [
            ("class:status-bar", " ⚕ "),
            ("class:status-bar-strong", "m"),
            ("class:status-bar-dim", " · "),
            ("class:status-bar-good", "25%"),
            ("class:status-bar-dim", " · "),
            ("class:status-bar-dim", "9m"),
            ("class:status-bar", " "),
        ]
    return [
        ("class:status-bar", " ⚕ "),
        ("class:status-bar-strong", "m"),
        ("class:status-bar-dim", " │ "),
        ("class:status-bar-dim", "50K/200K"),
        ("class:status-bar-dim", " │ "),
        ("class:status-bar-good", "[██░░░░░░░░]"),
        ("class:status-bar-dim", " "),
        ("class:status-bar-good", "25%"),
        ("class:status-bar-dim", " │ "),
        ("class:status-bar-dim", "9m"),
        ("class:status-bar-dim", " │ "),
        ("class:status-bar-dim", "⏲ 3s"),
        ("class:status-bar", " "),
    ]


_AC4_RESULT_FACTORIES = (
    pytest.param(lambda: [], id="result-empty-list"),
    pytest.param(lambda: [None, "", 0, False], id="result-falsey-list"),
    pytest.param(lambda: ("tuple",), id="result-tuple"),
    pytest.param(
        lambda: (value for value in ("generated",)),
        id="result-generator",
    ),
    pytest.param(lambda: "string aggregate", id="result-string"),
    pytest.param(lambda: None, id="result-none"),
)


class TestStatusBarPluginCrossRendererRegression:
    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS, ids=lambda value: f"width-{value}")
    def test_fragment_renderer_invokes_once_with_snapshot_and_exact_styles(self, width):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ) as get_snapshot, patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook", return_value=["ready"]
        ) as hook:
            fragments = _render_status_bar(cli_obj, "fragments", width)

        separator = " · " if width < 76 else " │ "
        expected = _plugin_status_bar_fragment_baseline(width)[:-1] + [
            ("class:status-bar-dim", separator),
            ("class:status-bar", "ready"),
            ("class:status-bar", " "),
        ]
        assert fragments == expected
        get_snapshot.assert_called_once_with()
        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize(
        "renderer", _RENDERERS, ids=lambda value: f"renderer-{value}"
    )
    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS, ids=lambda value: f"width-{value}")
    @pytest.mark.parametrize("result_factory", _AC4_RESULT_FACTORIES)
    def test_ac4_omission_forms_match_empty_list_without_dangling_separator(
        self, renderer, width, result_factory
    ):
        snapshot = _plugin_status_bar_snapshot()
        baseline_cli = _make_cli()
        candidate_cli = _make_cli()

        with patch.object(
            baseline_cli, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            baseline_cli, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook", return_value=[]
        ):
            empty_render = _render_status_bar(baseline_cli, renderer, width)

        with patch.object(
            candidate_cli, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            candidate_cli, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook", return_value=result_factory()
        ) as hook:
            candidate_render = _render_status_bar(candidate_cli, renderer, width)

        empty_text = _rendered_status_bar_text(empty_render)
        candidate_text = _rendered_status_bar_text(candidate_render)
        separator_glyph = "·" if width < 76 else "│"
        assert candidate_text == empty_text
        assert candidate_text.rstrip().endswith(separator_glyph) is False
        if renderer == "fragments":
            assert candidate_render == empty_render
        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize(
        "renderer", _RENDERERS, ids=lambda value: f"renderer-{value}"
    )
    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS, ids=lambda value: f"width-{value}")
    def test_successful_callback_uses_tier_separator_and_strict_width_bound(
        self, renderer, width
    ):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook", return_value=["x"]
        ) as hook:
            rendered = _render_status_bar(cli_obj, renderer, width)

        separator = " · " if width < 76 else " │ "
        text = _rendered_status_bar_text(rendered)
        assert separator + "x" in text
        assert cli_obj._status_bar_display_width(text) <= width
        if renderer == "fragments":
            assert rendered[-3:] == [
                ("class:status-bar-dim", separator),
                ("class:status-bar", "x"),
                ("class:status-bar", " "),
            ]
        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize(
        "renderer", _RENDERERS, ids=lambda value: f"renderer-{value}"
    )
    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS, ids=lambda value: f"width-{value}")
    def test_failing_callback_preserves_baseline_and_invokes_once(
        self, renderer, width
    ):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook",
            side_effect=RuntimeError("plugin failure"),
        ) as hook:
            rendered = _render_status_bar(cli_obj, renderer, width)

        if renderer == "text":
            assert rendered == _plugin_status_bar_baseline(width)
        else:
            assert rendered == _plugin_status_bar_fragment_baseline(width)
        assert cli_obj._status_bar_display_width(
            _rendered_status_bar_text(rendered)
        ) <= width
        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize(
        "renderer", _RENDERERS, ids=lambda value: f"renderer-{value}"
    )
    @pytest.mark.parametrize("width", _STATUS_BAR_WIDTHS, ids=lambda value: f"width-{value}")
    def test_overflow_trims_final_combined_text_to_terminal_width(
        self, renderer, width
    ):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        plugin_value = "x" * 200
        separator = " · " if width < 76 else " │ "

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook", return_value=[plugin_value]
        ) as hook:
            rendered = _render_status_bar(cli_obj, renderer, width)

        if renderer == "text":
            untrimmed = separator.join(
                [_plugin_status_bar_baseline(width), plugin_value]
            )
        else:
            baseline = _plugin_status_bar_fragment_baseline(width)
            untrimmed = (
                "".join(text for _, text in baseline[:-1])
                + separator
                + plugin_value
                + " "
            )
            assert rendered[0][0] == "class:status-bar"
            assert len(rendered) == 1
        text = _rendered_status_bar_text(rendered)
        assert text == cli_obj._trim_status_bar_text(untrimmed, width)
        assert text.endswith("...")
        assert cli_obj._status_bar_display_width(text) <= width
        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize(
        "renderer", _RENDERERS, ids=lambda value: f"renderer-{value}"
    )
    @pytest.mark.parametrize("width", (51, 52, 76), ids=lambda value: f"width-{value}")
    def test_truthy_non_strings_are_filtered_stringified_and_ordered(
        self, renderer, width
    ):
        cli_obj = _make_cli()
        snapshot = _plugin_status_bar_snapshot()
        separator = " · " if width < 76 else " │ "

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_is_session_yolo_active", return_value=False
        ), patch(
            "hermes_cli.plugins.invoke_hook",
            return_value=[None, 7, "", False, 2.5, 0],
        ) as hook:
            rendered = _render_status_bar(cli_obj, renderer, width)

        text = _rendered_status_bar_text(rendered)
        assert separator.join(("7", "2.5")) in text
        assert cli_obj._status_bar_display_width(text) <= width
        if renderer == "fragments":
            assert rendered[-5:] == [
                ("class:status-bar-dim", separator),
                ("class:status-bar", "7"),
                ("class:status-bar-dim", separator),
                ("class:status-bar", "2.5"),
                ("class:status-bar", " "),
            ]
        _assert_status_bar_hook_call(hook, snapshot)

    @pytest.mark.parametrize(
        ("visible", "model_picker"),
        ((False, None), (True, object())),
        ids=("hidden", "model-picker"),
    )
    def test_hidden_and_model_picker_fragment_renders_are_hook_free(
        self, visible, model_picker
    ):
        cli_obj = _make_cli()
        cli_obj._status_bar_visible = visible
        cli_obj._model_picker_state = model_picker

        with patch.object(cli_obj, "_get_status_bar_snapshot") as get_snapshot, patch(
            "hermes_cli.plugins.invoke_hook"
        ) as hook:
            assert cli_obj._get_status_bar_fragments() == []

        get_snapshot.assert_not_called()
        hook.assert_not_called()

    def test_fragment_fallback_does_not_redispatch_hook(self):
        cli_obj = _make_cli()
        cli_obj._status_bar_visible = True
        snapshot = _plugin_status_bar_snapshot()

        with patch.object(
            cli_obj, "_get_status_bar_snapshot", return_value=snapshot
        ), patch.object(
            cli_obj, "_get_tui_terminal_width", return_value=52
        ), patch.object(
            cli_obj,
            "_is_session_yolo_active",
            side_effect=[RuntimeError("fragment failure"), False],
        ), patch(
            "hermes_cli.plugins.invoke_hook", return_value=["x"]
        ) as hook:
            rendered = cli_obj._get_status_bar_fragments()

        assert rendered == [("class:status-bar", " ⚕ m · 25% · 9m · x ")]
        _assert_status_bar_hook_call(hook, snapshot)

    def test_fragment_snapshot_failure_uses_live_model_and_width_bound(self):
        cli_obj = _make_cli()
        cli_obj._status_bar_visible = True
        cli_obj.agent = SimpleNamespace(model="fallback/live-model-with-a-long-name")

        with patch.object(
            cli_obj,
            "_get_status_bar_snapshot",
            side_effect=RuntimeError("snapshot failure"),
        ), patch.object(cli_obj, "_get_tui_terminal_width", return_value=20):
            rendered = cli_obj._get_status_bar_fragments()

        text = _rendered_status_bar_text(rendered)
        assert "fallback/live" in text
        assert cli_obj.model not in text
        assert cli_obj._status_bar_display_width(text) <= 20

    def test_one_registered_callback_renders_all_eight_combinations_with_metadata(self):
        manager = PluginManager()
        context = PluginContext(
            PluginManifest(name="status-bar-renderer-regression", source="user"),
            manager,
        )
        received = []

        def render_status(**kwargs):
            received.append(kwargs)
            return "x"

        context.register_hook("on_status_bar_render", render_status)
        snapshot = _plugin_status_bar_snapshot()

        with patch("hermes_cli.plugins.get_plugin_manager", return_value=manager):
            for renderer in _RENDERERS:
                for width in _STATUS_BAR_WIDTHS:
                    cli_obj = _make_cli()
                    with patch.object(
                        cli_obj, "_get_status_bar_snapshot", return_value=snapshot
                    ), patch.object(
                        cli_obj, "_is_session_yolo_active", return_value=False
                    ):
                        rendered = _render_status_bar(cli_obj, renderer, width)

                    text = _rendered_status_bar_text(rendered)
                    separator = " · " if width < 76 else " │ "
                    assert separator + "x" in text, (renderer, width, rendered)
                    if renderer == "fragments":
                        assert ("class:status-bar", "x") in rendered

        assert manager.has_hook("on_status_bar_render")
        assert len(received) == len(_RENDERERS) * len(_STATUS_BAR_WIDTHS)
        for payload in received:
            assert set(payload) == {"snapshot", "telemetry_schema_version"}
            assert payload["snapshot"] == snapshot
            assert payload["snapshot"] is not snapshot
            assert payload["telemetry_schema_version"] == "hermes.observer.v1"


class TestCLIStatusBar:
    def test_context_style_thresholds(self):
        cli_obj = _make_cli()

        assert cli_obj._status_bar_context_style(None) == "class:status-bar-dim"
        assert cli_obj._status_bar_context_style(10) == "class:status-bar-good"
        assert cli_obj._status_bar_context_style(50) == "class:status-bar-warn"
        assert cli_obj._status_bar_context_style(81) == "class:status-bar-bad"
        assert cli_obj._status_bar_context_style(95) == "class:status-bar-critical"

    def test_build_status_bar_text_for_wide_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "claude-sonnet-4-20250514" in text
        assert "12.4K/200K" in text
        assert "6%" in text
        assert "$0.06" not in text  # cost hidden by default
        assert "15m" in text

    def test_post_compression_sentinel_does_not_render_negative(self):
        """Right after a compression, last_prompt_tokens is parked at the -1
        sentinel until the next API call reports real usage. The status bar
        must clamp it to 0 instead of rendering "-1/200K" / "-1%".
        """
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=-1,
            context_length=200_000,
        )

        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["context_tokens"] == 0
        assert snapshot["context_percent"] == 0

        text = cli_obj._build_status_bar_text(width=120)
        assert "-1" not in text
        assert "0/200K" in text

    def test_input_height_counts_prompt_only_on_first_wrapped_row(self):
        # Regression for prompt_toolkit classic CLI resize glitches: the prompt
        # is inserted by BeforeInput only on logical line 0. At three terminal
        # cells, "⚔ " leaves one cell for the first input character, but
        # wrapped continuation rows use the full three cells. Estimating every
        # wrapped row as one-cell wide over-allocates the TextArea and can leave
        # stale prompt/input cells visible after resize.
        assert cli_mod._estimate_tui_input_height(["abcdef"], "⚔ ", 3) == 3

    def test_input_height_counts_wide_characters_using_cell_width(self):
        # Prompt width (2 cells) + ten CJK chars (20 cells) = 22 display cells,
        # which wraps to two rows at 14 terminal columns.
        assert cli_mod._estimate_tui_input_height(["你" * 10], "❯ ", 14) == 2

    def test_input_height_clamps_zero_width_to_one_cell(self):
        # Some terminals briefly report zero columns during resize. Treat that
        # as a one-cell terminal rather than falling back to a fake wide width.
        assert cli_mod._estimate_tui_input_height(["abcd"], "", 0) == 4

    def test_build_status_bar_text_no_cost_in_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10000,
            completion_tokens=5000,
            total_tokens=15000,
            api_calls=7,
            context_tokens=50000,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=120)
        assert "$" not in text  # cost is never shown in status bar

    def test_build_status_bar_text_collapses_for_narrow_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10000,
            completion_tokens=2400,
            total_tokens=12400,
            api_calls=7,
            context_tokens=12400,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=60)

        assert "⚕" in text
        assert "$0.06" not in text  # cost hidden by default
        assert "15m" in text
        assert "200K" not in text

    def test_build_status_bar_text_handles_missing_agent(self):
        cli_obj = _make_cli()

        text = cli_obj._build_status_bar_text(width=100)

        assert "⚕" in text
        assert "claude-sonnet-4-20250514" in text

    def test_compression_count_shown_in_wide_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=3,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "🗜️ 3" in text

    def test_compression_count_hidden_when_zero(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=0,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "🗜️" not in text

    def test_compression_count_shown_in_medium_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_400,
            total_tokens=12_400,
            api_calls=7,
            context_tokens=12_400,
            context_length=200_000,
            compressions=2,
        )

        text = cli_obj._build_status_bar_text(width=60)

        assert "🗜️ 2" in text

    def test_compression_count_hidden_in_narrow_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_400,
            total_tokens=12_400,
            api_calls=7,
            context_tokens=12_400,
            context_length=200_000,
            compressions=5,
        )

        text = cli_obj._build_status_bar_text(width=50)

        assert "🗜️" not in text

    def test_compression_count_style_thresholds(self):
        cli_obj = _make_cli()

        assert cli_obj._compression_count_style(1) == "class:status-bar-dim"
        assert cli_obj._compression_count_style(4) == "class:status-bar-dim"
        assert cli_obj._compression_count_style(5) == "class:status-bar-warn"
        assert cli_obj._compression_count_style(9) == "class:status-bar-warn"
        assert cli_obj._compression_count_style(10) == "class:status-bar-bad"
        assert cli_obj._compression_count_style(25) == "class:status-bar-bad"

    def test_compression_count_in_wide_fragments(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=7,
        )
        cli_obj._status_bar_visible = True

        frags = cli_obj._get_status_bar_fragments()
        frag_texts = [text for _, text in frags]

        assert "🗜️ 7" in frag_texts
        frag_styles = {text: style for style, text in frags}
        assert frag_styles["🗜️ 7"] == "class:status-bar-warn"

    def test_compression_count_absent_from_fragments_when_zero(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=0,
        )
        cli_obj._status_bar_visible = True

        frags = cli_obj._get_status_bar_fragments()
        frag_texts = [text for _, text in frags]

        assert not any("🗜️" in t for t in frag_texts)

    def test_minimal_tui_chrome_threshold(self):
        cli_obj = _make_cli()

        assert cli_obj._use_minimal_tui_chrome(width=63) is True
        assert cli_obj._use_minimal_tui_chrome(width=64) is False

    def test_bottom_input_rule_hides_on_narrow_terminals(self):
        cli_obj = _make_cli()

        assert cli_obj._tui_input_rule_height("top", width=50) == 1
        assert cli_obj._tui_input_rule_height("bottom", width=50) == 0
        assert cli_obj._tui_input_rule_height("bottom", width=90) == 1

    def test_input_rules_hide_after_resize_until_next_input(self):
        """When _status_bar_suppressed_after_resize is set, both rules hide.

        See _recover_after_resize — column shrink reflows already-rendered
        bars into scrollback, so we hide the separators while the reflow
        settles, then clear the flag (either via the scheduled unsuppress
        timer or the next submitted input).
        """
        cli_obj = _make_cli()
        cli_obj._status_bar_suppressed_after_resize = True

        assert cli_obj._tui_input_rule_height("top", width=90) == 0
        assert cli_obj._tui_input_rule_height("bottom", width=90) == 0

        cli_obj._status_bar_suppressed_after_resize = False
        assert cli_obj._tui_input_rule_height("top", width=90) == 1
        assert cli_obj._tui_input_rule_height("bottom", width=90) == 1

    def test_scheduled_unsuppress_clears_flag_and_repaints_without_input(self):
        """The status bar returns during idle after a resize, without a keypress.

        Regression: the suppression flag was only cleared on the next
        *submitted* input, so a resize/reflow followed by idle left the bar
        hidden indefinitely even while the refresh clock kept ticking. The
        scheduled unsuppress timer must clear the flag and invalidate the app
        on its own.
        """
        cli_obj = _make_cli()
        cli_obj._status_bar_unsuppress_timer = None
        cli_obj._status_bar_suppressed_after_resize = True
        app = MagicMock()
        app.loop = None  # force the synchronous _clear path

        # Schedule with ~0 delay so the timer fires promptly under test.
        cli_obj._schedule_status_bar_unsuppress(app, delay=0.01)
        time.sleep(0.1)

        assert cli_obj._status_bar_suppressed_after_resize is False
        app.invalidate.assert_called()
        # Bar chrome is visible again with no submitted input.
        assert cli_obj._tui_input_rule_height("top", width=90) == 1

    def test_scheduled_unsuppress_debounces_resize_storm(self):
        """A fresh resize cancels the pending unsuppress and restarts it."""
        cli_obj = _make_cli()
        cli_obj._status_bar_unsuppress_timer = None
        cli_obj._status_bar_suppressed_after_resize = True
        app = MagicMock()
        app.loop = None

        # First schedule (long delay) then a second should cancel the first.
        cli_obj._schedule_status_bar_unsuppress(app, delay=5.0)
        first_timer = cli_obj._status_bar_unsuppress_timer
        assert first_timer is not None
        cli_obj._schedule_status_bar_unsuppress(app, delay=0.01)
        assert first_timer is not cli_obj._status_bar_unsuppress_timer
        assert not first_timer.is_alive() or first_timer.finished.is_set()
        time.sleep(0.1)
        assert cli_obj._status_bar_suppressed_after_resize is False

    def test_scrollback_box_width_returns_viewport_width(self):
        """Decorative scrollback boxes use the full viewport width.

        The previous clamp (max 56 cols) was reverted in favour of the
        prompt_toolkit ``_output_screen_diff`` monkey-patch landed in
        #26137, which keeps chrome out of scrollback at the source.
        We accept that an aggressive column-shrink may visually reflow
        already printed Panel borders — that's a cosmetic artifact of
        stamped scrollback history, not a live-render bug.
        """
        from cli import HermesCLI

        # Floor at 32 — narrow terminals still get something usable
        # (avoids negative ``'─' * (w - 2)`` math).
        assert HermesCLI._scrollback_box_width(20) == 32
        assert HermesCLI._scrollback_box_width(32) == 32
        # Above the floor, return the actual viewport width — no cap.
        assert HermesCLI._scrollback_box_width(48) == 48
        assert HermesCLI._scrollback_box_width(80) == 80
        assert HermesCLI._scrollback_box_width(120) == 120
        assert HermesCLI._scrollback_box_width(200) == 200

    def test_agent_spacer_reclaimed_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._agent_running = True

        assert cli_obj._agent_spacer_height(width=50) == 0
        assert cli_obj._agent_spacer_height(width=90) == 1
        cli_obj._agent_running = False
        assert cli_obj._agent_spacer_height(width=90) == 0

    def test_spinner_line_hidden_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "thinking"

        assert cli_obj._spinner_widget_height(width=50) == 0
        assert cli_obj._spinner_widget_height(width=90) == 1
        cli_obj._spinner_text = ""
        assert cli_obj._spinner_widget_height(width=90) == 0

    def test_spinner_height_uses_display_width_for_wide_characters(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "你" * 40
        cli_obj._tool_start_time = 0

        assert cli_obj._spinner_widget_height(width=64) == 2

    def test_spinner_elapsed_format_is_fixed_width_to_reduce_wrap_jitter(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "running tool"

        # Pin the clock: time.monotonic()'s epoch is arbitrary (often near
        # boot), so deriving _tool_start_time from the real monotonic clock
        # made the test fail on hosts where monotonic() < 65.2 — the start
        # time went negative, the (t0 > 0) guard in _render_spinner_text
        # dropped the "(elapsed)" suffix entirely, and the split below hit an
        # IndexError. A fixed clock keeps both elapsed paths deterministic.
        with patch.object(cli_mod.time, "monotonic", return_value=1000.0):
            # <60s path
            cli_obj._tool_start_time = 1000.0 - 9.2
            short = cli_obj._render_spinner_text()

            # >=60s path
            cli_obj._tool_start_time = 1000.0 - 65.2
            long = cli_obj._render_spinner_text()

        short_elapsed = short.split("(", 1)[1].rstrip(")")
        long_elapsed = long.split("(", 1)[1].rstrip(")")

        assert len(short_elapsed) == len(long_elapsed)
        assert "m" in long_elapsed and "s" in long_elapsed

    def test_voice_status_bar_compacts_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = True
        cli_obj._voice_continuous = True

        fragments = cli_obj._get_voice_status_fragments(width=50)

        assert fragments == [("class:voice-status", " 🎤 Ctrl+B ")]

    def test_voice_recording_status_bar_compacts_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = True
        cli_obj._voice_processing = False

        fragments = cli_obj._get_voice_status_fragments(width=50)

        assert fragments == [("class:voice-status-recording", " ● REC ")]

    # Round-13 Copilot review regressions on #19835. The label in voice
    # status bar / recording hint / placeholder must render the
    # configured ``voice.record_key`` — not hardcoded Ctrl+B. Pinning
    # the cache (``set_voice_record_key_cache``) keeps display in sync
    # with the prompt_toolkit binding without re-reading config on
    # every render.
    def test_voice_status_bar_renders_configured_ctrl_letter(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = False
        cli_obj._voice_continuous = False
        cli_obj.set_voice_record_key_cache("ctrl+o")

        wide = cli_obj._get_voice_status_fragments(width=120)
        assert any("Ctrl+O to record" in text for _cls, text in wide)

        compact = cli_obj._get_voice_status_fragments(width=50)
        assert compact == [("class:voice-status", " 🎤 Ctrl+O ")]

    def test_voice_recording_status_bar_renders_configured_named_key(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = True
        cli_obj._voice_processing = False
        cli_obj.set_voice_record_key_cache("ctrl+space")

        fragments = cli_obj._get_voice_status_fragments(width=120)

        assert fragments == [("class:voice-status-recording", " ● REC  Ctrl+Space to stop ")]

    def test_voice_status_bar_falls_back_to_ctrl_b_without_cache(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = False
        cli_obj._voice_continuous = False
        # No cache set — mirrors pre-startup state; fall back to
        # documented Ctrl+B default (Copilot round-13 review).

        compact = cli_obj._get_voice_status_fragments(width=50)

        assert compact == [("class:voice-status", " 🎤 Ctrl+B ")]

    def test_voice_status_bar_renders_malformed_config_as_default(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = False
        cli_obj._voice_continuous = False
        # Non-string / typoed configs fall through the formatter to the
        # documented default so the status bar never advertises an
        # invalid shortcut.
        cli_obj.set_voice_record_key_cache(True)

        compact = cli_obj._get_voice_status_fragments(width=50)

        assert compact == [("class:voice-status", " 🎤 Ctrl+B ")]


class TestCLIUsageReport:
    def test_show_usage_omits_cost_reporting(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=1,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        # Token counts and session metadata still shown.
        assert "Model:" in output
        assert "Input tokens:" in output
        assert "Output tokens:" in output
        assert "Total tokens:" in output
        assert "Session duration:" in output
        assert "Compressions:" in output
        # Cost and cache-hit reporting is removed everywhere.
        assert "Total cost:" not in output
        assert "Cost status:" not in output
        assert "Cost source:" not in output
        assert "Cache read tokens:" not in output
        assert "Cache write tokens:" not in output


class TestStatusBarWidthSource:
    """Ensure status bar fragments don't overflow the terminal width."""

    def _make_wide_cli(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=100_000,
            completion_tokens=5_000,
            total_tokens=105_000,
            api_calls=20,
            context_tokens=100_000,
            context_length=200_000,
        )
        cli_obj._status_bar_visible = True
        return cli_obj

    def test_fragments_fit_within_announced_width(self):
        """Total fragment text length must not exceed the width used to build them."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        for width in (40, 52, 76, 80, 120, 200):
            mock_app = MagicMock()
            mock_app.output.get_size.return_value = MagicMock(columns=width)

            with patch("prompt_toolkit.application.get_app", return_value=mock_app):
                frags = cli_obj._get_status_bar_fragments()

            total_text = "".join(text for _, text in frags)
            display_width = cli_obj._status_bar_display_width(total_text)
            assert display_width <= width, (
                f"At width={width}, fragment total {display_width} cells overflows "
                f"({total_text!r})"
            )

    def test_fragments_use_pt_width_over_shutil(self):
        """When prompt_toolkit reports a width, shutil.get_terminal_size must not be used."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app) as mock_get_app, \
             patch("shutil.get_terminal_size") as mock_shutil:
            cli_obj._get_status_bar_fragments()

        mock_shutil.assert_not_called()

    def test_fragments_fall_back_to_shutil_when_no_app(self):
        """Outside a TUI context (no running app), shutil must be used as fallback."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        with patch("prompt_toolkit.application.get_app", side_effect=Exception("no app")), \
             patch("shutil.get_terminal_size", return_value=MagicMock(columns=100)) as mock_shutil:
            frags = cli_obj._get_status_bar_fragments()

        mock_shutil.assert_called()
        assert len(frags) > 0

    def test_build_status_bar_text_uses_pt_width(self):
        """_build_status_bar_text() must also prefer prompt_toolkit width."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=80)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app), \
             patch("shutil.get_terminal_size") as mock_shutil:
            text = cli_obj._build_status_bar_text()  # no explicit width

        mock_shutil.assert_not_called()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_explicit_width_skips_pt_lookup(self):
        """An explicit width= argument must bypass both PT and shutil lookups."""
        from unittest.mock import patch
        cli_obj = self._make_wide_cli()

        with patch("prompt_toolkit.application.get_app") as mock_get_app, \
             patch("shutil.get_terminal_size") as mock_shutil:
            text = cli_obj._build_status_bar_text(width=100)

        mock_get_app.assert_not_called()
        mock_shutil.assert_not_called()
        assert len(text) > 0


class TestIdleSinceLastTurn:
    """Time-since-last-final-agent-response read-out on the status bar."""

    def test_hidden_before_first_turn(self):
        assert HermesCLI._format_idle_since(None, turn_live=False) == ""

    def test_hidden_while_turn_is_live(self):
        assert HermesCLI._format_idle_since(time.time() - 30, turn_live=True) == ""

    def test_shows_compact_idle_time_after_turn(self):
        label = HermesCLI._format_idle_since(time.time() - 42, turn_live=False)
        assert label.startswith("✓ ")
        assert label == "✓ 42s"

    def test_scales_to_minutes(self):
        label = HermesCLI._format_idle_since(time.time() - 3 * 60, turn_live=False)
        assert label == "✓ 3m"

    def test_snapshot_carries_idle_since(self):
        cli_obj = _make_cli()
        cli_obj._last_turn_finished_at = time.time() - 10
        cli_obj._prompt_start_time = None
        cli_obj._prompt_duration = 5.0
        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["idle_since"].startswith("✓ ")

    def test_snapshot_idle_empty_during_live_turn(self):
        cli_obj = _make_cli()
        cli_obj._last_turn_finished_at = time.time() - 10
        cli_obj._prompt_start_time = time.time()
        cli_obj._prompt_duration = 0.0
        snapshot = cli_obj._get_status_bar_snapshot()
        assert snapshot["idle_since"] == ""

    def test_wide_status_bar_text_includes_idle(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )
        cli_obj._last_turn_finished_at = time.time() - 42
        cli_obj._prompt_start_time = None
        cli_obj._prompt_duration = 7.0
        text = cli_obj._build_status_bar_text(width=160)
        assert "✓ 42s" in text
