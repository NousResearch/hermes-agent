"""RED repro for #123945: cache TTL legend alongside Cache pricing."""

from hermes_cli.auth_model_picker import _ModelPickerRows


def test_cache_ttl_legend_shown_with_cache_pricing():
    pricing = {
        "anthropic/claude-sonnet-4": {
            "prompt": "0.000003",
            "completion": "0.000015",
            "input_cache_read": "0.0000003",
        },
    }
    rows = _ModelPickerRows(
        ["anthropic/claude-sonnet-4"], pricing, current_model="",
        sale_chrome=False,
    )
    title = rows.menu_title()
    assert "Cache" in title
    assert "TTL" in title, f"RED: no TTL legend in picker title: {title!r}"


def test_no_ttl_legend_without_cache_pricing():
    pricing = {
        "model-a": {"prompt": "0.000003", "completion": "0.000015"},
    }
    rows = _ModelPickerRows(["model-a"], pricing, current_model="", sale_chrome=False)
    assert "TTL" not in rows.menu_title()
