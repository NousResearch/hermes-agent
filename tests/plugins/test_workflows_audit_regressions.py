from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JS = ROOT / "plugins/workflows/dashboard/dist/index.js"
CSS = ROOT / "plugins/workflows/dashboard/dist/style.css"


def test_feed_panel_cannot_share_the_fixed_editor_column():
    text = JS.read_text(encoding="utf-8")
    body = text[text.index('className: "hermes-workflows-body"'):]
    assert body.index("renderInputFeedPanel()") < body.index("renderBottomPanel()")
    css = CSS.read_text(encoding="utf-8")
    assert ".hermes-workflows-run-mode" in css
