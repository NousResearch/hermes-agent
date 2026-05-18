from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]


def _css_rule(source: str, selector: str) -> str:
    match = re.search(rf"{re.escape(selector)}\s*\{{(?P<body>.*?)\}}", source, re.DOTALL)
    assert match is not None, f"missing CSS rule for {selector}"
    return match.group("body")


def _chat_page_source() -> str:
    return (ROOT / "web" / "src" / "pages" / "ChatPage.tsx").read_text(
        encoding="utf-8"
    )


def test_dashboard_document_can_grow_beyond_viewport_for_browser_zoom_pan() -> None:
    css = (ROOT / "web" / "src" / "index.css").read_text(encoding="utf-8")

    html_rule = _css_rule(css, "html")
    body_rule = _css_rule(css, "body")
    root_rule = _css_rule(css, "#root")

    assert not re.search(r"(?m)^\s*height:\s*100dvh\s*;", html_rule)
    assert "max-height:" not in html_rule
    assert not re.search(r"(?m)^\s*height:\s*100%\s*;", body_rule)
    assert "max-height:" not in root_rule

    assert "min-height:" in html_rule
    assert "min-height:" in body_rule
    assert "min-height:" in root_rule
    assert "overflow: auto" in html_rule
    assert "overflow: auto" in body_rule
    assert "overflow: visible" in root_rule


def test_scaled_vertical_wheel_stays_with_tui_transcript_scroll() -> None:
    chat_page = _chat_page_source()

    assert "function isHorizontalPan(ev: WheelEvent): boolean" in chat_page
    assert "ev.ctrlKey || ev.metaKey || isVisualViewportScaled()" not in chat_page
    assert re.search(
        r"if\s*\(\s*isVisualViewportScaled\(\)\s*\)\s*\{\s*"
        r"return\s+isHorizontalPan\(ev\);\s*\}",
        chat_page,
    )
    assert "if (shouldLetBrowserHandleWheel(ev))" in chat_page
    assert "const delta = ev.deltaY" in chat_page
    assert "wsRef.current.send(seq)" in chat_page


def test_scaled_horizontal_pan_inside_terminal_host_reaches_browser() -> None:
    chat_page = _chat_page_source()

    assert "function isHorizontalPan(ev: WheelEvent): boolean" in chat_page
    assert "function documentCanPanHorizontally(): boolean" in chat_page
    assert re.search(
        r"return\s+documentCanPanHorizontally\(\)\s*&&\s*isHorizontalPan\(ev\)",
        chat_page,
    )
    assert "host.addEventListener(\"wheel\", keepBrowserViewportWheelOutOfTerminal" in chat_page
    assert "capture: true" in chat_page
    assert "passive: true" in chat_page
