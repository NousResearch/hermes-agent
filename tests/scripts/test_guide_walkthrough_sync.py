"""Tests for ``scripts/check_guide_walkthrough_sync.py``.

The checker is the CI gate that keeps the five-step first-run walkthrough
captions identical across the two Persian-guide surfaces: guide.html's
``<figure class="shot">`` figcaptions and web/src/components/
PersianGuide.tsx's WALKTHROUGH array. The tree-level test asserts the
current surfaces are in sync; the unit tests pin the normalizer's contract
(markup stripping, entity unescaping, ZWNJ preservation) and the failure
paths (missing array, missing figures, count/order/caption drift) against
small in-memory fixtures via tmp_path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_guide_walkthrough_sync.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_guide_walkthrough_sync", CHECKER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_guide_walkthrough_sync"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


# ---------------------------------------------------------------------------
# Tree-level gate — the shipped surfaces must currently be in sync.
# ---------------------------------------------------------------------------


def test_current_tree_is_in_sync(checker):
    html_figs = checker.extract_html_figures()
    tsx_figs = checker.extract_tsx_walkthrough()
    assert len(html_figs) == 5
    assert len(tsx_figs) == 5
    assert [src for src, _ in html_figs] == [src for src, _ in tsx_figs]
    assert [cap for _, cap in html_figs] == [cap for _, cap in tsx_figs]


# ---------------------------------------------------------------------------
# Normalizer contract.
# ---------------------------------------------------------------------------


class TestNormalizers:
    def test_html_caption_strips_markup(self, checker):
        raw = "۲ — روی <strong>فارسی</strong> کلیک کنید."
        assert checker._normalize_caption(raw) == "۲ — روی فارسی کلیک کنید."

    def test_html_caption_unescapes_entities(self, checker):
        raw = "a &amp; b &lt;c&gt;"
        assert checker._normalize_caption(raw) == "a & b <c>"

    def test_normalization_collapses_whitespace(self, checker):
        assert checker._normalize_caption("a\n  b\t c") == "a b c"

    def test_zwnj_is_preserved_as_meaningful(self, checker):
        # نیم‌فاصله is real Persian orthography: its presence/absence is drift.
        with_zwnj = checker._normalize_caption("پیش\u200cفرض")
        without_zwnj = checker._normalize_caption("پیشفرض")
        assert with_zwnj != without_zwnj

    def test_zwsp_and_bom_are_dropped(self, checker):
        text = "a\u200bb\ufeffc"
        assert checker._normalize_caption(text) == "abc"

    def test_tsx_and_html_normalize_to_same_form(self, checker):
        html_side = checker._normalize_caption("۱ — <strong>فارسی</strong> را انتخاب کنید.")
        tsx_side = checker._normalize_tsx_caption("۱ — فارسی را انتخاب کنید.")
        assert html_side == tsx_side


# ---------------------------------------------------------------------------
# Extraction failure paths.
# ---------------------------------------------------------------------------


class TestExtraction:
    def test_extract_html_figures_pairs_src_with_caption(self, checker):
        figs = checker.extract_html_figures()
        assert all(src.endswith(".png") for src, _ in figs)
        assert all(cap.startswith(("۱", "۲", "۳", "۴", "۵")) for _, cap in figs)

    def test_missing_walkthrough_array_fails(self, checker, tmp_path, monkeypatch):
        fake = tmp_path / "PersianGuide.tsx"
        fake.write_text("export const Other = 1;\n", encoding="utf-8")
        monkeypatch.setattr(checker, "PERSIAN_GUIDE", fake)
        with pytest.raises(SystemExit, match="no WALKTHROUGH array"):
            checker.extract_tsx_walkthrough()

    def test_missing_figure_class_is_not_extracted(self, checker, tmp_path, monkeypatch):
        fake = tmp_path / "guide.html"
        fake.write_text(
            '<figure class="other"><img src="guide-images/01-x.png">'
            "<figcaption>x</figcaption></figure>",
            encoding="utf-8",
        )
        monkeypatch.setattr(checker, "GUIDE_HTML", fake)
        assert checker.extract_html_figures() == []


# ---------------------------------------------------------------------------
# main() gate behavior on synthetic fixtures (drift simulation without
# touching the real guide files).
# ---------------------------------------------------------------------------


HTML_FIXTURE = """
<figure class="shot">
  <img src="guide-images/01-a.png" loading="lazy">
  <figcaption>۱ — کپشن اول</figcaption>
</figure>
<figure class="shot">
  <img src="guide-images/02-b.png" loading="lazy">
  <figcaption>۲ — کپشن دوم</figcaption>
</figure>
<section id="quickstart">
  <h2>راه‌اندازی سریع در ۶ گام</h2>
  <ol class="steps">
    <li><strong>گام یک</strong> — <code dir="ltr">npm install</code> و بعد <code dir="ltr">pip install -e ".[web,dev]"</code>.</li>
    <li><strong>گام دو</strong> — <code dir="ltr">hermes gateway run</code>.</li>
  </ol>
</section>
"""

TSX_FIXTURE_IN_SYNC = """const WALKTHROUGH: Array<{ src: string; caption: string }> = [
  { src: "guide-images/01-a.png", caption: "۱ — کپشن اول" },
  { src: "guide-images/02-b.png", caption: "۲ — کپشن دوم" },
];

const QUICKSTART: Array<{ cmd: string; desc: string }> = [
  {
    cmd: "npm install",
    desc: "از ریشهٔ مخزن.",
  },
  {
    cmd: "hermes gateway run",
    desc: "اجرای دروازه.",
  },
];
"""


def _install_fixtures(checker, monkeypatch, tmp_path, tsx_text):
    (tmp_path / "guide.html").write_text(HTML_FIXTURE, encoding="utf-8")
    fake_tsx = tmp_path / "PersianGuide.tsx"
    fake_tsx.write_text(tsx_text, encoding="utf-8")
    monkeypatch.setattr(checker, "GUIDE_HTML", tmp_path / "guide.html")
    monkeypatch.setattr(checker, "PERSIAN_GUIDE", fake_tsx)


@pytest.fixture
def gate(checker, tmp_path, monkeypatch):
    def _run(tsx_text: str) -> int:
        _install_fixtures(checker, monkeypatch, tmp_path, tsx_text)
        return checker.main([])

    return _run


class TestGate:
    def test_in_sync_fixture_passes(self, gate):
        assert gate(TSX_FIXTURE_IN_SYNC) == 0

    def test_caption_drift_fails(self, gate):
        drifted = TSX_FIXTURE_IN_SYNC.replace("کپشن اول", "کپشن ۱")
        assert gate(drifted) == 1

    def test_count_drift_fails(self, gate):
        fewer = TSX_FIXTURE_IN_SYNC.replace(
            '  { src: "guide-images/02-b.png", caption: "۲ — کپشن دوم" },\n', ""
        )
        assert gate(fewer) == 1

    def test_missing_figure_fails(self, gate):
        missing = TSX_FIXTURE_IN_SYNC.replace("guide-images/01-a.png", "guide-images/09-x.png")
        assert gate(missing) == 1

    def test_order_drift_fails(self, gate):
        reordered = (
            'const WALKTHROUGH: Array<{ src: string; caption: string }> = [\n'
            '  { src: "guide-images/02-b.png", caption: "۲ — کپشن دوم" },\n'
            '  { src: "guide-images/01-a.png", caption: "۱ — کپشن اول" },\n'
            "];\n"
        )
        assert gate(reordered) == 1


# ---------------------------------------------------------------------------
# Quickstart commands — TSX QUICKSTART.cmd must match guide.html's
# <section id="quickstart"> steps verbatim, in order. Desc prose is free.
# ---------------------------------------------------------------------------


TSX_FIXTURE_WALKTHROUGH_ONLY = """const WALKTHROUGH: Array<{ src: string; caption: string }> = [
  { src: "guide-images/01-a.png", caption: "۱ — کپشن اول" },
  { src: "guide-images/02-b.png", caption: "۲ — کپشن دوم" },
];
"""

QS_HTML_FIXTURE = """
<section id="quickstart">
  <h2>راه‌اندازی سریع در ۶ گام</h2>
  <ol class="steps">
    <li><strong>گام یک</strong> — <code dir="ltr">npm install</code> و بعد <code dir="ltr">pip install -e ".[web,dev]"</code>.</li>
    <li><strong>گام دو</strong> — <code dir="ltr">hermes gateway run</code>.</li>
  </ol>
</section>
"""

QS_TSX_IN_SYNC = """const QUICKSTART: Array<{ cmd: string; desc: string }> = [
  {
    cmd: "npm install",
    desc: "از ریشهٔ مخزن.",
  },
  {
    cmd: "hermes gateway run",
    desc: "اجرای دروازه.",
  },
];
"""

QS_TSX_DRIFTED_CMD = """const QUICKSTART: Array<{ cmd: string; desc: string }> = [
  {
    cmd: "npm ci",
    desc: "از ریشهٔ مخزن.",
  },
  {
    cmd: "hermes gateway run",
    desc: "اجرای دروازه.",
  },
];
"""


def _install_qs_fixtures(checker, monkeypatch, tmp_path, html_text, tsx_text):
    # HTML_FIXTURE already carries the in-sync quickstart section; tsx_text
    # replaces only the TSX side (base + variant QUICKSTART array).
    (tmp_path / "guide.html").write_text(HTML_FIXTURE, encoding="utf-8")
    fake_tsx = tmp_path / "PersianGuide.tsx"
    fake_tsx.write_text(
        TSX_FIXTURE_WALKTHROUGH_ONLY + "\n" + tsx_text, encoding="utf-8"
    )
    monkeypatch.setattr(checker, "GUIDE_HTML", tmp_path / "guide.html")
    monkeypatch.setattr(checker, "PERSIAN_GUIDE", fake_tsx)


@pytest.fixture
def qs_gate(checker, tmp_path, monkeypatch):
    def _run(tsx_text: str) -> int:
        _install_qs_fixtures(checker, monkeypatch, tmp_path, QS_HTML_FIXTURE, tsx_text)
        return checker.main([])

    return _run


class TestQuickstartGate:
    def test_in_sync_fixture_passes(self, qs_gate):
        assert qs_gate(QS_TSX_IN_SYNC) == 0

    def test_cmd_drift_fails(self, qs_gate):
        assert qs_gate(QS_TSX_DRIFTED_CMD) == 1

    def test_desc_prose_is_free(self, qs_gate):
        # Rewording desc on one surface must NOT fail the gate — only the
        # command itself is the contract.
        reworded = QS_TSX_IN_SYNC.replace("از ریشهٔ مخزن.", "روی ریشهٔ مخزن اجرا شود.")
        assert qs_gate(reworded) == 0

    def test_extra_html_probe_commands_do_not_cause_false_drift(self, qs_gate):
        # guide.html steps embed extra probe commands (hermes doctor, pip ...)
        # inside the same step; the TSX row carries only the primary command.
        # Containment semantics must keep this in sync.
        assert qs_gate(QS_TSX_IN_SYNC) == 0

    def test_step_count_drift_fails(self, qs_gate):
        one_fewer = QS_TSX_IN_SYNC.replace(
            '  {\n    cmd: "hermes gateway run",\n    desc: "اجرای دروازه.",\n  },\n', ""
        )
        assert qs_gate(one_fewer) == 1

    def test_reordered_cmds_fail(self, qs_gate):
        reordered = """const QUICKSTART: Array<{ cmd: string; desc: string }> = [
  {
    cmd: "hermes gateway run",
    desc: "اجرای دروازه.",
  },
  {
    cmd: "npm install",
    desc: "از ریشهٔ مخزن.",
  },
];
"""
        assert qs_gate(reordered) == 1


class TestQuickstartExtraction:
    def test_extract_html_steps_normalizes_and_strips_markup(self, checker):
        steps = checker.extract_html_quickstart_steps()
        assert len(steps) == 6
        assert "npm install" in steps[0]
        assert "hermes setup --portal" in steps[1]

    def test_extract_tsx_cmds_returns_command_sequence(self, checker):
        cmds = checker.extract_tsx_quickstart_cmds()
        assert cmds == [
            "npm install",
            "hermes setup --portal",
            "hermes gateway run",
            "hermes dashboard --port 9119",
            "hermes config set display.language fa",
            "start-hermes-stack.cmd",
        ]

    def test_missing_quickstart_section_fails_closed(self, checker, tmp_path, monkeypatch):
        fake = tmp_path / "guide.html"
        fake.write_text(
            '<section id="other"><p>no quickstart here</p></section>',
            encoding="utf-8",
        )
        monkeypatch.setattr(checker, "GUIDE_HTML", fake)
        assert checker.extract_html_quickstart_steps() == []
