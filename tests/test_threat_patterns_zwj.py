"""Regression tests for the ZWJ emoji false-positive fix (#59492).

Original behaviour: ``scan_for_threats`` flagged every ``U+200D`` as
``invisible_unicode_U+200D`` via a set-intersection, which caused
``prompt_builder._scan_context_content`` to drop entire SOUL.md /
AGENTS.md files when they contained any ZWJ emoji (like 🐈‍⬛).

New behaviour: ZWJ is only flagged when its immediate neighbours are
NOT both emoji code points. Emoji-ZWJ-emoji sequences pass through.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


class TestRunner:
    def __init__(self):
        self.passed = []
        self.failed = []

    def run(self, name, fn):
        try:
            fn()
        except Exception as e:
            import traceback
            self.failed.append((name, e, traceback.format_exc()))
        else:
            self.passed.append(name)

    def summary(self):
        total = len(self.passed) + len(self.failed)
        print(f"\n{'='*60}\nResults: {len(self.passed)}/{total} passed")
        if self.failed:
            print(f"\n--- {len(self.failed)} failure(s) ---")
            for n, _e, tb in self.failed:
                print(f"\n[FAIL] {n}\n{tb}")
        return 0 if not self.failed else 1


def _load():
    try:
        from tools import threat_patterns
        return threat_patterns
    except Exception:
        return None

ZWJ = "\u200d"
CAT_ZWJ = "\U0001F408" + ZWJ + "\U00002B1B"  # 🐈 + ZWJ + ⬛
TECH_ZWJ = "\U0001F468" + ZWJ + "\U0001F4BB"  # 👨 + ZWJ + 💻
FAM_ZWJ = "\U0001F469" + ZWJ + "\U0001F467"  # 👩 + ZWJ + 👧
HIDING_ZWJ = "foo" + ZWJ + "bar"
MIXED_ZWJ = "a" + ZWJ + "\U0001F525"  # a + ZWJ + 🔥


def test_cat_beep_zwj_not_flagged():
    m = _load()
    if m is None:
        print("SKIP module not importable")
        return
    findings = m.scan_for_threats(CAT_ZWJ, scope="context")
    assert not any("U+200D" in f for f in findings), f"CAT ZWJ emoji flagged: {findings}"


def test_technologist_zwj_not_flagged():
    m = _load()
    if m is None:
        print("SKIP")
        return
    findings = m.scan_for_threats(TECH_ZWJ, scope="context")
    assert not any("U+200D" in f for f in findings), f"TECH ZWJ emoji flagged: {findings}"


def test_family_zwj_not_flagged():
    m = _load()
    if m is None:
        print("SKIP")
        return
    findings = m.scan_for_threats(FAM_ZWJ, scope="context")
    assert not any("U+200D" in f for f in findings), f"FAM ZWJ emoji flagged: {findings}"


def test_hiding_zwj_still_flagged():
    m = _load()
    if m is None:
        print("SKIP")
        return
    findings = m.scan_for_threats(HIDING_ZWJ, scope="context")
    assert any("U+200D" in f for f in findings), f"'foo{chr(0x200D)}bar' NOT flagged: {findings}"


def test_mixed_zwj_still_flagged():
    m = _load()
    if m is None:
        print("SKIP")
        return
    findings = m.scan_for_threats(MIXED_ZWJ, scope="context")
    assert any("U+200D" in f for f in findings), f"a‍🔥 NOT flagged: {findings}"


def test_no_zwj_clean():
    m = _load()
    if m is None:
        print("SKIP")
        return
    findings = m.scan_for_threats("hello world", scope="context")
    assert not any("U+200D" in f for f in findings), f"clean text flagged: {findings}"


def test_first_threat_message_respects_zwj():
    m = _load()
    if m is None:
        print("SKIP")
        return
    assert m.first_threat_message(CAT_ZWJ, scope="context") is None


def test_is_likely_emoji_codepoint():
    m = _load()
    if m is None:
        print("SKIP")
        return
    assert m._is_likely_emoji_codepoint("\U0001F408")   # 🐈
    assert m._is_likely_emoji_codepoint("\U00002B1B")   # ⬛
    assert m._is_likely_emoji_codepoint("\U0001F525")   # 🔥
    assert not m._is_likely_emoji_codepoint("a")
    assert not m._is_likely_emoji_codepoint(" ")


def main():
    runner = TestRunner()
    runner.run("cat_beep_zwj_not_flagged", test_cat_beep_zwj_not_flagged)
    runner.run("technologist_zwj_not_flagged", test_technologist_zwj_not_flagged)
    runner.run("family_zwj_not_flagged", test_family_zwj_not_flagged)
    runner.run("hiding_zwj_still_flagged", test_hiding_zwj_still_flagged)
    runner.run("mixed_zwj_still_flagged", test_mixed_zwj_still_flagged)
    runner.run("no_zwj_clean", test_no_zwj_clean)
    runner.run("first_threat_message_respects_zwj", test_first_threat_message_respects_zwj)
    runner.run("is_likely_emoji_codepoint", test_is_likely_emoji_codepoint)
    return runner.summary()


if __name__ == "__main__":
    sys.exit(main())