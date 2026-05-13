#!/usr/bin/env python3
"""
CJK Script Detector — distinguish Japanese, Chinese, Korean
Uses langdetect (statistical) + Unicode range analysis (deterministic).

Usage:
    python3 scripts/cjk_detect.py "突然のご連絡失礼いたします"
    python3 scripts/cjk_detect.py --check-file /path/to/draft.txt
"""

import sys
import re
import argparse
from collections import Counter

# Unicode ranges
RANGES = {
    "hiragana":     (0x3040, 0x309F),
    "katakana":     (0x30A0, 0x30FF),
    "hangul":       (0xAC00, 0xD7AF),
    "cjk_unified":  (0x4E00, 0x9FFF),
    "halfwidth_kana": (0xFF65, 0xFF9F),
}

# Words that STRONGLY indicate a language (non-overlapping)
SIGNATURE_WORDS = {
    "ja": ["の", "です", "ます", "した", "する", "され", "いただき", "いただく", "致し", "いたし"],
    "zh": ["的", "是", "在", "了", "和", "与", "联系", "我们", "您", "详情", "薪酬", "薪资", "台帳", "台账"],
    "ko": [],  # Hangul is the uniquely Korean script; Hanja overlaps with CJK too much
}

# Simplified Chinese-specific characters (not used in Japanese)
SIMPLIFIED_ONLY = set("来时为个们么对现发后过没都还开关见觉可用".split("") if False else [])
# Traditional/SJIS-based characters (used in Japanese but not simplified Chinese)
JAPANESE_SPECIFIC_KANJI = set()


def unicode_range_report(text: str) -> dict:
    """Return character counts per Unicode range."""
    report = {k: 0 for k in RANGES}
    cjk_chars = []
    other_chars = []

    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in RANGES.items():
            if lo <= cp <= hi:
                report[name] += 1
                if name == "cjk_unified":
                    cjk_chars.append(ch)
                break
        else:
            if not ch.isspace() and not re.match(r"[\x00-\x7F]", ch):
                other_chars.append(ch)

    return {
        "hiragana": report["hiragana"],
        "katakana": report["katakana"],
        "hangul": report["hangul"],
        "cjk_ideographs": len(cjk_chars),
        "cjk_chars_sample": cjk_chars[:20],
        "other_nonlatin": other_chars[:20],
    }


def signature_word_scan(text: str, lang: str) -> list:
    """Find signature words for a language in text."""
    found = []
    for word in SIGNATURE_WORDS.get(lang, []):
        if word in text:
            found.append(word)
    return found


def detect_language(text: str) -> dict:
    """
    Two-layer detection:
    1. langdetect (statistical, probabilistic)
    2. Unicode range analysis (deterministic)
    3. Signature word scan (rule-based)

    Returns a verdict with confidence.
    """
    try:
        from langdetect import detect, detect_langs
        langdetect_langs = detect_langs(text)
        top_lang = langdetect_langs[0].lang
        top_prob = langdetect_langs[0].prob
    except Exception:
        top_lang = None
        top_prob = 0.0
        langdetect_langs = []

    ranges = unicode_range_report(text)

    # Signature word scan
    ja_signatures = signature_word_scan(text, "ja")
    zh_signatures = signature_word_scan(text, "zh")
    ko_signatures = signature_word_scan(text, "ko")

    # Verdict logic
    hiragana = ranges["hiragana"]
    katakana = ranges["katakana"]
    hangul = ranges["hangul"]

    # Korean is easy — Hangul is unique to Korean
    if hangul > 0:
        verdict = "ko"
        reason = "Hangul characters detected (U+AC00–U+D7AF) — uniquely Korean"
        confidence = 0.99
        script = "korean"
    # Hiragana is nearly unique to Japanese (rarely used elsewhere)
    elif hiragana > 0:
        verdict = "ja"
        reason = f"Hiragana detected ({hiragana} chars, U+3040–U+309F) — strongly indicates Japanese"
        confidence = 0.95
        script = "japanese"
    # Katakana + CJK — likely Japanese (for loanwords)
    elif katakana > 0 and ranges["cjk_ideographs"] > 0:
        verdict = "ja"
        reason = f"Katakana ({katakana} chars) + CJK ideographs — likely Japanese"
        confidence = 0.85
        script = "japanese"
    # CJK only — need langdetect + signature words
    elif ranges["cjk_ideographs"] > 0:
        # Strong ZH signatures → override to Chinese even without langdetect
        if zh_signatures:
            verdict = "zh"
            confidence = 0.95
            reason = f"CJK-only text, ZH signatures detected: {zh_signatures}"
            script = "chinese"
        elif top_lang in ("ja", "ko", "zh-cn", "zh-tw", "zh"):
            verdict = top_lang
            confidence = min(top_prob, 0.90)
            reason = f"langdetect: {top_lang} ({top_prob:.2f}), CJK-only text"
            script = {"ja": "japanese", "zh-cn": "chinese", "zh-tw": "chinese", "zh": "chinese", "ko": "korean"}.get(top_lang, "unknown")
        else:
            verdict = "unknown"
            confidence = 0.0
            reason = "Cannot determine — CJK text without distinguishing script features"
            script = "unknown"
    # Latin-only
    elif re.match(r"^[\x00-\x7F\s\.,!?]*$", text):
        verdict = "en"
        confidence = 1.0
        reason = "ASCII/latin characters only"
        script = "english"
    else:
        verdict = "mixed"
        confidence = 0.5
        reason = "Mixed script detected"
        script = "mixed"

    # Override confidence if signature words found
    if ja_signatures:
        confidence = max(confidence, 0.95)
        reason += f" | JA signatures: {ja_signatures}"
    if zh_signatures:
        confidence = max(confidence, 0.95)
        reason += f" | ZH signatures: {zh_signatures}"
    if ko_signatures:
        confidence = max(confidence, 0.95)
        reason += f" | KO signatures: {ko_signatures}"

    return {
        "verdict": verdict,
        "confidence": confidence,
        "script": script,
        "reason": reason,
        "unicode_report": ranges,
        "langdetect_langs": [(str(l), float(p)) for l, p in langdetect_langs],
        "ja_signatures": ja_signatures,
        "zh_signatures": zh_signatures,
        "ko_signatures": ko_signatures,
    }


def verify_output(text: str, expected_script: str) -> dict:
    """
    Verify that output text matches the expected script.
    expected_script: 'english', 'japanese', 'chinese', 'korean'
    """
    result = detect_language(text)
    script = result["script"]

    # Map expected to detected scripts
    expected_to_detected = {
        "english": ["english", "unknown"],
        "japanese": ["japanese"],
        "chinese": ["chinese"],
        "korean": ["korean"],
    }

    allowed = expected_to_detected.get(expected_script, [])
    passed = script in allowed

    return {
        "passed": passed,
        "expected": expected_script,
        "detected": script,
        "confidence": result["confidence"],
        "reason": result["reason"],
        "details": result,
        "message": f"{'✅ PASS' if passed else '❌ FAIL'}: expected {expected_script}, detected {script} (confidence {result['confidence']:.2f})\n  → {result['reason']}",
    }


def print_report(text: str, verbose: bool = False):
    result = detect_language(text)
    print(f"\n📝 Input: {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"\n🎯 Verdict: {result['script'].upper()} ({result['verdict']})")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Reason: {result['reason']}")

    if verbose:
        print(f"\n📊 Unicode Analysis:")
        ur = result["unicode_report"]
        print(f"   Hiragana:      {ur['hiragana']}")
        print(f"   Katakana:      {ur['katakana']}")
        print(f"   Hangul:        {ur['hangul']}")
        print(f"   CJK Ideographs: {ur['cjk_ideographs']}")
        if ur['cjk_chars_sample']:
            print(f"   CJK sample:    {''.join(ur['cjk_chars_sample'][:10])}")

        print(f"\n🔍 Signature Words:")
        print(f"   JA: {result['ja_signatures'] or 'none'}")
        print(f"   ZH: {result['zh_signatures'] or 'none'}")
        print(f"   KO: {result['ko_signatures'] or 'none'}")

        print(f"\n🔎 langdetect: {result['langdetect_langs'][:3]}")


def main():
    parser = argparse.ArgumentParser(description="CJK Script Detector")
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--check", dest="check_file", help="Check file contents")
    parser.add_argument("--verify", dest="verify_script", help="Verify against script (english/japanese/chinese/korean)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.check_file:
        with open(args.check_file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        return

    result = detect_language(text)
    print_report(text, verbose=args.verbose)

    if args.verify_script:
        v = verify_output(text, args.verify_script)
        print(f"\n{v['message']}")


if __name__ == "__main__":
    main()
