#!/usr/bin/env python3
"""
CJK/JA/ZH/KO Script Detector — production-grade.
Detects language of CJK text and verifies copy-paste purity.

Detection layers (in order):
  1. Unicode Hangul      → Korean (U+AC00–U+D7AF, uniquely Korean)
  2. Unicode Hiragana    → Japanese (U+3040–U+309F, uniquely Japanese)
  3. pykakasi readings   → Japanese vs Chinese (JP kanji get readings, CN chars don't)
  4. CJClassifier        → Statistical fallback for ambiguous CJK-only text

Key insight: pykakasi.convert() tokenizes and returns hiragana/romaji readings.
  - Pure Japanese: ALL tokens have readings
  - Pure Chinese:  SOME tokens have EMPTY readings (chars pykakasi doesn't know)
  - Mixed (contamination): any token with empty reading = contaminating char

Usage:
    python3 scripts/cjk_detect.py "突然のご連絡失礼いたします"     # detect
    python3 scripts/cjk_detect.py "突然のご連絡失礼いたします" -v  # verbose
    python3 scripts/cjk_detect.py --verify ja "日本国の首都は東京"  # check purity
    echo $?   # 0 = clean Japanese, 1 = contaminated

    # Batch
    echo "日本\n中国\n联系我们" | python3 scripts/cjk_detect.py --batch
"""

import sys
import os
import re
import argparse
import warnings

# ── Suppress pykakasi deprecation warnings ────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── pykakasi setup ─────────────────────────────────────────────────────────────
try:
    import pykakasi
    _kks = pykakasi.kakasi()
    PYKAKASI_AVAILABLE = True
except ImportError:
    PYKAKASI_AVAILABLE = False


# ── CJClassifier setup ─────────────────────────────────────────────────────────
CJCLASSIFIER_AVAILABLE = False
_script_dir = os.path.dirname(os.path.abspath(__file__))

for _path in (
    os.path.join(_script_dir, "cjclassifier_lib"),
    os.path.join(_script_dir, "cjclassifier_lib", "cjclassifier_lib"),
):
    if os.path.exists(_path):
        sys.path.insert(0, os.path.dirname(_path))
        try:
            from cjclassifier_lib.classifier import CJClassifier
            CJCLASSIFIER_AVAILABLE = True
            _cj = CJClassifier()
            break
        except ImportError:
            pass

LANG_DISPLAY = {
    "ja":      ("Japanese",         "🇯🇵"),
    "zh-cn":   ("Chinese (Simp)",    "🇨🇳"),
    "zh-tw":   ("Chinese (Trad)",    "🇹🇼"),
    "zh":      ("Chinese",           "🇨🇳"),
    "ko":      ("Korean",            "🇰🇷"),
    "en":      ("English",           "🇬🇧"),
    "mixed":   ("Mixed",             "⚠️"),
    "unknown": ("Unknown",           "❓"),
}


# ── Core detection ────────────────────────────────────────────────────────────

def detect(text: str) -> dict:
    """
    Detect the primary language of text.
    Returns {lang, script, confidence, method, tokens, empty_tokens}
    """
    if not text or not text.strip():
        return _result("unknown", 0.0, "empty input", [], [])

    tokens = []
    if PYKAKASI_AVAILABLE:
        try:
            raw_tokens = _kks.convert(text)
            # Normalize: merge adjacent tokens without reading
            tokens = _normalize_tokens(raw_tokens)
        except Exception:
            pass

    empty_tokens = [t["orig"] for t in tokens if not t["hira"] and not t["hepburn"]]
    all_tokens_n = len(tokens)
    empty_n = len(empty_tokens)

    hiragana_n = sum(1 for ch in text if 0x3040 <= ord(ch) <= 0x309F)
    katakana_n = sum(1 for ch in text if 0x30A0 <= ord(ch) <= 0x30FF)
    hangul_n   = sum(1 for ch in text if 0xAC00 <= ord(ch) <= 0xD7AF)
    cjk_n      = sum(1 for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF)

    # ── Layer 1: Korean (Hangul is unique) ────────────────────────────────────
    if hangul_n > 0:
        return _result("ko", 0.99, "Hangul block (U+AC00–U+D7AF)", tokens, empty_tokens)

    # ── Layer 2: Japanese Hiragana ─────────────────────────────────────────────
    if hiragana_n > 0 and hiragana_n / max(len(text), 1) > 0.03:
        return _result("ja", 0.98, f"Hiragana {hiragana_n} chars — uniquely Japanese", tokens, empty_tokens)

    # ── Layer 3: pykakasi reading analysis ────────────────────────────────────
    if all_tokens_n > 0 and cjk_n > 0:
        empty_ratio = empty_n / all_tokens_n

        # Pure Japanese: no empty readings
        if empty_n == 0 and cjk_n > 0:
            return _result("ja", 0.97, f"All {all_tokens_n} tokens have JP readings", tokens, empty_tokens)

        # Contaminated or mixed: some tokens have no readings
        if empty_n > 0:
            # If more than 30% tokens are empty → Chinese
            if empty_ratio > 0.30:
                return _result(
                    "zh-cn", 0.90,
                    f"pykakasi: {empty_n}/{all_tokens_n} tokens no reading "
                    f"({empty_ratio:.0%}) → likely Chinese",
                    tokens, empty_tokens
                )
            # Some empty tokens → mixed or ambiguous
            else:
                return _result(
                    "mixed", 0.75,
                    f"pykakasi: {empty_n}/{all_tokens_n} tokens have no reading "
                    f"(possible contamination)",
                    tokens, empty_tokens
                )

    # ── Layer 4: CJClassifier ───────────────────────────────────────────────────
    if CJCLASSIFIER_AVAILABLE and cjk_n > 0:
        try:
            classified = _cj.classify(text)
            scores = classified["scores"]
            best_lang = max(scores, key=lambda k: scores[k])
            sorted_scores = sorted(scores.values(), reverse=True)
            margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.5
            confidence = min(0.95, 0.60 + margin / 5)
            return _result(
                best_lang, confidence,
                f"CJClassifier: {best_lang} ({confidence:.2f})",
                tokens, empty_tokens
            )
        except Exception:
            pass

    # ── Layer 5: Signature words ───────────────────────────────────────────────
    ja_sigs = _scan_signatures(text, "ja")
    zh_sigs = _scan_signatures(text, "zh")
    if ja_sigs and not zh_sigs:
        return _result("ja", 0.75, f"JA signatures: {ja_sigs}", tokens, empty_tokens)
    if zh_sigs and not ja_sigs:
        return _result("zh-cn", 0.75, f"ZH signatures: {zh_sigs}", tokens, empty_tokens)
    if ja_sigs and zh_sigs:
        return _result("mixed", 0.60, f"Both: JA={ja_sigs}, ZH={zh_sigs}", tokens, empty_tokens)

    # ── ASCII fallback ──────────────────────────────────────────────────────────
    if re.match(r"^[\x00-\x7F\s\.,!?\'\-\"\:\;]*$", text):
        return _result("en", 1.0, "ASCII/latin only", tokens, empty_tokens)

    return _result("unknown", 0.0, "Cannot determine", tokens, empty_tokens)


def _normalize_tokens(raw_tokens: list) -> list:
    """
    pykakasi returns a flat list of tokens.
    We care about: orig, hira, hepburn.
    """
    return [
        {"orig": t.get("orig", ""), "hira": t.get("hira", ""), "hepburn": t.get("hepburn", "")}
        for t in raw_tokens
    ]


SIGNATURE_WORDS = {
    "ja": {"の", "です", "ます", "した", "する", "され", "いただき", "いただく", "致し", "いたし"},
    "zh": {"的", "是", "在", "了", "和", "与", "联系", "我们", "您", "详情",
           "薪酬", "薪资", "台帳", "台账", "职", "职位", "请发", "发送给"},
}


def _scan_signatures(text: str, lang: str) -> list:
    return [w for w in SIGNATURE_WORDS.get(lang, []) if w in text]


def _result(lang, confidence, method, tokens, empty_tokens):
    name, emoji = LANG_DISPLAY.get(lang, ("Unknown", "❓"))
    return {
        "lang": lang,
        "script": name,
        "confidence": round(confidence, 4),
        "method": method,
        "tokens": tokens,
        "empty_tokens": empty_tokens,
    }


# ── Verification ───────────────────────────────────────────────────────────────

def verify(text: str, expected_lang: str) -> dict:
    """
    Verify text is pure expected_lang.
    Returns {passed, expected, detected, confidence, bad_chars, message}
    bad_chars: contaminating characters (for expected=ja, these are non-JP CJK chars)
    """
    result = detect(text)
    detected = result["lang"]

    # Normalize aliases
    norm = {"zh-cn": "zh-cn", "zh-tw": "zh-tw", "zh": "zh-cn",
            "ja": "ja", "ko": "ko", "en": "en"}
    exp = norm.get(expected_lang, expected_lang)
    det = norm.get(detected, detected)

    # ZH aliasing
    passed = (exp == det) or (exp == "zh-cn" and det in ("zh", "zh-cn"))

    bad_chars = []
    if expected_lang == "ja":
        # For Japanese: empty_tokens = characters pykakasi can't read = contamination
        bad_chars = result["empty_tokens"]
        # Contamination always fails the purity check for copy-paste use
        if bad_chars:
            passed = False

    return {
        "passed": passed,
        "expected": expected_lang,
        "detected": detected,
        "confidence": result["confidence"],
        "method": result["method"],
        "bad_chars": bad_chars,
        "message": _format_verify_message(passed, expected_lang, detected,
                                          result["confidence"], result["method"], bad_chars),
    }


def _format_verify_message(passed, expected, detected, confidence, method, bad_chars):
    emoji = "✅ PASS" if passed else "❌ FAIL"
    msg = f"{emoji}: expected={expected}, detected={detected} " \
          f"(confidence {confidence:.2f})\n  → {method}"
    if bad_chars:
        chars_str = " ".join(bad_chars)
        msg += f"\n  ⚠️  Contaminating chars ({len(bad_chars)}): {chars_str}"
    return msg


# ── Reporter ───────────────────────────────────────────────────────────────────

def print_report(text: str, verbose: bool = False):
    result = detect(text)
    lang_display, emoji = LANG_DISPLAY.get(result["lang"], ("Unknown", "❓"))

    print(f"\n{'='*60}")
    print(f"{emoji} Input:  {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"{'='*60}")
    print(f"🎯 Script:   {lang_display} ({result['lang']})")
    print(f"   Confidence: {result['confidence']:.2f}")
    print(f"   Method:    {result['method']}")

    if verbose and result["tokens"]:
        tokens = result["tokens"]
        print(f"\n📖 pykakasi token analysis ({len(tokens)} tokens):")
        print(f"   {'Token':<8} {'Hiragana':<16} {'Romaji':<14} Status")
        print(f"   {'-'*50}")
        for t in tokens:
            status = "✅ JP" if (t["hira"] or t["hepburn"]) else "❌ EMPTY"
            print(f"   {t['orig']:<8} {t['hira']:<16} {t['hepburn']:<14} {status}")

        if result["empty_tokens"]:
            print(f"\n   ⚠️  Characters without JP readings "
                  f"({len(result['empty_tokens'])}): {result['empty_tokens']}")

    print()


def print_verify(text: str, expected: str):
    v = verify(text, expected)
    print(v["message"])


# ── Batch ──────────────────────────────────────────────────────────────────────

def batch_check(lines: list):
    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        result = detect(line)
        lang_display, emoji = LANG_DISPLAY.get(result["lang"], ("Unknown", "❓"))
        status = "✅" if result["confidence"] >= 0.80 else "⚠️"
        empty_n = len(result["empty_tokens"])
        extra = f" | {empty_n} empty" if empty_n else ""
        print(f"{status} [{result['lang']:<8}] {result['confidence']:.2f}{extra} | {line[:70]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CJK/JA/ZH/KO Detector & Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cjk_detect.py "突然のご連絡失礼いたします"         # detect language
  cjk_detect.py "突然のご連絡失礼いたします" -v      # verbose with token analysis
  cjk_detect.py --verify ja "日本国の首都は東京"     # check purity (exit 0=clean)
  cjk_detect.py --verify zh "联系我们获取更多详情"   # check Chinese purity
  echo "日本\\n中国" | cjk_detect.py --batch         # batch
        """
    )
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show pykakasi token-by-token analysis")
    parser.add_argument("--verify", dest="verify_lang",
                        help="Verify text is pure lang (ja/zh/ko/en). Exit 0=pass, 1=fail")
    parser.add_argument("--check", dest="check_file",
                        help="Check file contents")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: one line per text, stdin")
    args = parser.parse_args()

    if args.check_file:
        with open(args.check_file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        if not sys.stdin.isatty():
            batch_check(sys.stdin.readlines())
            return
        parser.print_help()
        return

    if args.verify_lang:
        print_verify(text, args.verify_lang)
        v = verify(text, args.verify_lang)
        sys.exit(0 if v["passed"] else 1)
    else:
        print_report(text, verbose=args.verbose)


if __name__ == "__main__":
    main()
