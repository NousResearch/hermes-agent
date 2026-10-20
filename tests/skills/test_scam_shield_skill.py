"""Tests for the scam-shield skill. Stdlib + pytest only, no network."""
import importlib.util
from pathlib import Path

import pytest

# Locate the skill's scanner relative to the repo root (tests/skills/ -> repo).
_REPO = Path(__file__).resolve().parents[2]
_SCAN = _REPO / "optional-skills" / "security" / "scam-shield" / "scripts" / "scan.py"


def _load_scan():
    spec = importlib.util.spec_from_file_location("scam_shield_scan", _SCAN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


scan = _load_scan()
PATTERNS = scan.load_patterns(_SCAN.parent.parent / "references" / "patterns.json")


def _findings(text):
    return scan.build_report(text, PATTERNS)["url_findings"]


def _is_lookalike(text):
    return any("маскируется" in n for n in _findings(text))


def test_patterns_file_loads_and_has_signals():
    assert PATTERNS["signals"]
    for sig in PATTERNS["signals"]:
        assert 0.0 <= sig["weight"] <= 1.0
        assert sig["id"] and sig["scheme_tag"] and sig["safe_action"]


def test_protected_brands_have_verified_domains():
    brands = PATTERNS["url_config"]["protected_brands"]
    assert len(brands) >= 20
    for brand, official in brands.items():
        assert official and all("." in d for d in official), brand


def test_seed_phrase_scam_scores_high():
    rep = scan.build_report(
        "Служба безопасности. Срочно подтверди кошелёк и введи сид фразу "
        "на https://metamask-verify.top иначе аккаунт будет заблокирован",
        PATTERNS,
    )
    assert rep["risk_score"] >= 80
    assert rep["risk_band"] in ("высокий", "очень высокий")
    assert "credential_theft" in rep["scheme_tags"]
    assert rep["safe_actions"]


def test_benign_message_scores_low():
    rep = scan.build_report("Привет, скинь презентацию с прошлой встречи, спасибо", PATTERNS)
    assert rep["risk_score"] < 20
    assert rep["risk_band"] == "низкий"
    assert rep["reasons"] == []


def test_score_never_exceeds_100():
    stacked = " ".join(s["keywords"][0] for s in PATTERNS["signals"])
    rep = scan.build_report(stacked + " https://binance-login.xyz", PATTERNS)
    assert 0 <= rep["risk_score"] <= 100


def test_otp_request_flags_account_takeover():
    rep = scan.build_report("Это банк, продиктуй код из смс", PATTERNS)
    assert "account_takeover" in rep["scheme_tags"]


def test_hyphenated_and_bare_brand_lookalikes_detected():
    assert _is_lookalike("open https://metamask-verify.top")   # hyphenated
    assert _is_lookalike("verify at https://metamask.top")     # bare brand
    assert _is_lookalike("login https://binance-login.top")    # hyphenated


def test_concatenated_lookalike_for_long_brand():
    # brand >= 6 chars: concatenated prefix in the SLD is caught
    assert _is_lookalike("go to https://binancelogin.top")


def test_official_domains_never_flagged():
    for url in ("https://metamask.io", "https://support.metamask.io/help",
                "https://binance.com", "https://gate.io", "https://gate.com",
                "https://ledger.com", "https://curve.finance"):
        assert not _is_lookalike(f"open {url}"), url


def test_no_false_positive_on_unrelated_words():
    # short brand token "gate" must NOT flag unrelated words containing it
    for url in ("https://colgate.com", "https://gateway.com", "https://floodgate.io"):
        assert not _is_lookalike(f"see {url}"), url


def test_report_always_has_disclaimer():
    rep = scan.build_report("", PATTERNS)
    assert rep["disclaimer"]
    assert rep["risk_score"] == 0


def test_noisy_or_is_monotonic_and_bounded():
    assert scan._noisy_or([]) == 0.0
    assert 0.0 < scan._noisy_or([0.5]) < 1.0
    assert scan._noisy_or([0.9, 0.9, 0.9]) < 1.0
    assert scan._noisy_or([0.5, 0.5]) > scan._noisy_or([0.5])


def test_english_output_mode():
    rep = scan.build_report(
        "URGENT: enter your seed phrase at https://metamask.top or account blocked",
        PATTERNS, lang="en",
    )
    assert rep["risk_band"] == "very high"
    assert rep["confidence"] in ("low", "medium", "high")
    assert "verdict" in rep["disclaimer"].lower()
    assert any("impersonates" in n for n in rep["url_findings"])
    # no Cyrillic characters leak into English output
    assert not any("\u0400" <= c <= "\u04ff" for a in rep["safe_actions"] for c in a)


def test_default_lang_is_russian():
    rep = scan.build_report("введи сид фразу на https://metamask.top", PATTERNS)
    assert rep["risk_band"] in ("высокий", "очень высокий")
    assert any("маскируется" in n for n in rep["url_findings"])
