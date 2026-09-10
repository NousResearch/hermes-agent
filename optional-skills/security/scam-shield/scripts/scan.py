#!/usr/bin/env python3
"""scan.py — deterministic scam/phishing signal scanner for the scam-shield skill.

Reads a message (and any URLs in it), matches it against an extensible pattern
set (``references/patterns.json``) and returns a probabilistic risk assessment.
It speaks in probabilities, not verdicts, and describes the *scheme* present in
the text — it never labels the sender as a person.

Cross-platform: standard library only, ``pathlib`` for paths, no shell calls.
Bilingual output: Russian by default, English via ``--lang en``.

Usage:
    python scripts/scan.py --text "message text here"
    python scripts/scan.py --file message.txt --json
    python scripts/scan.py --text "..." --lang en --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_PATTERNS = Path(__file__).resolve().parent.parent / "references" / "patterns.json"

_URL_RE = re.compile(r"""(?xi)\b(?:https?://|www\.)[^\s<>"'）)]+""")
_IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")

# --- localized strings -------------------------------------------------------

BANDS = {
    "ru": ["низкий", "повышенный", "высокий", "очень высокий"],
    "en": ["low", "elevated", "high", "very high"],
}
CONFIDENCE = {
    "ru": ["низкая", "средняя", "высокая"],
    "en": ["low", "medium", "high"],
}
DISCLAIMER = {
    "ru": ("Оценка вероятностная, а не вердикт. Анализируется схема сообщения, "
           "не личность отправителя. Низкий балл не гарантирует безопасность."),
    "en": ("This is a probabilistic estimate, not a verdict. It analyzes the "
           "message's scheme, not the sender as a person. A low score is not a "
           "guarantee of safety."),
}
URL_SAFE_ACTION = {
    "ru": "Не переходи по ссылке из сообщения — открой сайт вручную из закладки и сверь домен.",
    "en": "Don't open the link from the message — open the site manually from a bookmark and verify the domain.",
}
URL_NOTES = {
    "at_symbol": {
        "ru": "ссылка содержит '@' перед доменом ({u}…) — реальный хост скрыт",
        "en": "the link contains '@' before the domain ({u}…) — the real host is hidden",
    },
    "ip_host": {
        "ru": "ссылка ведёт на голый IP ({h}), а не на домен",
        "en": "the link points to a bare IP ({h}) instead of a domain",
    },
    "punycode": {
        "ru": "домен в punycode ({h}) — вероятна подмена символов",
        "en": "punycode domain ({h}) — likely character substitution",
    },
    "shortener": {
        "ru": "сокращатель ссылок ({b}) прячет настоящий адрес",
        "en": "link shortener ({b}) hides the real address",
    },
    "suspicious_tld": {
        "ru": "подозрительная зона .{t}",
        "en": "suspicious .{t} zone",
    },
    "many_subdomains": {
        "ru": "много поддоменов в хосте ({h})",
        "en": "many subdomains in the host ({h})",
    },
    "lookalike": {
        "ru": "домен маскируется под '{brand}' ({h}) — официальный: {off}",
        "en": "domain impersonates '{brand}' ({h}) — official: {off}",
    },
}
LABELS = {
    "ru": {"risk": "Риск", "conf": "уверенность", "schemes": "Сработавшие схемы:",
           "link": "ссылка", "todo": "Что делать:"},
    "en": {"risk": "Risk", "conf": "confidence", "schemes": "Matched schemes:",
           "link": "link", "todo": "What to do:"},
}

# --- core --------------------------------------------------------------------


def load_patterns(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"pattern file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _noisy_or(weights):
    """P(at least one) = 1 - prod(1-w). Caps below 1.0, diminishing returns."""
    product = 1.0
    for w in weights:
        w = max(0.0, min(1.0, float(w)))
        product *= (1.0 - w)
    return 1.0 - product


def extract_urls(text: str):
    return [m.group(0).rstrip(".,);") for m in _URL_RE.finditer(text)]


def _host_of(url: str) -> str:
    u = re.sub(r"^https?://", "", url, flags=re.I)
    u = re.sub(r"^www\.", "", u, flags=re.I)
    return u.split("/")[0].split("?")[0].split("#")[0].lower()


def _registrable(host: str) -> str:
    labels = host.split(".")
    return ".".join(labels[-2:]) if len(labels) >= 2 else host


def score_urls(urls, url_cfg, lang="ru"):
    """Return (weights, notes) for URL-based signals, notes localized to lang."""
    weights, notes = [], []
    w = url_cfg.get("weights", {})
    brands = url_cfg.get("protected_brands", {})

    def note(key, **kw):
        notes.append(URL_NOTES[key][lang].format(**kw))

    for url in urls:
        host = _host_of(url)
        if not host:
            continue
        if "@" in url.split("//")[-1].split("/")[0]:
            weights.append(w.get("at_symbol_in_url", 0.6)); note("at_symbol", u=url[:40])
        if _IP_HOST_RE.match(host):
            weights.append(w.get("ip_host", 0.55)); note("ip_host", h=host)
        if "xn--" in host:
            weights.append(w.get("punycode", 0.7)); note("punycode", h=host)
        base = host[4:] if host.startswith("www.") else host
        if base in url_cfg.get("shorteners", []):
            weights.append(w.get("shortener", 0.35)); note("shortener", b=base)
        tld = host.rsplit(".", 1)[-1] if "." in host else ""
        if tld in url_cfg.get("suspicious_tlds", []):
            weights.append(w.get("suspicious_tld", 0.4)); note("suspicious_tld", t=tld)
        if len(host.split(".")) >= 5:
            weights.append(w.get("many_subdomains", 0.3)); note("many_subdomains", h=host)
        # Lookalike: brand appears as a distinct token in the host, but the
        # registrable domain is not one of that brand's official domains.
        # Token-boundary matching avoids false positives on unrelated words
        # ("colgate.com"/"gateway.com" not flagged for brand "gate"); for
        # brands >= 6 chars a concatenated prefix ("binancelogin.top") matches.
        reg = _registrable(host)
        host_tokens = [t for t in re.split(r"[^a-z0-9]+", host) if t]
        sld = host.split(".")[-2] if len(host.split(".")) >= 2 else host
        for brand, official in brands.items():
            if reg in official:
                continue
            if brand in host_tokens or (len(brand) >= 6 and sld.startswith(brand)):
                weights.append(w.get("lookalike_domain", 0.75))
                note("lookalike", brand=brand, h=host, off=", ".join(official))
                break
    return weights, notes


def match_signals(text: str, signals):
    lowered = text.lower()
    fired = {}
    for sig in signals:
        hit = any(kw.lower() in lowered for kw in sig.get("keywords", []))
        if not hit:
            for rx in sig.get("regexes", []):
                try:
                    if re.search(rx, lowered):
                        hit = True
                        break
                except re.error:
                    continue
        if hit and sig["id"] not in fired:
            fired[sig["id"]] = sig
    return list(fired.values())


def band(score: int, lang="ru") -> str:
    b = BANDS[lang]
    if score >= 80:
        return b[3]
    if score >= 50:
        return b[2]
    if score >= 20:
        return b[1]
    return b[0]


def build_report(text: str, patterns: dict, lang: str = "ru") -> dict:
    signals = patterns.get("signals", [])
    url_cfg = patterns.get("url_config", {})

    fired = match_signals(text, signals)
    urls = extract_urls(text)
    url_weights, url_notes = score_urls(urls, url_cfg, lang)

    weights = [s.get("weight", 0.5) for s in fired] + url_weights
    score = int(round(_noisy_or(weights) * 100))

    categories = {s.get("scheme_tag", s["id"]) for s in fired}
    if url_weights:
        categories.add("suspicious_link")
    n = len(categories)
    conf = CONFIDENCE[lang][0] if n <= 1 else (CONFIDENCE[lang][1] if n == 2 else CONFIDENCE[lang][2])

    reasons = [
        {"id": s["id"], "scheme": s.get("scheme_tag"), "why": s.get("description")}
        for s in fired
    ]
    key = "safe_action_en" if lang == "en" else "safe_action"
    safe_actions, seen = [], set()
    for s in fired:
        a = s.get(key) or s.get("safe_action")
        if a and a not in seen:
            safe_actions.append(a); seen.add(a)
    if url_notes:
        safe_actions.append(URL_SAFE_ACTION[lang])

    return {
        "risk_score": score,
        "risk_band": band(score, lang),
        "confidence": conf,
        "scheme_tags": sorted(categories),
        "reasons": reasons,
        "url_findings": url_notes,
        "safe_actions": safe_actions,
        "disclaimer": DISCLAIMER[lang],
    }


def _format_human(rep: dict, lang: str = "ru") -> str:
    L = LABELS[lang]
    lines = [f"{L['risk']}: {rep['risk_score']}/100 ({rep['risk_band']}), {L['conf']} {rep['confidence']}"]
    if rep["reasons"]:
        lines.append(L["schemes"])
        for r in rep["reasons"]:
            lines.append(f"  • {r['scheme']}: {r['why']}")
    for note in rep["url_findings"]:
        lines.append(f"  • {L['link']}: {note}")
    if rep["safe_actions"]:
        lines.append(L["todo"])
        for a in rep["safe_actions"]:
            lines.append(f"  → {a}")
    lines.append(rep["disclaimer"])
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Scam/phishing signal scanner.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="message text to scan")
    src.add_argument("--file", help="path to a UTF-8 file with the message")
    p.add_argument("--patterns", default=str(DEFAULT_PATTERNS), help="pattern JSON path")
    p.add_argument("--lang", choices=["ru", "en"], default="ru", help="output language")
    p.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = p.parse_args(argv)

    text = args.text if args.text is not None else Path(args.file).read_text(encoding="utf-8")
    patterns = load_patterns(Path(args.patterns))
    report = build_report(text, patterns, args.lang)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_format_human(report, args.lang))
    return 0


if __name__ == "__main__":
    sys.exit(main())
