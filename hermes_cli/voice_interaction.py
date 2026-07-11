"""Voice interaction helpers for low-latency acknowledgements and spoken text.

This module is deliberately dependency-light: it is used from both the CLI
voice loop and the TTS tool, including tests that run without audio packages.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping

ACK_FAST = "Got it."
ACK_CHECKING = "Checking that now."
ACK_LOOKING = "I'll look at that."
ACK_APPROVAL = "I need approval before I can run that."

EXPRESSIVE_TAGS = {"sighs", "laughs", "whispers", "slow", "excited"}
_EXPRESSIVE_TAG_RE = re.compile(r"\[(sighs|laughs|whispers|slow|excited)\]", re.IGNORECASE)
_ANY_BRACKET_TAG_RE = re.compile(r"\[[a-z][a-z -]{1,30}\]", re.IGNORECASE)

ONES = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen",
}
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}

_ACRONYMS = {
    "BGP": "B G P", "OSPF": "O S P F", "STP": "S T P", "RSTP": "R S T P",
    "LACP": "L A C P", "DHCP": "D H C P", "DNS": "D N S", "SNMP": "S N M P",
    "SSO": "S S O", "MFA": "M F A", "OU": "O U", "AD": "A D", "LDAP": "L D A P",
    "S3": "S three", "MTU": "M T U", "TCP": "T C P", "UDP": "U D P", "DC": "D C",
    "AE": "A E", "ETH": "E T H",
}
_TERMS = {
    "TrueNAS": "true nas", "MinIO": "min I O", "Authentik": "authentik", "Ansible": "ansible",
    "Proxmox": "proxmox", "Kerberos": "kerberos", "NetBox": "net box", "FortiGate": "forti gate",
    "WireGuard": "wire guard", "RustFS": "rust F S", "NDFC": "N D F C", "vSphere": "V sphere",
}

_SLOW_INTENT_RE = re.compile(
    r"\b(check|look up|find|search|web|google|read|open|file|repo|git|run|shell|command|test|build|deploy|install|debug|fix|create|write|edit|analy[sz]e|compare|summari[sz]e|delegate|research)\b",
    re.IGNORECASE,
)
_APPROVAL_INTENT_RE = re.compile(r"\b(sudo|rm\s+-rf|delete|drop|reboot|shutdown|restart service|kubectl delete|force push)\b", re.IGNORECASE)


def _words_under_100(n: int) -> str:
    if n < 20:
        return ONES[n]
    q, r = divmod(n, 10)
    return TENS[q] if r == 0 else f"{TENS[q]} {ONES[r]}"


def number_words(n: int, *, compact_hundreds: bool = False) -> str:
    """Return concise spoken English for small technical numbers.

    ``compact_hundreds`` speaks 192 as "one ninety two", which is clearer for
    IP octets and ports than formal "one hundred ninety two".
    """
    if n < 0:
        return "minus " + number_words(-n, compact_hundreds=compact_hundreds)
    if n < 100:
        return _words_under_100(n)
    if n < 1000:
        h, r = divmod(n, 100)
        if compact_hundreds and r:
            return f"{ONES[h]} {_words_under_100(r)}"
        return f"{ONES[h]} hundred" + (f" {_words_under_100(r)}" if r else "")
    if n < 10000:
        th, r = divmod(n, 1000)
        return f"{number_words(th)} thousand" + (f" {number_words(r, compact_hundreds=compact_hundreds)}" if r else "")
    return " ".join(ONES[int(ch)] for ch in str(n))


def _ip_repl(match: re.Match[str]) -> str:
    parts = [number_words(int(p), compact_hundreds=True) for p in match.group(1).split(".")]
    cidr = match.group(2)
    spoken = " dot ".join(parts)
    if cidr:
        spoken += " slash " + number_words(int(cidr[1:]))
    return spoken


def _domain_repl(match: re.Match[str]) -> str:
    token = match.group(0)
    # Leave already-normalized IPs to the IP pass.
    if re.fullmatch(r"\d+(?:\.\d+)+", token):
        return token
    labels = token.split(".")
    return " dot ".join(normalize_technical_text(label, _domain_label=True) for label in labels)


def _svc_repl(match: re.Match[str]) -> str:
    prefix = (match.group(1) or "").rstrip("\\")
    account = "Service account " + match.group(2).replace("_", " ").replace("-", " ").removeprefix("svc ")
    if prefix:
        return f"{prefix.lower()} slash {account}"
    return account


def _spell_acronym_token(token: str) -> str:
    if token.upper() in _ACRONYMS:
        return _ACRONYMS[token.upper()]
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", token)
    if m and m.group(1).upper() in _ACRONYMS:
        digits = m.group(2)
        if len(digits) > 1 and digits.startswith("0"):
            num = " ".join(ONES[int(ch)] for ch in digits)
        else:
            num = number_words(int(digits), compact_hundreds=True)
        return f"{_ACRONYMS[m.group(1).upper()]} {num}"
    if token.isupper() and len(token) <= 4:
        return " ".join(token)
    return token


def normalize_technical_text(text: str, *, _domain_label: bool = False) -> str:
    """Normalize technical text into a speech-friendly transcript."""
    if not text:
        return ""
    s = str(text)
    s = re.sub(r",\s*", ", ", s)
    if not _domain_label:
        s = re.sub(r"\b([A-Z][A-Z0-9_-]*\\)?(svc[_-][A-Za-z0-9_-]+)\b", _svc_repl, s)
        s = re.sub(r"\b(\d{1,3}(?:\.\d{1,3}){3})(/\d{1,2})?\b", _ip_repl, s)
        s = re.sub(r"\b(TCP|UDP)/(\d{1,5})\b", lambda m: f"{_ACRONYMS[m.group(1)]} port {number_words(int(m.group(2)), compact_hundreds=True)}", s)
        s = re.sub(r"(?<!\w):(\d{1,5})\b", lambda m: f" port {number_words(int(m.group(1)), compact_hundreds=True)}", s)
        s = re.sub(r"(?<!\d)/(\d{1,2})\b", lambda m: "slash " + number_words(int(m.group(1))), s)
        s = re.sub(r"\bVLANs?\s+((?:\d+\s*,\s*)+(?:and\s*)?\d+|\d+)\b", _vlan_repl, s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b", _domain_repl, s)
        s = re.sub(r"\b([A-Z]{1,4})=([A-Za-z0-9_-]+)", lambda m: f"{_spell_acronym_token(m.group(1))} {m.group(2)}", s)
    for term, spoken in _TERMS.items():
        s = re.sub(rf"\b{re.escape(term)}\b", spoken, s)
    s = re.sub(r"\b([A-Z]{2,4}|S3|[A-Z]{1,3}\d{1,3}|[A-Za-z]{1,3}\d{1,3})\b", lambda m: _spell_acronym_token(m.group(1)), s)
    s = re.sub(r"\bkea-dhcp(\d+)\b", lambda m: f"kea D H C P {number_words(int(m.group(1)))}", s, flags=re.IGNORECASE)
    s = re.sub(r"\bsystemctl\b", "system C T L", s, flags=re.IGNORECASE)
    if not _domain_label:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def _vlan_repl(match: re.Match[str]) -> str:
    raw = match.group(1)
    nums = [int(n) for n in re.findall(r"\d+", raw)]
    plural = len(nums) != 1 or match.group(0).lower().startswith("vlans")
    prefix = "V lans" if plural else "V lan"
    words = [number_words(n, compact_hundreds=True) for n in nums]
    if len(words) == 1:
        body = words[0]
    elif len(words) == 2:
        body = f"{words[0]} and {words[1]}"
    else:
        body = ", ".join(words[:-1]) + f", and {words[-1]}"
    return f"{prefix} {body}"


def strip_markdown_for_speech(text: str) -> str:
    """Remove structures that should not be spoken literally."""
    s = str(text or "")
    s = re.sub(r"```[\s\S]*?```", " ", s)
    s = re.sub(r"^\s*\|.*\|\s*$", " ", s, flags=re.MULTILINE)
    s = re.sub(r"\{[\s\S]{20,}\}", " ", s) if s.strip().startswith("{") else s
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"\*(.+?)\*", r"\1", s)
    s = re.sub(r"`(.+?)`", r"\1", s)
    s = re.sub(r"^#+\s*", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*[-*]\s+", "", s, flags=re.MULTILINE)
    s = re.sub(r"---+", "", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def resolve_voice_tts_profile(voice_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return voice config with the active ``voice.tts_profile`` applied.

    Profiles are intentionally conservative: they can disable normalization or
    expressive tags for a mode, but they do not enable expressive tags unless
    the user also explicitly set ``expressive_tags_enabled``.
    """
    if not isinstance(voice_config, Mapping):
        return {}
    resolved = dict(voice_config)
    profiles = voice_config.get("tts_profiles")
    profile_name = str(voice_config.get("tts_profile") or "").strip()
    profile = (
        profiles.get(profile_name)
        if isinstance(profiles, Mapping) and profile_name
        else None
    )
    if not isinstance(profile, Mapping):
        return resolved
    resolved["active_tts_profile"] = profile_name
    if "normalize" in profile:
        resolved["normalize"] = bool(profile.get("normalize"))
    if "expressive_tags" in profile and not bool(profile.get("expressive_tags")):
        resolved["expressive_tags_enabled"] = False
    return resolved


def expressive_tags_allowed(config: Mapping[str, Any], model_id: str) -> bool:
    voice_cfg = config.get("voice", {}) if isinstance(config, Mapping) else {}
    tts_cfg = config.get("tts", {}) if isinstance(config, Mapping) else {}
    if not isinstance(voice_cfg, Mapping):
        voice_cfg = {}
    voice_cfg = resolve_voice_tts_profile(voice_cfg)
    if not isinstance(tts_cfg, Mapping):
        tts_cfg = {}
    enabled = bool(voice_cfg.get("expressive_tags_enabled", False) or tts_cfg.get("expressive_tags_enabled", False))
    if not enabled:
        return False
    allowlist = voice_cfg.get("expressive_tag_model_allowlist") or tts_cfg.get("expressive_tag_model_allowlist") or []
    if isinstance(allowlist, str):
        allowlist = [item.strip() for item in allowlist.split(",") if item.strip()]
    normalized = (model_id or "").strip().lower().rsplit("/", 1)[-1]
    return normalized in {str(item).strip().lower().rsplit("/", 1)[-1] for item in allowlist}


def strip_unsupported_expressive_tags(text: str, *, config: Mapping[str, Any] | None = None, model_id: str = "") -> str:
    if config is not None and expressive_tags_allowed(config, model_id):
        # Keep at most one whitelisted tag; strip extras and all unknown tags.
        seen = False
        def keep_one(m: re.Match[str]) -> str:
            nonlocal seen
            if seen:
                return ""
            seen = True
            return f"[{m.group(1).lower()}]"
        s = _EXPRESSIVE_TAG_RE.sub(keep_one, text)
        return _ANY_BRACKET_TAG_RE.sub(lambda m: m.group(0) if _EXPRESSIVE_TAG_RE.fullmatch(m.group(0)) else "", s)
    return _ANY_BRACKET_TAG_RE.sub("", text)


def prepare_spoken_text(text: str, *, config: Mapping[str, Any] | None = None, model_id: str = "") -> str:
    """Clean final text for TTS: no markdown/code/JSON spam, then technical normalization."""
    voice_cfg: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        maybe_voice = config.get("voice", {})
        voice_cfg = resolve_voice_tts_profile(
            maybe_voice if isinstance(maybe_voice, Mapping) else {}
        )
    s = strip_markdown_for_speech(text)
    s = strip_unsupported_expressive_tags(s, config=config, model_id=model_id)
    if voice_cfg.get("normalize", True):
        s = normalize_technical_text(s)
    return re.sub(r"\s+", " ", s).strip()


def choose_voice_acknowledgement(user_text: Any, *, requires_approval: bool = False) -> str:
    """Return a short acknowledgement for likely-slow voice turns, or ''."""
    if requires_approval:
        return ACK_APPROVAL
    if not isinstance(user_text, str):
        return ACK_CHECKING
    text = user_text.strip()
    if not text:
        return ""
    if _APPROVAL_INTENT_RE.search(text):
        return ACK_APPROVAL
    if _SLOW_INTENT_RE.search(text):
        if re.search(r"\b(check|run|debug|fix|deploy|install|test|build|research|search|web|file|repo)\b", text, re.IGNORECASE):
            return ACK_CHECKING
        words = len(text.split())
        if words <= 5:
            return ACK_FAST
        return ACK_LOOKING
    return ""


def elevenlabs_pronunciation_locators(tts_config: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return configured ElevenLabs pronunciation dictionary locators.

    Accepts either dictionaries already in ElevenLabs SDK shape or bare locator
    IDs under ``tts.elevenlabs.pronunciation_dictionary_locators``.
    """
    el = tts_config.get("elevenlabs", {}) if isinstance(tts_config, Mapping) else {}
    if not isinstance(el, Mapping):
        return []
    raw = el.get("pronunciation_dictionary_locators") or el.get("pronunciation_dictionaries") or []
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    result: list[dict[str, str]] = []
    if not isinstance(raw, Iterable):
        return result
    for item in raw:
        if isinstance(item, Mapping):
            locator_id = str(item.get("pronunciation_dictionary_id") or item.get("id") or "").strip()
            version_id = str(item.get("version_id") or item.get("pronunciation_dictionary_version_id") or "").strip()
        else:
            locator_id = str(item).strip()
            version_id = ""
        if locator_id:
            entry = {"pronunciation_dictionary_id": locator_id}
            if version_id:
                entry["version_id"] = version_id
            result.append(entry)
    return result
