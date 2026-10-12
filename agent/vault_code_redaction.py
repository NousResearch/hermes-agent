"""Best-effort explicit-context redaction for registered CVC/OTP values.

Not an exfiltration boundary: bare, relabelled or transformed codes can pass.
No arbitrary browser commands are restricted. Password/PAN policy stays separate.
"""
import json
import re
from html import unescape
from dataclasses import dataclass


@dataclass(frozen=True)
class _JSONNumber:
    raw: str


@dataclass(frozen=True)
class _JSONObject:
    pairs: tuple


def _dump_preserving_numbers(value):
    """Emit parser-owned numbers and ordered pairs without lossy conversion."""
    if isinstance(value, _JSONNumber):
        return value.raw
    if isinstance(value, _JSONObject):
        return "{" + ",".join(json.dumps(key, ensure_ascii=False) + ":" + _dump_preserving_numbers(child)
                              for key, child in value.pairs) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_dump_preserving_numbers(child) for child in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)

_MARKER = "«redacted-vault-secret»"
_LABELS = {
    "cvc": r"cvc|cvv2?|csc|cc[-_ ]?(?:csc|cvc)|card[-_ ]?(?:cvc|cvv)|security[-_ ]?code|card[-_ ]?verification[-_ ]?code|код\s+безопасности",
    "otp": r"otp|totp|otp[-_ ]?code|one[-_ ]?time[-_ ]?code|verification[-_ ]?code|authentication[-_ ]?code|two[-_ ]?factor[-_ ]?code|2fa[-_ ]?code|код\s+подтверждения|одноразовый\s+код",
}
_FIELDS = {
    "cvc": {"cvc", "cvv", "cvv2", "csc", "cccsc", "cccvc", "cardcvc", "cardcvv", "securitycode", "cardverificationcode", "кодбезопасности"},
    "otp": {"otp", "totp", "otpcode", "onetimecode", "verificationcode", "authenticationcode", "twofactorcode", "2facode", "кодподтверждения", "одноразовыйкод"},
}
_DESCRIPTORS = {"name", "id", "autocomplete", "label", "aria-label", "ariaLabel"}
_INPUT = re.compile(r"<input\b[^>]{0,4096}>", re.I)
_ATTRIBUTE = re.compile(r"([\w:-]+)\s*=\s*([\"'])(.*?)\2", re.S)


def _field_kind(name):
    if not isinstance(name, str):
        return None
    normalized = re.sub(r"[^\w]|_", "", name.casefold())
    for kind, fields in _FIELDS.items():
        if normalized in fields:
            return kind
    return None


def redact_registered_codes(text, entries):
    """Mask exact registered code values only in explicit fields/assignments."""
    values = {kind: set() for kind in _LABELS}
    for value, kinds in entries:
        for kind in kinds:
            if kind in values:
                values[kind].add(value)
    values = {kind: group for kind, group in values.items() if group}
    if not values:
        return text

    patterns = {}
    for kind, group in values.items():
        alternatives = "|".join(re.escape(v) for v in sorted(group, key=len, reverse=True))
        # A relationship, not a hotword window: don't mask a nearby unrelated total.
        # Reject a match inside a larger identifier, year, or decimal amount.
        patterns[kind] = re.compile(
            rf"(?<!\w)(?:{_LABELS[kind]})(?!\w)"
            rf"([\"']?\s{{0,16}}[:=]\s{{0,16}}[\"']?|\s{{1,16}}[\"']?)"
            rf"(?P<code>{alternatives})(?!\w|[.,]\d)", re.I,
        )

    def scalar(value, kind):
        if kind in values and isinstance(value, (str, int, _JSONNumber)) and not isinstance(value, bool):
            raw = value.raw if isinstance(value, _JSONNumber) else str(value)
            if raw in values[kind]:
                return _MARKER
        return value

    def structured(value, depth):
        if depth > 16:
            return value
        if isinstance(value, _JSONObject):
            metadata = dict(value.pairs)
            control_kind = next((_field_kind(metadata[key]) for key in _DESCRIPTORS
                                 if key in metadata and _field_kind(metadata[key]) in values), None)
            output = []
            for key, child in value.pairs:
                kind = _field_kind(key) or (control_kind if key == "value" else None)
                cleaned = scalar(child, kind)
                output.append((key, cleaned if cleaned != child else structured(child, depth + 1)))
            return _JSONObject(tuple(output))
        if isinstance(value, list):
            return [structured(child, depth + 1) for child in value]
        if isinstance(value, str):
            return clean_text(value, depth + 1)
        return value

    def html_input(match):
        tag = match.group(0)
        attrs = {m.group(1).casefold(): unescape(m.group(3)) for m in _ATTRIBUTE.finditer(tag)}
        kind = next((_field_kind(attrs[key]) for key in ("name", "id", "autocomplete", "aria-label")
                     if key in attrs and _field_kind(attrs[key]) in values), None)
        if kind is None or attrs.get("value") not in values[kind]:
            return tag
        return _ATTRIBUTE.sub(lambda attr: (
            attr.group(1) + "=" + attr.group(2) + _MARKER + attr.group(2)
            if attr.group(1).casefold() == "value" else attr.group(0)
        ), tag)

    def clean_text(raw, depth=0):
        if depth <= 16 and raw.lstrip().startswith(("{", "[", '"')):
            try:
                parsed = json.loads(raw, parse_int=_JSONNumber, parse_float=_JSONNumber, parse_constant=_JSONNumber,
                                    object_pairs_hook=lambda pairs: _JSONObject(tuple(pairs)))
            except (ValueError, RecursionError):
                pass
            else:
                cleaned = structured(parsed, depth + 1)
                if cleaned != parsed:
                    return _dump_preserving_numbers(cleaned)
                # Valid JSON was examined by field/value structure. Never run
                # assignment regexes over its keys or syntax afterwards.
                return raw
        result = _INPUT.sub(html_input, raw)
        for pattern in patterns.values():
            result = pattern.sub(lambda m: m.group(0)[:m.start("code") - m.start()] + _MARKER, result)
        return result

    return clean_text(text)
