"""Login / checkout form control classifier for vault autofill.

Python port (~170 LOC) of Merit-Systems/OpenInstinct's
``lib/manager/server/kernel-login-autofill.ts`` (MIT). Classifies visible
input controls on a page into login-autofill tokens. The vault fill path
uses the classification to select the single best current-password control
(the identifier is agent-visible metadata and is typed by the agent
itself via normal input tools).

Scoring:
- exact autocomplete-token match ................ 100
- type=password (not new/confirm/create/repeat) .. 90
- type=email / type=tel .......................... 85
- label/name regex heuristics .................. 70-75
Hard exclusions: autocomplete ``new-password`` / ``one-time-code``, and
label/name text matching ``(new|confirm|create|repeat)\\s*password``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import itertools

LOGIN_AUTOFILL_TOKENS = ("username", "email", "tel", "current-password")

# Payment / address autocomplete tokens (WHATWG) the checkout fill targets. ``cc-exp`` (combined
# MM/YY) is derived at fill time from exp_month + exp_year. Field-name/label heuristics below back
# up sites that omit autocomplete attributes.
PAYMENT_AUTOFILL_TOKENS = ("cc-number", "cc-name", "cc-exp", "cc-exp-month", "cc-exp-year", "cc-csc")
ADDRESS_AUTOFILL_TOKENS = ("address-line1", "address-line2", "address-level2", "address-level1",
                           "postal-code", "country-name", "country")
_CHECKOUT_HEURISTICS = (
    (re.compile(r"\b(?:card\s*number|cardnumber|ccnumber|cc\s*num|pan)\b"), "cc-number"),
    (re.compile(r"\b(?:name\s*on\s*card|cardholder|cc\s*name|ccname)\b"), "cc-name"),
    (re.compile(r"\b(?:cvc|cvv|csc|security\s*code|card\s*code)\b"), "cc-csc"),
    (re.compile(r"\b(?:exp(?:iry|iration)?\s*month|exp\s*mm|ccmonth)\b"), "cc-exp-month"),
    (re.compile(r"\b(?:exp(?:iry|iration)?\s*year|exp\s*yy(?:yy)?|ccyear)\b"), "cc-exp-year"),
    (re.compile(r"\b(?:exp(?:iry|iration)?(?:\s*date)?|mm\s*yy|valid\s*thru)\b"), "cc-exp"),
    (re.compile(r"\b(?:address\s*(?:line\s*)?2|apt|suite|unit)\b"), "address-line2"),
    (re.compile(r"\b(?:address(?:\s*line\s*1)?|street)\b"), "address-line1"),
    (re.compile(r"\b(?:city|town|locality)\b"), "address-level2"),
    (re.compile(r"\b(?:state|province|region|county)\b"), "address-level1"),
    (re.compile(r"\b(?:zip|postal|postcode)\b"), "postal-code"),
    (re.compile(r"\b(?:country)\b"), "country-name"),
)

_EXCLUDED_AUTOCOMPLETE = {"new-password", "one-time-code"}

_RE_EXCLUDED_PASSWORD = re.compile(r"\b(?:new|confirm|create|repeat)\s*password\b")
_RE_EMAIL = re.compile(r"\b(?:e[\s-]?mail|email address)\b")
_RE_TEL = re.compile(r"\b(?:phone|telephone|mobile)\b")
_RE_USERNAME = re.compile(
    r"\b(?:user\s*name|username|login|account|member|membership|mileageplus)\b"
)


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


@dataclass(frozen=True)
class LoginControl:
    """Descriptor of a visible input control, as inspected in the page."""

    autocomplete: str
    form_index: Optional[int]
    index: int
    label: str
    name: str
    type: str
    max_length: Optional[int] = None
    group_index: Optional[int] = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LoginControl:
        form_index = raw.get("formIndex", raw.get("form_index"))
        max_length = raw.get("maxLength", raw.get("max_length"))
        return cls(
            autocomplete=str(raw.get("autocomplete") or ""),
            form_index=int(form_index) if form_index is not None else None,
            index=int(raw.get("index") or 0),
            label=str(raw.get("label") or ""),
            name=str(raw.get("name") or ""),
            type=str(raw.get("type") or ""),
            max_length=int(max_length) if max_length is not None else None,
            group_index=raw.get("groupIndex", raw.get("group_index")),
        )


@dataclass(frozen=True)
class ClassifiedLoginControl:
    control: LoginControl
    score: int
    token: str


def classify_login_control(control: LoginControl) -> Optional[ClassifiedLoginControl]:
    """Classify one control, or return None if it is not a login fill target."""
    autocomplete_tokens = [
        t for t in control.autocomplete.lower().split() if t
    ]
    if any(t in _EXCLUDED_AUTOCOMPLETE for t in autocomplete_tokens):
        return None

    for token in LOGIN_AUTOFILL_TOKENS:
        if token in autocomplete_tokens:
            return ClassifiedLoginControl(control, 100, token)

    searchable = _normalize_text(
        " ".join(part for part in (control.name, control.label) if part)
    )
    if _RE_EXCLUDED_PASSWORD.search(searchable):
        return None
    if control.type == "password":
        return ClassifiedLoginControl(control, 90, "current-password")
    if control.type == "email":
        return ClassifiedLoginControl(control, 85, "email")
    if control.type == "tel":
        return ClassifiedLoginControl(control, 85, "tel")
    if _RE_EMAIL.search(searchable):
        return ClassifiedLoginControl(control, 75, "email")
    if _RE_TEL.search(searchable):
        return ClassifiedLoginControl(control, 75, "tel")
    if _RE_USERNAME.search(searchable):
        return ClassifiedLoginControl(control, 70, "username")
    return None


_RE_OTP = re.compile(
    r"\b(?:one[\s-]?time|verification|security|auth(?:entication|enticator)?|2fa|two[\s-]?factor|mfa|totp|otp|"
    r"passcode|sms)\b.*\b(?:code|pin|token)\b|\b(?:otp|totp|2fa|mfa|verification\s*code|passcode)\b|"
    r"\benter\s+(?:(?:the|a)\s+)?[4-8]\s+digit\s+code\b"
)


_RE_POSITION = re.compile(r"\bdigit\s+(\d+)\s+of\s+(\d+)\b")
_RE_POSITION_HINT = re.compile(r"\bdigit\s+\w+\s+of\b")
_RE_NON_OTP_CODE = re.compile(r"\b(?:postal|postcode|zip|country|promo|promotion|coupon|discount|card|cvc|cvv|csc)\b")
_RE_ENTER_CODE = re.compile(r"\benter\s+code\b")


def _otp_position(control: LoginControl) -> Optional[tuple[int, int]]:
    text = _normalize_text(control.label)
    matches = list(_RE_POSITION.finditer(text))
    if (len(matches) != 1 or control.type not in ("text", "tel", "number", "password", "")
            or _RE_NON_OTP_CODE.search(_normalize_text(control.name + " " + control.label))
            or not (_RE_OTP.search(text) or _RE_ENTER_CODE.search(text))):
        return None
    return tuple(map(int, matches[0].groups()))


def _ordered_otp_boxes(controls: list[LoginControl]) -> list[LoginControl]:
    """Only a complete contextual positional widget, never a partial/mixed group."""
    boxes = sorted((c for c in controls if _RE_POSITION_HINT.search(_normalize_text(c.label))), key=lambda c: c.index)
    positions = [_otp_position(c) for c in boxes]
    n = len(boxes)
    if not 4 <= n <= 8 or positions != [(i, n) for i in range(1, n + 1)]:
        return []
    forms = {c.form_index for c in boxes}
    groups = {c.group_index for c in boxes}
    if len(forms) != 1 or len(groups) != 1 or (forms == {None} and groups == {None}):
        return []
    if not all(b.index - a.index == 1 for a, b in itertools.pairwise(boxes)):
        return []
    return boxes


def classify_otp_controls(controls: list[LoginControl]) -> list[ClassifiedLoginControl]:
    """The controls that take a second-factor code. ``autocomplete=one-time-code`` is authoritative
    for ordinary fields; positional labels additionally require a complete, scoped widget.
    Otherwise use name/label OTP heuristics on supported input types. Some sites split
    the code into one input per digit: contextual ``digit N of M`` labels work even without
    ``maxlength=1``. Inspection provides actual form/container identity, not class-name guesses."""
    out: list[ClassifiedLoginControl] = []
    positional_indices = {c.index for c in _ordered_otp_boxes(controls)}
    for c in controls:
        if _RE_POSITION_HINT.search(_normalize_text(c.label)):
            if c.index in positional_indices:
                out.append(ClassifiedLoginControl(c, 70, "one-time-code"))
            continue
        tokens = c.autocomplete.lower().split()
        if "one-time-code" in tokens:
            out.append(ClassifiedLoginControl(c, 100, "one-time-code"))
            continue
        if c.type not in ("text", "tel", "number", "password", ""):
            continue
        if _RE_OTP.search(_normalize_text(" ".join(p for p in (c.name, c.label) if p))):
            out.append(ClassifiedLoginControl(c, 70, "one-time-code"))
    return out


def select_password_fill(
    classified: list[ClassifiedLoginControl],
    password: str,
) -> list[dict[str, Any]]:
    """Select the single best current-password control to fill.

    The vault fill path is password-only: the identifier is agent-visible
    metadata and is typed by the agent via normal input tools. This picks
    the highest-scoring ``current-password`` control (ties broken by DOM
    order) and returns ``[{"index": int, "token": "current-password",
    "value": password}]`` or ``[]`` when no password field exists.
    """
    passwords = [c for c in classified if c.token == "current-password"]
    if not passwords or not password:
        return []
    best_password = min(passwords, key=lambda c: (-c.score, c.control.index))
    return [
        {
            "index": best_password.control.index,
            "token": "current-password",
            "value": password,
        }
    ]


def classify_checkout_control(control: LoginControl) -> Optional[ClassifiedLoginControl]:
    """Classify one control as a payment/address fill target (autocomplete token exact match 100,
    label/name heuristic 70), or None. Password/email inputs are never checkout targets."""
    tokens = [t for t in control.autocomplete.lower().split() if t]
    for token in PAYMENT_AUTOFILL_TOKENS + ADDRESS_AUTOFILL_TOKENS:
        if token in tokens:
            return ClassifiedLoginControl(control, 100, "country-name" if token == "country" else token)
    if control.type in ("password", "email"):
        return None
    searchable = _normalize_text(" ".join(part for part in (control.name, control.label) if part))
    for pattern, token in _CHECKOUT_HEURISTICS:
        if pattern.search(searchable):
            return ClassifiedLoginControl(control, 70, token)
    return None


def select_checkout_fills(classified: list[ClassifiedLoginControl], secret: dict[str, str],
                          field_tokens: dict[str, str]) -> list[dict[str, Any]]:
    """Map a payment/address secret payload onto the best control per autocomplete token.

    ``field_tokens`` is ``PAYMENT_FIELDS`` / ``ADDRESS_FIELDS`` (agent/vault_store.py). A combined
    ``cc-exp`` control gets ``MM/YY`` from exp_month + exp_year and then suppresses the separate
    month/year fills. Returns ``[{"index", "token", "value"}]``: one control per token, highest
    score then DOM order.
    """
    values: dict[str, str] = {tok: secret[f] for f, tok in field_tokens.items() if secret.get(f)}
    if "cc-exp-month" in values and "cc-exp-year" in values:
        values["cc-exp"] = f"{values['cc-exp-month'].zfill(2)}/{values['cc-exp-year'][-2:]}"
    fills: list[dict[str, Any]] = []
    for token, value in values.items():
        candidates = sorted((c for c in classified if c.token == token), key=lambda c: (-c.score, c.control.index))
        if candidates:
            fills.append({"index": candidates[0].control.index, "token": token, "value": value})
    if any(f["token"] == "cc-exp" for f in fills):
        fills = [f for f in fills if f["token"] not in ("cc-exp-month", "cc-exp-year")]
    return fills


# JS expression evaluated in the page to inspect candidate input controls.
# Ported from OpenInstinct's nativeLoginControlInspectionExpression.
# Inspection stamps every input with ``<nonce>:<index>`` under a per-inspection attribute; the fill
# script resolves targets by the stamp of ITS OWN inspection instead of re-querying by position, so
# neither a DOM reflow nor a second inspection in between can redirect the password into another field.
INSPECTION_STAMP_ATTR = "data-hermes-vault-slot"


def build_otp_fills(otp_controls: list[ClassifiedLoginControl], code: str) -> list[dict[str, Any]]:
    """One fill per box. Default: the single best-scoring code field takes the whole code.

    Split only a complete ordered positional widget, or the classic adjacent ``maxlength=1``
    boxes, with actual common form/group identity. Invalid positional widgets/length mismatches
    fail closed; they must never receive a whole-code fallback. Ordinary non-positional inputs
    retain the single best-field fallback."""
    if not otp_controls or not code:
        return []
    controls = [c.control for c in otp_controls]
    if any(_RE_POSITION_HINT.search(_normalize_text(c.label)) for c in controls):
        positional = _ordered_otp_boxes(controls)
        if not positional or len(positional) != len(code):
            return []
        return [{"index": b.index, "token": "one-time-code", "value": ch} for b, ch in zip(positional, code)]
    best = max(otp_controls, key=lambda c: c.score)
    boxes = sorted((c for c in otp_controls if c.control.max_length == 1), key=lambda c: c.control.index)
    if (len(boxes) == len(code)
            and len({b.control.form_index for b in boxes}) == 1
            and len({b.control.group_index for b in boxes}) == 1
            and (boxes[0].control.form_index is not None or boxes[0].control.group_index is not None)
            and all(b.control.index - a.control.index == 1 for a, b in itertools.pairwise(boxes))):
        return [{"index": b.control.index, "token": "one-time-code", "value": ch} for b, ch in zip(boxes, code)]
    return [{"index": best.control.index, "token": "one-time-code", "value": code}]


def build_otp_probe_js() -> str:
    """Read-only tab predicate using the same visibility/label rules as inspection."""
    inspection = _LOGIN_CONTROL_INSPECTION_JS_TEMPLATE.replace("__NONCE__", "null")
    template = """(() => {
      const controls = JSON.parse(__INSPECTION__);
      const pattern = new RegExp(__PATTERN__);
      const positionPattern = new RegExp(__POSITION__, "g");
      const positionHint = new RegExp(__POSITION_HINT__);
      const excludedPattern = new RegExp(__EXCLUDED__);
      const enterPattern = new RegExp(__ENTER__);
      const norm = t => t.normalize("NFKD").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
      const positional = controls.filter(c => positionHint.test(norm(c.label))).sort((a,b) => a.index-b.index);
      const n = positional.length;
      const valid = n >= 4 && n <= 8 && positional.every((c,i) => {
        const text = norm(c.label);
        const matches = [...text.matchAll(positionPattern)];
        return ["text", "tel", "number", "password", ""].includes(c.type)
          && matches.length === 1 && Number(matches[0][1]) === i+1 && Number(matches[0][2]) === n
          && !excludedPattern.test(norm(c.name + " " + c.label)) && (pattern.test(text) || enterPattern.test(text))
          && c.formIndex === positional[0].formIndex && c.groupIndex === positional[0].groupIndex
          && (c.formIndex != null || c.groupIndex != null)
          && (i === 0 || c.index === positional[i-1].index+1);
      });
      return controls.some(c => {
        if (positionHint.test(norm(c.label))) return valid;
        if (c.autocomplete.toLowerCase().split(/\\s+/).includes("one-time-code")) return true;
        if (!["text", "tel", "number", "password", ""].includes(c.type)) return false;
        const text = [c.name, c.label].join(" ").normalize("NFKD").toLowerCase()
          .replace(/[^a-z0-9]+/g, " ").trim();
        return pattern.test(text);
      });
    })()"""
    replacements = {
        "__INSPECTION__": inspection,
        "__PATTERN__": json.dumps(_RE_OTP.pattern),
        "__POSITION__": json.dumps(_RE_POSITION.pattern),
        "__POSITION_HINT__": json.dumps(_RE_POSITION_HINT.pattern),
        "__EXCLUDED__": json.dumps(_RE_NON_OTP_CODE.pattern),
        "__ENTER__": json.dumps(_RE_ENTER_CODE.pattern),
    }
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    return template


def build_inspection_js(nonce: str) -> str:
    return _LOGIN_CONTROL_INSPECTION_JS_TEMPLATE.replace("__NONCE__", json.dumps(nonce))


_LOGIN_CONTROL_INSPECTION_JS_TEMPLATE = """(() => {
  const nonce = __NONCE__;
  const elements = Array.from(document.querySelectorAll("input, select"));
  const forms = Array.from(document.forms);
  const groups = [];
  if (nonce !== null) elements.forEach((element, index) => element.setAttribute("data-hermes-vault-slot", nonce + ":" + index));
  const out = elements.flatMap((element, index) => {
    if (element.disabled || element.readOnly) return [];
    if (["hidden", "submit", "button", "reset", "file", "image", "checkbox", "radio"].includes(element.type)) return [];
    const style = getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || element.getClientRects().length === 0) return [];
    const labels = element.labels ? Array.from(element.labels, (l) => l.textContent || "") : [];
    const ariaText = (element.getAttribute("aria-labelledby") || "")
      .split(/\\s+/).filter(Boolean)
      .map((id) => { const n = document.getElementById(id); return n ? (n.textContent || "") : ""; })
      .join(" ");
    const resolvedFormIndex = element.form ? forms.indexOf(element.form) : -1;
    let group = element.parentElement;
    while (group && !["FORM", "BODY", "HTML"].includes(group.tagName)) {
      if (group.querySelectorAll && group.querySelectorAll("input, select").length > 1) break;
      group = group.parentElement;
    }
    if (group && ["FORM", "BODY", "HTML"].includes(group.tagName)) group = null;
    if (group && !groups.includes(group)) groups.push(group);
    return [{
      autocomplete: element.autocomplete || "",
      formIndex: resolvedFormIndex >= 0 ? resolvedFormIndex : null,
      groupIndex: group ? groups.indexOf(group) : null,
      index,
      maxLength: element.maxLength > 0 ? element.maxLength : null,
      label: [
        ...labels,
        element.getAttribute("aria-label") || "",
        ariaText,
        element.getAttribute("placeholder") || "",
        element.getAttribute("title") || "",
      ].join(" "),
      name: [element.name, element.id].join(" "),
      type: element.tagName === "SELECT" ? "select" : (element.type || ""),
    }];
  });
  return JSON.stringify(out);
})()"""


def build_fill_js(fills: list[dict[str, Any]], expected_origin: str, nonce: str = "") -> str:
    """Build a JS expression that fills the selected controls and reports only a count. The
    returned expression never echoes the values back.

    ``expected_origin`` is asserted against ``window.location.origin`` synchronously inside the SAME
    evaluated script, immediately before any write. If the page navigated between inspection and fill
    (TOCTOU), the script writes nothing and returns ``{"refused": "origin_changed", "found": <actual>}``:
    proof scope equals mutation scope (#88706). Targets resolve by the ``<nonce>:<index>`` stamp of
    THIS inspection; a ``current-password`` fill additionally requires ``type=password``; ``<select>``
    controls (country, state, expiry month) match an option by value or visible text. No marker is
    left on filled controls so later model-driven DOM reads cannot address them deterministically.
    """
    payload = json.dumps(
        [{"index": f["index"], "token": f.get("token", "current-password"), "value": f["value"]} for f in fills]
    )
    return (_FILL_JS_TEMPLATE.replace("__EXPECTED_ORIGIN__", json.dumps(expected_origin))
            .replace("__FILLS__", payload).replace("__NONCE__", json.dumps(nonce)))


_FILL_JS_TEMPLATE = """(() => {
  const expectedOrigin = __EXPECTED_ORIGIN__;
  if (window.location.origin !== expectedOrigin) {
    return JSON.stringify({ refused: "origin_changed", found: window.location.origin });
  }
  const fills = __FILLS__;
  const nonce = __NONCE__;
  let filled = 0;
  const norm = (t) => String(t || "").trim().toLowerCase();
  for (const f of fills) {
    const el = document.querySelector('[data-hermes-vault-slot="' + nonce + ':' + f.index + '"]');
    if (!el || (f.token === "current-password" && el.type !== "password")) continue;
    try {
      if (el.tagName === "SELECT") {
        const want = norm(f.value);
        const opt = Array.from(el.options).find((o) => [o.value, o.textContent].some((t) => norm(t) === want || norm(t) === want.replace(/^0/, "")));
        if (opt) { el.value = opt.value; el.dispatchEvent(new Event("change", { bubbles: true })); filled += 1; }
        continue;
      }
      el.focus();
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
      // one-time-code split into single-character boxes: f.value is the slice for THIS box (see build_otp_fills)
      if (setter && setter.set) { setter.set.call(el, f.value); } else { el.value = f.value; }
      el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText" }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      if (el.value.length > 0) filled += 1;
    } catch (e) { /* skip */ }
  }
  document.querySelectorAll("[data-hermes-vault-slot]").forEach((n) => n.removeAttribute("data-hermes-vault-slot"));
  return JSON.stringify({ filled });
})()"""
