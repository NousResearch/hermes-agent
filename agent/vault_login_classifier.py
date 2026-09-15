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
Hard exclusions: autocomplete ``new-password`` (unless explicitly enabled for
a generated credential) / ``one-time-code``, and
label/name text matching ``(new|confirm|create|repeat)\\s*password``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

_EXCLUDED_AUTOCOMPLETE = {"one-time-code"}

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

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LoginControl":
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
        )


@dataclass(frozen=True)
class ClassifiedLoginControl:
    control: LoginControl
    score: int
    token: str


def classify_login_control(
    control: LoginControl, *, allow_new_password: bool = False
) -> Optional[ClassifiedLoginControl]:
    """Classify one control, or return None if it is not a login fill target."""
    autocomplete_tokens = [
        t for t in control.autocomplete.lower().split() if t
    ]
    if any(t in _EXCLUDED_AUTOCOMPLETE for t in autocomplete_tokens):
        return None
    if "new-password" in autocomplete_tokens:
        if allow_new_password and control.type == "password":
            return ClassifiedLoginControl(control, 100, "new-password")
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
    r"passcode|sms)\b.*\b(?:code|pin|token)\b|\b(?:otp|totp|2fa|mfa|verification\s*code|passcode)\b"
)


def classify_otp_controls(controls: List[LoginControl]) -> List[ClassifiedLoginControl]:
    """The controls that take a second-factor code. ``autocomplete=one-time-code`` is authoritative;
    otherwise a text/tel/number input whose name/label says code/OTP/2FA/verification. Some sites split
    the code into one input per digit (``maxlength=1`` boxes): they are returned in DOM order and the
    fill spreads the code across them."""
    out: List[ClassifiedLoginControl] = []
    for c in controls:
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
    classified: List[ClassifiedLoginControl],
    password: str,
    *,
    allow_new_password: bool = False,
) -> List[Dict[str, Any]]:
    """Select the single best allowed password control to fill.

    The vault fill path is password-only: the identifier is agent-visible
    metadata and is typed by the agent via normal input tools. This picks
    the highest-scoring ``current-password`` control, or an exact
    ``new-password`` control when explicitly enabled (ties broken by DOM
    order), and returns ``[{"index": int, "token": str,
    "value": password}]`` or ``[]`` when no password field exists.
    """
    allowed_tokens = {"current-password"}
    if allow_new_password:
        allowed_tokens.add("new-password")
    passwords = [c for c in classified if c.token in allowed_tokens]
    if not passwords or not password:
        return []
    if allow_new_password:
        explicit_new_passwords = [
            candidate for candidate in passwords if candidate.token == "new-password"
        ]
        if explicit_new_passwords:
            passwords = explicit_new_passwords
    ranked = sorted(passwords, key=lambda c: (-c.score, c.control.index))
    best_password = ranked[0]
    selected = [best_password]
    if best_password.token == "new-password":
        # Signup forms commonly expose password + confirmation with the exact
        # same token. Fill at most those two controls, and never cross forms.
        selected = [
            candidate
            for candidate in ranked
            if candidate.token == "new-password"
            and candidate.control.form_index == best_password.control.form_index
        ][:2]
    return [
        {
            "index": candidate.control.index,
            "token": candidate.token,
            "autocomplete": candidate.control.autocomplete.lower().strip(),
            "fingerprint": {
                "autocomplete": candidate.control.autocomplete.lower().strip(),
                "form_index": candidate.control.form_index,
                "label": candidate.control.label,
                "max_length": candidate.control.max_length,
                "name": candidate.control.name,
                "type": candidate.control.type,
            },
            "value": password,
        }
        for candidate in selected
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


def select_checkout_fills(classified: List[ClassifiedLoginControl], secret: Dict[str, str],
                          field_tokens: Dict[str, str]) -> List[Dict[str, Any]]:
    """Map a payment/address secret payload onto the best control per autocomplete token.

    ``field_tokens`` is ``PAYMENT_FIELDS`` / ``ADDRESS_FIELDS`` (agent/vault_store.py). A combined
    ``cc-exp`` control gets ``MM/YY`` from exp_month + exp_year and then suppresses the separate
    month/year fills. Returns ``[{"index", "token", "value"}]``: one control per token, highest
    score then DOM order.
    """
    values: Dict[str, str] = {tok: secret[f] for f, tok in field_tokens.items() if secret.get(f)}
    if "cc-exp-month" in values and "cc-exp-year" in values:
        values["cc-exp"] = f"{values['cc-exp-month'].zfill(2)}/{values['cc-exp-year'][-2:]}"
    fills: List[Dict[str, Any]] = []
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


def build_otp_fills(otp_controls: List[ClassifiedLoginControl], code: str) -> List[Dict[str, Any]]:
    """One fill per box. Default: the single best-scoring code field takes the whole code.

    Per-digit entry only when the page unmistakably uses it: exactly len(code) OTP controls that are all
    ``maxlength=1``, all in the same form, and adjacent in DOM order (the classic N-box widget). Anything
    looser (several code-like inputs scattered over a page) gets ONE field, never a digit sprayed across
    unrelated inputs."""
    best = max(otp_controls, key=lambda c: c.score)
    boxes = sorted((c for c in otp_controls if c.control.max_length == 1), key=lambda c: c.control.index)
    if (len(boxes) == len(code)
            and len({b.control.form_index for b in boxes}) == 1
            and all(b.control.index - a.control.index == 1 for a, b in zip(boxes, boxes[1:]))):
        return [{"index": b.control.index, "token": "one-time-code", "value": ch} for b, ch in zip(boxes, code)]
    return [{"index": best.control.index, "token": "one-time-code", "value": code}]


def build_inspection_js(nonce: str) -> str:
    return _LOGIN_CONTROL_INSPECTION_JS_TEMPLATE.replace("__NONCE__", json.dumps(nonce))


_LOGIN_CONTROL_INSPECTION_JS_TEMPLATE = """(() => {
  const nonce = __NONCE__;
  const elements = Array.from(document.querySelectorAll("input, select"));
  const forms = Array.from(document.forms);
  const stateKey = "__hermesVaultInspection:" + nonce;
  Object.defineProperty(globalThis, stateKey, {
    value: { elements, originalForms: elements.map((element) => element.form) },
    configurable: true,
  });
  elements.forEach((element, index) => element.setAttribute("data-hermes-vault-slot", nonce + ":" + index));
  const out = elements.flatMap((element, index) => {
    if (element.disabled || element.readOnly || element.matches(":disabled")) return [];
    if (["hidden", "submit", "button", "reset", "file", "image", "checkbox", "radio"].includes(element.type)) return [];
    const style = getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || Number.parseFloat(style.opacity || "1") <= 0.01) return [];
    if (typeof element.checkVisibility === "function" && !element.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) return [];
    const rect = element.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0 || rect.bottom <= 0 || rect.right <= 0 || rect.top >= window.innerHeight || rect.left >= window.innerWidth) return [];
    if (element.getClientRects().length === 0) return [];
    const labels = element.labels ? Array.from(element.labels, (l) => l.textContent || "") : [];
    const ariaText = (element.getAttribute("aria-labelledby") || "")
      .split(/\\s+/).filter(Boolean)
      .map((id) => { const n = document.getElementById(id); return n ? (n.textContent || "") : ""; })
      .join(" ");
    const resolvedFormIndex = element.form ? forms.indexOf(element.form) : -1;
    return [{
      autocomplete: element.autocomplete || "",
      formIndex: resolvedFormIndex >= 0 ? resolvedFormIndex : null,
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


def build_fill_js(fills: List[Dict[str, Any]], expected_origin: str, nonce: str = "") -> str:
    """Build a JS expression that fills the selected controls and reports only a count. The
    returned expression never echoes the values back.

    ``expected_origin`` is asserted against ``window.location.origin`` synchronously inside the SAME
    evaluated script, immediately before any write. If the page navigated between inspection and fill
    (TOCTOU), the script writes nothing and returns ``{"refused": "origin_changed", "found": <actual>}``:
    proof scope equals mutation scope (#88706). Targets resolve by the ``<nonce>:<index>`` stamp of
    THIS inspection and exact element/form object references retained in the supervisor's isolated
    vault world; a password fill additionally requires ``type=password``; ``<select>``
    controls (country, state, expiry month) match an option by value or visible text. No marker is
    left on filled controls so later model-driven DOM reads cannot address them deterministically.
    """
    payload = json.dumps(
        [
            {
                "index": f["index"],
                "token": f.get("token", "current-password"),
                "value": f["value"],
                **({"autocomplete": f["autocomplete"]} if "autocomplete" in f else {}),
                **({"fingerprint": f["fingerprint"]} if "fingerprint" in f else {}),
            }
            for f in fills
        ]
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
  const stateKey = "__hermesVaultInspection:" + nonce;
  const state = globalThis[stateKey];
  delete globalThis[stateKey];
  const forms = Array.from(document.forms);
  let filled = 0;
  const norm = (t) => String(t || "").trim().toLowerCase();
  for (const f of fills) {
    const stamped = document.querySelector('[data-hermes-vault-slot="' + nonce + ':' + f.index + '"]');
    const el = state && state.elements ? state.elements[f.index] : null;
    if (!el || stamped !== el || !state.originalForms || state.originalForms[f.index] !== el.form) continue;
    if ((f.token === "current-password" || f.token === "new-password") && el.type !== "password") continue;
    try {
      if (f.token === "current-password" || f.token === "new-password") {
        if (!f.fingerprint || el.disabled || el.readOnly || el.matches(":disabled")) continue;
        if (["hidden", "submit", "button", "reset", "file", "image", "checkbox", "radio"].includes(el.type)) continue;
        const style = getComputedStyle(el);
        if (style.display === "none" || style.visibility === "hidden" || Number.parseFloat(style.opacity || "1") <= 0.01) continue;
        if (typeof el.checkVisibility === "function" && !el.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true })) continue;
        const rect = el.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0 || rect.bottom <= 0 || rect.right <= 0 || rect.top >= window.innerHeight || rect.left >= window.innerWidth) continue;
        if (el.getClientRects().length === 0) continue;
        const liveAutocomplete = norm(el.autocomplete);
        const liveTokens = liveAutocomplete.split(/\\s+/).filter(Boolean);
        const labels = el.labels ? Array.from(el.labels, (label) => label.textContent || "") : [];
        const ariaText = (el.getAttribute("aria-labelledby") || "")
          .split(/\\s+/).filter(Boolean)
          .map((id) => { const node = document.getElementById(id); return node ? (node.textContent || "") : ""; })
          .join(" ");
        const liveFormIndexRaw = el.form ? forms.indexOf(el.form) : -1;
        const liveFormIndex = liveFormIndexRaw >= 0 ? liveFormIndexRaw : null;
        const liveLabel = [
          ...labels,
          el.getAttribute("aria-label") || "",
          ariaText,
          el.getAttribute("placeholder") || "",
          el.getAttribute("title") || "",
        ].join(" ");
        const liveName = [el.name, el.id].join(" ");
        const liveMaxLength = el.maxLength > 0 ? el.maxLength : null;
        const liveType = el.tagName === "SELECT" ? "select" : (el.type || "");
        if (liveAutocomplete !== norm(f.fingerprint.autocomplete)) continue;
        if (liveFormIndex !== f.fingerprint.form_index) continue;
        if (liveLabel !== f.fingerprint.label) continue;
        if (liveName !== f.fingerprint.name) continue;
        if (liveMaxLength !== f.fingerprint.max_length) continue;
        if (liveType !== f.fingerprint.type) continue;
        if (f.token === "new-password" && !liveTokens.includes("new-password")) continue;
        if (f.token === "current-password" && (liveTokens.includes("new-password") || liveTokens.includes("one-time-code"))) continue;
      }
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
