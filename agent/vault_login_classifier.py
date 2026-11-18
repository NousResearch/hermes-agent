"""Login / checkout form control classifier for vault autofill.

Python port (~170 LOC) of Merit-Systems/OpenInstinct's
``lib/manager/server/kernel-login-autofill.ts`` (MIT). Classifies visible
input controls on a page into login-autofill tokens. The vault fill path
uses the classification to select the single best current-password control
and, when asked, the login's identifier control.

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
    r"passcode|sms)\b.*\b(?:code|pin|token)\b|\b(?:otp|totp|2fa|mfa|verification\s*code|passcode)\b"
)


def classify_otp_controls(controls: list[LoginControl]) -> list[ClassifiedLoginControl]:
    """The controls that take a second-factor code. ``autocomplete=one-time-code`` is authoritative;
    otherwise a text/tel/number input whose name/label says code/OTP/2FA/verification. Some sites split
    the code into one input per digit (``maxlength=1`` boxes): they are returned in DOM order and the
    fill spreads the code across them."""
    out: list[ClassifiedLoginControl] = []
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
    classified: list[ClassifiedLoginControl],
    password: str,
) -> list[dict[str, Any]]:
    """Select the single best current-password control to fill.

    This picks the highest-scoring ``current-password`` control (ties broken
    by DOM order) and returns ``[{"index": int, "token": "current-password",
    "value": password}]`` or ``[]`` when no password field exists. The
    identifier is a separate fill: see the tool's identifier selection.
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

    Per-digit entry only when the page unmistakably uses it: exactly len(code) OTP controls that are all
    ``maxlength=1``, all in the same form, and adjacent in DOM order (the classic N-box widget). Anything
    looser (several code-like inputs scattered over a page) gets ONE field, never a digit sprayed across
    unrelated inputs."""
    best = max(otp_controls, key=lambda c: c.score)
    boxes = sorted((c for c in otp_controls if c.control.max_length == 1), key=lambda c: c.control.index)
    if (len(boxes) == len(code)
            and len({b.control.form_index for b in boxes}) == 1
            and all(b.control.index - a.control.index == 1 for a, b in itertools.pairwise(boxes))):
        return [{"index": b.control.index, "token": "one-time-code", "value": ch} for b, ch in zip(boxes, code)]
    return [{"index": best.control.index, "token": "one-time-code", "value": code}]


def build_inspection_js(nonce: str) -> str:
    return _LOGIN_CONTROL_INSPECTION_JS_TEMPLATE.replace("__NONCE__", json.dumps(nonce))


_LOGIN_CONTROL_INSPECTION_JS_TEMPLATE = """(() => {
  const nonce = __NONCE__;
  // Shadow-piercing collect: many logins (Home Assistant, custom-element SPAs) render their
  // form inside a shadow root, where document.querySelectorAll never sees it. Walk every
  // element (shadow hosts can be anything, not just inputs) and recurse into open shadow roots.
  // Forms are collected in the same walk: document.forms misses a <form> inside a shadow root, which
  // would leave its controls ungrouped (formIndex null) and merge split OTP boxes across forms.
  const elements = [];
  const forms = [];
  (function walk(root) {
    for (const e of root.querySelectorAll("*")) {
      if (e.tagName === "INPUT" || e.tagName === "SELECT") elements.push(e);
      else if (e.tagName === "FORM") forms.push(e);
      if (e.shadowRoot) walk(e.shadowRoot);
    }
  })(document);
  elements.forEach((element, index) => element.setAttribute("data-hermes-vault-slot", nonce + ":" + index));
  const out = elements.flatMap((element, index) => {
    if (element.disabled || element.readOnly) return [];
    if (["hidden", "submit", "button", "reset", "file", "image", "checkbox", "radio"].includes(element.type)) return [];
    const style = getComputedStyle(element);
    if (style.display === "none" || style.visibility === "hidden" || element.getClientRects().length === 0) return [];
    const labels = element.labels ? Array.from(element.labels, (l) => l.textContent || "") : [];
    const ariaText = (element.getAttribute("aria-labelledby") || "")
      .split(/\\s+/).filter(Boolean)
      .map((id) => { const r = element.getRootNode(); const n = (r.getElementById ? r.getElementById(id) : null) || document.getElementById(id); return n ? (n.textContent || "") : ""; })
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
    return (_guarded(_FILL_JS_TEMPLATE, expected_origin).replace("__OPTION_MATCH__", _OPTION_MATCH_JS)
            .replace("__FILLS__", payload).replace("__NONCE__", json.dumps(nonce)))


# The opening of every script that carries a vault value or decides where one is typed. Strict mode
# first: a sloppy-mode function is reachable from any page function it calls through ``caller``, and
# its source text, literal values included, with it. Then the origin, before any value is referenced:
# a page that navigated (a login that submits on input, a retargeted tab) gets a refusal and nothing
# else. ``window.location`` is unforgeable, so the page cannot answer this check for another origin.
_ORIGIN_GUARD_JS = """  "use strict";
  if (window.location.origin !== __EXPECTED_ORIGIN__) {
    return JSON.stringify({ refused: "origin_changed", found: window.location.origin });
  }"""


def _guarded(template: str, expected_origin: str) -> str:
    return template.replace("__ORIGIN_GUARD__", _ORIGIN_GUARD_JS.replace(
        "__EXPECTED_ORIGIN__", json.dumps(expected_origin)))


# How a requested value names a <select> option: by value or visible text, trimmed and
# case-insensitive, and an expiry month "07" also names option "7". The fill script picks the first
# option this matches; the read-back accepts the selected option only if it matches too, so the two
# can never disagree about what "the right option" is.
_OPTION_MATCH_JS = """  const norm = (t) => String(t || "").trim().toLowerCase();
  const optionMatches = (o, value) => {
    const want = norm(value);
    return [o.value, o.textContent].some((t) => norm(t) === want || norm(t) === want.replace(/^0/, ""));
  };"""


_FILL_JS_TEMPLATE = """(() => {
__ORIGIN_GUARD__
  const fills = __FILLS__;
  const nonce = __NONCE__;
  // Stamped controls may live inside a shadow root (see the inspection script's walk): collect the
  // document and every open shadow root once, then look each stamp up in those roots.
  const roots = [];
  (function walk(root) {
    roots.push(root);
    for (const e of root.querySelectorAll("*")) if (e.shadowRoot) walk(e.shadowRoot);
  })(document);
  const find = (selector) => roots.flatMap((r) => Array.from(r.querySelectorAll(selector)));
  let filled = 0;
__OPTION_MATCH__
  for (const f of fills) {
    const el = find('[data-hermes-vault-slot="' + nonce + ':' + f.index + '"]')[0];
    if (!el || (f.token === "current-password" && el.type !== "password")) continue;
    try {
      if (el.tagName === "SELECT") {
        const opt = Array.from(el.options).find((o) => optionMatches(o, f.value));
        if (opt) { el.value = opt.value; el.dispatchEvent(new Event("change", { bubbles: true, composed: true })); filled += 1; }
        continue;
      }
      el.focus();
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value");
      // one-time-code split into single-character boxes: f.value is the slice for THIS box (see build_otp_fills)
      if (setter && setter.set) { setter.set.call(el, f.value); } else { el.value = f.value; }
      // composed: a component listening on its shadow host must see the value change too.
      // data: is what a framework-controlled input reads. React/Vue ignore an InputEvent without it
      // and leave their own state empty, which reads as a filled DOM field the page still rejects.
      el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: f.value, inputType: "insertText" }));
      el.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
      if (el.value.length > 0) filled += 1;
    } catch (e) { /* skip */ }
  }
  // Stamps stay on the controls: this script cannot tell whether a controlled input kept the value,
  // because the framework re-renders after it returns. build_fill_verify_js reads back and unstamps.
  return JSON.stringify({ filled });
})()"""


def build_fill_verify_js(fills: list[dict[str, Any]], nonce: str = "", *, expected_origin: str,
                         unstamp: bool = True) -> str:
    """Build a JS expression that reads the filled controls back and removes the inspection stamps.

    It carries every expected value (the password too), so it opens with the fill script's origin
    guard: on a page that left ``expected_origin`` it returns ``{"refused": "origin_changed"}`` before
    any value is referenced. On the bound origin the page already holds the value it was typed, so
    comparing in the clear there exposes nothing new.

    The fill script's count is a claim about the write; this is the claim about the page. Called one
    turn of the event loop later, it catches a framework that reverted the value, a field that was
    re-rendered (stamp gone) and a partial write. Returns counts only — no value leaves the page.

    ``unstamp=False`` keeps the inspection stamps: the trusted-typing path verifies ONE field
    between two ``Input.insertText`` calls and must still find the next field by its stamp.

    A ``<select>`` holds the value when its selected option matches it the way the fill script
    picks options. A text control holds it when it reads back exactly — except that a value made only
    of digits and separators also holds it when a display mask reformatted it without changing what
    it says: a card number or postal code with the same digits regrouped (``4111 1111 …``), a phone
    number with the same digits and the same leading ``+`` (``(555) 123-4567``), an expiry with the
    same month and year (``7/29`` for ``07/29``; ``1/203`` is another date). A dropped or added digit,
    any letter, and every other token (passwords, usernames, CVC) stay exact.
    """
    payload = json.dumps(
        [{"index": f["index"], "token": f.get("token", "current-password"), "value": f["value"]} for f in fills]
    )
    return (_guarded(_VERIFY_JS_TEMPLATE, expected_origin).replace("__OPTION_MATCH__", _OPTION_MATCH_JS)
            .replace("__FILLS__", payload).replace("__NONCE__", json.dumps(nonce))
            .replace("__UNSTAMP__", "true" if unstamp else "false"))


def build_focus_js(fills: list[dict[str, Any]], nonce: str = "", *, expected_origin: str,
                   refocus: bool = True) -> str:
    """Build a JS expression that focuses ONE stamped field and asserts it really holds focus.

    ``Input.insertText`` types into whatever element currently has focus, so the caller MUST
    know which element that is before typing a secret into it. This script focuses the element
    stamped ``<nonce>:<index>`` by the inspection and only reports success when
    ``document.activeElement`` IS that element — a page that moves focus (a re-render, a modal,
    an autofocus elsewhere) is refused instead of receiving the secret.

    Focus is read through open shadow roots (a focused input inside a component reports its HOST as
    ``document.activeElement``), the same roots the inspection and fill scripts reach.
    ``refocus=False`` only asserts: between the keys of one field the page must not have moved focus,
    and pulling it back would hide that it did. The fill script's origin guard runs first: the next
    keystroke goes wherever focus is, so focus is only vouched for on ``expected_origin``.

    Takes the same ``fills`` shape as :func:`build_fill_js` but is meant for exactly one field.
    Returns JSON: ``{"focused": <index>}`` or
    ``{"refused": "focus_lost", "tag": <tag>, "found": <active tag or null>}`` /
    ``{"refused": "not_password"}`` / ``{"refused": "not_found"}`` / ``{"refused": "origin_changed"}``.
    """
    payload = json.dumps(
        [{"index": f["index"], "token": f.get("token", "current-password")} for f in fills]
    )
    return (_guarded(_FOCUS_JS_TEMPLATE, expected_origin).replace("__FILLS__", payload).replace("__NONCE__", json.dumps(nonce))
            .replace("__REFOCUS__", "true" if refocus else "false"))


_FOCUS_JS_TEMPLATE = """(() => {
__ORIGIN_GUARD__
  const fills = __FILLS__;
  const nonce = __NONCE__;
  const f = fills[0];
  if (!f) return JSON.stringify({ refused: "not_found" });
  const roots = [];
  (function walk(root) {
    roots.push(root);
    for (const e of root.querySelectorAll("*")) if (e.shadowRoot) walk(e.shadowRoot);
  })(document);
  const find = (selector) => roots.flatMap((r) => Array.from(r.querySelectorAll(selector)));
  const el = find('[data-hermes-vault-slot="' + nonce + ':' + f.index + '"]')[0];
  if (!el) return JSON.stringify({ refused: "not_found" });
  // Same guard as the fill script: a password fill never lands in a text box.
  if (f.token === "current-password" && el.type !== "password") {
    return JSON.stringify({ refused: "not_password" });
  }
  if (__REFOCUS__) { try { el.focus(); } catch (e) { /* focus can throw on a detached node */ } }
  let active = document.activeElement;
  while (active && active.shadowRoot && active.shadowRoot.activeElement) active = active.shadowRoot.activeElement;
  if (active === el) return JSON.stringify({ focused: f.index });
  return JSON.stringify({ refused: "focus_lost", tag: el.tagName,
                          found: active ? active.tagName : null });
})()"""


def build_origin_check_js(expected_origin: str) -> str:
    """Build a JS expression that asserts the page's origin inside the page itself.

    The trusted-typing path writes through CDP, outside any page script, so it needs the same
    in-page origin assertion the JS fill path has: if the page navigated between the origin
    pre-check and the keystrokes, nothing is typed.
    """
    return ("(() => { if (window.location.origin !== " + json.dumps(expected_origin) + ") "
            "return JSON.stringify({ refused: 'origin_changed', found: window.location.origin }); "
            "return JSON.stringify({ ok: true }); })()")


def build_unstamp_js(nonce: str = "") -> str:
    """Build a JS expression that removes every inspection stamp from the page.

    Used when a fill aborts half-way: the stamps address controls deterministically, and a stamp
    left behind would let a later read address the control the fill just targeted.
    """
    return ("(() => { const roots = []; (function walk(root) { roots.push(root); "
            "for (const e of root.querySelectorAll('*')) if (e.shadowRoot) walk(e.shadowRoot); "
            "})(document); const find = (s) => roots.flatMap((r) => Array.from(r.querySelectorAll(s))); "
            "find('[data-hermes-vault-slot]').forEach((n) => n.removeAttribute('data-hermes-vault-slot')); "
            "return JSON.stringify({ unstamped: true }); })()")


_CLEAR_JS_TEMPLATE = """(() => {
__ORIGIN_GUARD__
  const fills = __FILLS__;
  const nonce = __NONCE__;
  const roots = [];
  (function walk(root) {
    roots.push(root);
    for (const e of root.querySelectorAll("*")) if (e.shadowRoot) walk(e.shadowRoot);
  })(document);
  const find = (selector) => roots.flatMap((r) => Array.from(r.querySelectorAll(selector)));
  let cleared = 0;
  for (const f of fills) {
    const el = find('[data-hermes-vault-slot="' + nonce + ':' + f.index + '"]')[0];
    if (!el) continue;
    if (el.tagName === "SELECT") continue;
    const proto = el.tagName === "TEXTAREA" ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
    const setter = Object.getOwnPropertyDescriptor(proto, "value");
    if (setter && setter.set) setter.set.call(el, ""); else el.value = "";
    el.dispatchEvent(new InputEvent("input", { bubbles: true, composed: true, data: "", inputType: "deleteContentBackward" }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    cleared += 1;
  }
  return JSON.stringify({ cleared });
})()"""


def build_clear_js(fills: list[dict[str, Any]], nonce: str = "", *, expected_origin: str) -> str:
    """Build a JS expression that empties stamped text controls so they can be typed again.

    A trusted bulk insert can land the wrong value in a field with an input mask (a phone number,
    a date, a card): the mask drops or reformats characters as they stream in. The fill then reads
    a mismatch, clears the field and retypes it key by key. Clearing goes through the native value
    setter plus an ``input`` event carrying ``data: ""`` so a controlled input ends up consistent
    with an empty field instead of holding a value its state does not know about. Like every
    script of a fill it acts only on ``expected_origin``.
    """
    payload = json.dumps([{"index": f["index"]} for f in fills])
    return (_guarded(_CLEAR_JS_TEMPLATE, expected_origin).replace("__FILLS__", payload)
            .replace("__NONCE__", json.dumps(nonce)))


_VERIFY_JS_TEMPLATE = r"""(() => {
__ORIGIN_GUARD__
  const fills = __FILLS__;
  const nonce = __NONCE__;
  const roots = [];
  (function walk(root) {
    roots.push(root);
    for (const e of root.querySelectorAll("*")) if (e.shadowRoot) walk(e.shadowRoot);
  })(document);
  const find = (selector) => roots.flatMap((r) => Array.from(r.querySelectorAll(selector)));
__OPTION_MATCH__
  const digitsAndSeparators = (v) => /^[\d\s()+\-./]*$/.test(v) && /\d/.test(v);
  const digits = (v) => v.replace(/\D/g, "");
  const groups = (v) => v.match(/\d+/g) || [];
  const sameDigits = (want, shown) => digits(want) === digits(shown);
  // Per token: when a display mask's rewrite still says the same thing.
  const sameMeaning = {
    "cc-number": sameDigits,
    "postal-code": sameDigits,
    // The + is part of the number: 15551234567 dials somewhere else than +15551234567.
    "tel": (want, shown) => want.trim().startsWith("+") === shown.trim().startsWith("+")
      && sameDigits(want, shown),
    // Month and year stay two numbers (1/203 is not 12/03); a mask may drop the month's leading
    // zero (7/29) or the separator (0729), never move a digit across it.
    "cc-exp": (want, shown) => {
      const w = groups(want), g = groups(shown);
      if (g.length === 1) return g[0] === digits(want);
      return g.length === w.length && w.every((part, i) => Number(part) === Number(g[i]));
    },
  };
  const holds = (f, shown) => {
    const want = String(f.value);
    if (shown === want) return true;
    const same = sameMeaning[f.token];
    return Boolean(same) && digitsAndSeparators(want) && digitsAndSeparators(shown) && same(want, shown);
  };
  let landed = 0;
  let reverted = 0;
  const missing = [];
  for (const f of fills) {
    const el = find('[data-hermes-vault-slot="' + nonce + ':' + f.index + '"]')[0];
    if (!el) { missing.push(f.index); continue; }
    if (el.tagName === "SELECT") {
      const selected = el.options[el.selectedIndex];
      if (selected && optionMatches(selected, f.value)) landed += 1; else reverted += 1;
      continue;
    }
    if (holds(f, String(el.value))) landed += 1; else reverted += 1;
  }
  if (__UNSTAMP__) find("[data-hermes-vault-slot]").forEach((n) => n.removeAttribute("data-hermes-vault-slot"));
  return JSON.stringify({ landed, reverted, missing });
})()"""
