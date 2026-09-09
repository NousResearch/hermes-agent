"""Login-form control classifier for vault autofill.

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

LOGIN_AUTOFILL_TOKENS = ("username", "email", "tel", "current-password")

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

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LoginControl":
        form_index = raw.get("formIndex", raw.get("form_index"))
        return cls(
            autocomplete=str(raw.get("autocomplete") or ""),
            form_index=int(form_index) if form_index is not None else None,
            index=int(raw.get("index") or 0),
            label=str(raw.get("label") or ""),
            name=str(raw.get("name") or ""),
            type=str(raw.get("type") or ""),
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


def select_password_fill(
    classified: List[ClassifiedLoginControl],
    password: str,
) -> List[Dict[str, Any]]:
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
    best_password = sorted(
        passwords, key=lambda c: (-c.score, c.control.index)
    )[0]
    return [
        {
            "index": best_password.control.index,
            "token": "current-password",
            "value": password,
        }
    ]


# JS expression evaluated in the page to inspect candidate input controls.
# Ported from OpenInstinct's nativeLoginControlInspectionExpression.
LOGIN_CONTROL_INSPECTION_JS = """(() => {
  const elements = Array.from(document.querySelectorAll("input"));
  const forms = Array.from(document.forms);
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
    return [{
      autocomplete: element.autocomplete || "",
      formIndex: resolvedFormIndex >= 0 ? resolvedFormIndex : null,
      index,
      label: [
        ...labels,
        element.getAttribute("aria-label") || "",
        ariaText,
        element.getAttribute("placeholder") || "",
        element.getAttribute("title") || "",
      ].join(" "),
      name: [element.name, element.id].join(" "),
      type: element.type || "",
    }];
  });
  return JSON.stringify(out);
})()"""


def build_fill_js(fills: List[Dict[str, Any]], expected_origin: str) -> str:
    """Build a JS expression that fills the selected inputs and reports
    only a count. The returned expression never echoes the values back.

    ``expected_origin`` is asserted against ``window.location.origin``
    synchronously inside the SAME evaluated script, immediately before any
    write. If the page navigated between inspection and fill (TOCTOU), the
    script writes nothing and returns
    ``{"refused": "origin_changed", "found": <actual origin>}`` — proof
    scope equals mutation scope (#88706). No marker attribute is set on
    filled controls: filled fields must not be deterministically
    addressable by later model-driven DOM reads.
    """
    payload = json.dumps(
        [{"index": f["index"], "value": f["value"]} for f in fills]
    )
    expected = json.dumps(expected_origin)
    return (
        "(() => {\n"
        f"  const expectedOrigin = {expected};\n"
        "  if (window.location.origin !== expectedOrigin) {\n"
        "    return JSON.stringify({ refused: \"origin_changed\", found: window.location.origin });\n"
        "  }\n"
        f"  const fills = {payload};\n"
        "  const elements = Array.from(document.querySelectorAll(\"input\"));\n"
        "  let filled = 0;\n"
        "  for (const f of fills) {\n"
        "    const el = elements[f.index];\n"
        "    if (!el) continue;\n"
        "    try {\n"
        "      el.focus();\n"
        "      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, \"value\");\n"
        "      if (setter && setter.set) { setter.set.call(el, f.value); } else { el.value = f.value; }\n"
        "      el.dispatchEvent(new InputEvent(\"input\", { bubbles: true, inputType: \"insertText\" }));\n"
        "      el.dispatchEvent(new Event(\"change\", { bubbles: true }));\n"
        "      if (el.value.length > 0) filled += 1;\n"
        "    } catch (e) { /* skip */ }\n"
        "  }\n"
        "  return JSON.stringify({ filled });\n"
        "})()"
    )
