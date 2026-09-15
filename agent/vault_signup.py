"""Validation and page-side guards for generated registration credentials."""
import json
import re
import unicodedata
from urllib.parse import urlsplit


def validate_label(label: str) -> str:
    if (not isinstance(label, str) or not label or label != label.strip()
            or len(label) > 120 or label.startswith("-")
            or any(unicodedata.category(c).startswith("C") for c in label)
            or label.startswith("[Pending signup ")):
        raise ValueError("Invalid signup label")
    return label


def validate_origin(origin: str) -> str:
    if not isinstance(origin, str) or any(c.isspace() or ord(c) < 32 for c in origin):
        raise ValueError("Invalid signup origin")
    try:
        parts = urlsplit(origin)
        valid = (parts.scheme in ("https", "http") and parts.hostname and not parts.username
                 and not parts.password and not parts.path and not parts.query and not parts.fragment
                 and parts.port != 0 and "\\" not in origin)
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Invalid signup origin")
    return origin


# Shared by Python classification and the synchronous JS recheck.
_CURRENT = r"\b(?:current|old|existing)\s*(?:account\s*)?password\b"
_NEW = r"\b(?:new|create|confirm|repeat|retype)\s*(?:(?:a|your|new)\s+)?password\b|\bpassword\s*(?:confirmation|confirm|repeat)\b"


def classify_signup_controls(controls):
    """Only positively identified, visible new/confirmation password controls, in one form."""
    if not isinstance(controls, list) or any(not isinstance(c, dict) for c in controls):
        return []
    selected = []
    for c in controls:
        if (type(c.get("index")) is not int or c["index"] < 0
                or c.get("type") != "password" or c.get("visible") is False
                or c.get("disabled") or c.get("readOnly")):
            continue
        tokens = str(c.get("autocomplete", "")).lower().split()
        text = " ".join(str(c.get(k, "")) for k in ("name", "label"))
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"[^a-z0-9]+", " ", text.lower())
        if "current-password" in tokens or re.search(_CURRENT, text):
            continue
        if "new-password" in tokens or re.search(_NEW, text):
            selected.append(c)
    if len({c["index"] for c in selected}) != len(selected):
        return []
    if len({c.get("formIndex") for c in selected}) > 1:
        return []
    if len(selected) == 1 and "new-password" not in str(selected[0].get("autocomplete", "")).lower().split():
        return []
    return selected


def build_signup_inspection_js(nonce):
    from agent.vault_login_classifier import build_inspection_js
    return build_inspection_js(nonce).replace("data-hermes-vault-slot", "data-hermes-signup-" + nonce)


def build_signup_cleanup_js(nonce):
    attr = json.dumps("data-hermes-signup-" + nonce)
    return "(() => { const attr = " + attr + "; document.querySelectorAll('[' + attr + ']').forEach(el => el.removeAttribute(attr)); return true; })()"


def build_signup_fill_js(controls, password, origin, nonce):
    substitutions = {
        "__ORIGIN__": json.dumps(origin), "__NONCE__": json.dumps(nonce),
        "__INDICES__": json.dumps([c["index"] for c in controls]),
        "__CURRENT__": _CURRENT, "__NEW__": _NEW, "__PASSWORD__": json.dumps(password),
    }
    return re.sub(r"__[A-Z]+__", lambda m: substitutions[m[0]], _FILL)



_FILL = r'''(() => {
  if (window.location.origin !== __ORIGIN__) return JSON.stringify({filled: 0});
  const indices = __INDICES__, nonce = __NONCE__;
  const attr = 'data-hermes-signup-' + nonce;
  const matches = indices.map(i => document.querySelectorAll('[' + attr + '="' + nonce + ':' + i + '"]'));
  if (matches.some(nodes => nodes.length !== 1)) return JSON.stringify({filled: 0});
  const targets = matches.map(nodes => nodes[0]);
  const eligible = el => {
    if (!el || el.type !== 'password' || el.disabled || el.readOnly) return false;
    const style = getComputedStyle(el);
    if (style.display === 'none' || style.visibility !== 'visible' || Number.parseFloat(style.opacity) === 0 || !el.getClientRects().length) return false;
    const tokens = (el.autocomplete || '').toLowerCase().split(/\s+/);
    const labels = Array.from(el.labels || [], l => l.textContent || '');
    const aria = (el.getAttribute('aria-labelledby') || '').split(/\s+/).map(id => document.getElementById(id)?.textContent || '');
    const text = [el.name, el.id, ...labels, ...aria, ...['aria-label','placeholder','title'].map(k => el.getAttribute(k) || '')]
      .join(' ').replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase().replace(/[^a-z0-9]+/g, ' ');
    return !tokens.includes('current-password') && !/__CURRENT__/.test(text)
      && (tokens.includes('new-password') || /__NEW__/.test(text));
  };
  if (!targets.length || !targets.every(eligible) || new Set(targets.map(el => el.form)).size !== 1)
    return JSON.stringify({filled: 0});
  if (targets.length === 1 && !(targets[0].autocomplete || '').toLowerCase().split(/\s+/).includes('new-password'))
    return JSON.stringify({filled: 0});
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
  // Set every value before firing page handlers, which may replace the confirmation control.
  if (window.location.origin !== __ORIGIN__) return JSON.stringify({filled: 0});
  for (const el of targets) setter.call(el, __PASSWORD__);
  for (const el of targets) {
    el.removeAttribute(attr);
    el.dispatchEvent(new InputEvent('input', {bubbles: true, inputType: 'insertText'}));
    el.dispatchEvent(new Event('change', {bubbles: true}));
  }
  return JSON.stringify({filled: targets.length});
})()'''
