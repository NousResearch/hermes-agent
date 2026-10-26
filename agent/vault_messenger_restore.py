"""Scoped adapter for a saved secret in Messenger's nonstandard text PIN field.

The login item must be named exactly ``facebook messenger pin`` and explicitly
bind ``https://www.facebook.com``. Only the English restore dialog below is
supported. This deliberately does not broaden generic password/OTP detection;
all writes retain the existing supervisor, origin, nonce and redaction path.
"""

import json

from agent.vault_login_classifier import build_fill_js, build_inspection_js

RESTORE_LABEL = "facebook messenger pin"
RESTORE_ORIGIN = "https://www.facebook.com"

# Shared by tab selection, inspection, and the synchronous write-time guard.
_RESTORE_TARGETS_JS = r"""(() => {
  if (window.location.origin !== "https://www.facebook.com") return [];
  const visible = el => {
    const s = getComputedStyle(el);
    return s.display !== 'none' && s.visibility !== 'hidden' && el.getClientRects().length > 0;
  };
  return Array.from(document.querySelectorAll('input')).filter(el => {
    if (el.type !== 'text' || el.disabled || el.readOnly || !visible(el) ||
        el.getAttribute('aria-label') !== 'PIN') return false;
    const dialog = el.closest('[role="dialog"]');
    if (!dialog || !visible(dialog)) return false;
    return Array.from(dialog.querySelectorAll('h1,h2,h3,[role="heading"]')).some(h =>
      visible(h) && h.textContent.trim() === 'Enter your PIN to restore your chats');
  });
})()"""
RESTORE_TAB_PROBE = f"({_RESTORE_TARGETS_JS}).length === 1"


def build_restore_inspection_js(nonce: str) -> str:
    """Inspect only the unique restore field, retaining the usual nonce stamp."""
    return f"""(() => {{
      const targets = {_RESTORE_TARGETS_JS};
      if (targets.length !== 1) return JSON.stringify([]);
      const controls = JSON.parse({build_inspection_js(nonce)});
      const stamp = targets[0].getAttribute('data-hermes-vault-slot');
      return JSON.stringify(controls.filter(c => stamp === {json.dumps(nonce)} + ':' + c.index));
    }})()"""


def build_restore_fill_js(fills: list[dict], expected_origin: str, nonce: str) -> str:
    """Revalidate the exact stamped target and dialog before any secret is written."""
    indices = json.dumps([f["index"] for f in fills])
    return f"""(() => {{
      const targets = {_RESTORE_TARGETS_JS};
      const indices = {indices};
      if (targets.length !== 1 || indices.length !== 1 ||
          targets[0].getAttribute('data-hermes-vault-slot') !== {json.dumps(nonce)} + ':' + indices[0])
        return JSON.stringify({{filled: 0}});
      return {build_fill_js(fills, expected_origin, nonce)};
    }})()"""
