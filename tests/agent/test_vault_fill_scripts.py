"""Behaviour of the generated vault page scripts, run in node against a minimal synthetic DOM.

The scripts are what decides whether a secret is typed and whether a fill is reported as landed, so
they are executed, not grepped: each test builds a page (stamped controls, optional open shadow
roots, focus), evaluates the generated expression and reads its JSON verdict. No browser, no network.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from agent.vault_login_classifier import build_fill_js, build_fill_verify_js, build_focus_js

NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

NONCE = "n1"

# A DOM just large enough for the vault scripts: elements with attributes, open shadow roots,
# querySelectorAll for "*" and the stamp attribute, focus with shadow-host retargeting, <select>.
_FAKE_DOM = r"""
class Node {
  constructor(tag, props) {
    Object.assign(this, { tagName: tag.toUpperCase(), attrs: {}, children: [], shadowRoot: null,
                          value: "", type: "", parent: null, root: null }, props || {});
    this.listeners = [];
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  removeAttribute(k) { delete this.attrs[k]; }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  dispatchEvent(e) { for (const l of this.listeners) l(e); return true; }
  focus() { focusOn(this); }
}
class Root {
  constructor(host) { this.host = host; this.children = []; this.activeElement = null; }
  all() { const out = []; const walk = (n) => { for (const c of n.children) { out.push(c); walk(c); } }; walk(this); return out; }
  querySelectorAll(sel) {
    const nodes = this.all();
    if (sel === "*") return nodes;
    let m = sel.match(/^\[([\w-]+)="(.*)"\]$/);
    if (m) return nodes.filter((n) => n.getAttribute(m[1]) === m[2]);
    m = sel.match(/^\[([\w-]+)\]$/);
    if (m) return nodes.filter((n) => n.getAttribute(m[1]) !== null);
    throw new Error("unsupported selector " + sel);
  }
}
const document = new Root(null);
globalThis.document = document;
globalThis.window = { location: { origin: "https://example.com" } };
class InputEvent { constructor(type, init) { Object.assign(this, { type }, init || {}); } }
class Event { constructor(type, init) { Object.assign(this, { type }, init || {}); } }
globalThis.InputEvent = InputEvent; globalThis.Event = Event;
const valueDesc = { set(v) { this.value = v; } };
globalThis.HTMLInputElement = { prototype: {} };
globalThis.HTMLTextAreaElement = { prototype: {} };
Object.defineProperty(HTMLInputElement.prototype, "value", { set(v) { this._v = v; }, configurable: true });
// Focus as the browser reports it: every root's activeElement is the element in that root on the
// path to the focused node, so a node inside a shadow root shows up as its HOST at document level.
function focusOn(node) {
  const clear = (root) => { root.activeElement = null; for (const n of root.all()) if (n.shadowRoot) clear(n.shadowRoot); };
  clear(document);
  let n = node;
  while (n) { n.root.activeElement = n; n = n.root.host; }
}
function add(root, tag, props) {
  const n = new Node(tag, props); n.root = root; root.children.push(n); return n;
}
function shadowHost(root) {
  const host = add(root, "div"); host.shadowRoot = new Root(host); return host;
}
function stamp(n, index) { n.setAttribute("data-hermes-vault-slot", "__NONCE__:" + index); return n; }
function select(root, options, selected) {
  const s = add(root, "select");
  s.options = options.map(([value, text]) => ({ value, textContent: text }));
  s.selectedIndex = selected;
  Object.defineProperty(s, "value", {
    get() { const o = this.options[this.selectedIndex]; return o ? o.value : ""; },
    set(v) { this.selectedIndex = this.options.findIndex((o) => o.value === v); },
  });
  return s;
}
"""


def _run(setup: str, expression: str) -> dict:
    """Build the page with ``setup`` (JS using the fake-DOM helpers), evaluate ``expression``."""
    code = (_FAKE_DOM.replace("__NONCE__", NONCE) + setup
            + "\nprocess.stdout.write(String(" + expression + "));")
    out = subprocess.run([NODE, "-e", code], capture_output=True, text=True, check=True, timeout=30)
    return json.loads(out.stdout)


class TestFocusAssertion:
    def test_a_focused_password_inside_an_open_shadow_root_holds_focus(self):
        """Discovery and fill reach inputs inside open shadow roots; the focus check must too. At
        document level the focused element is the shadow HOST, so a shallow compare would refuse a
        field that really holds focus."""
        setup = """
          const host = shadowHost(document);
          const pw = stamp(add(host.shadowRoot, "input", { type: "password" }), 1);
        """
        js = build_focus_js([{"index": 1, "token": "current-password"}], nonce=NONCE)
        assert _run(setup, js) == {"focused": 1}

    def test_focus_moved_to_another_shadow_input_is_refused(self):
        setup = """
          const host = shadowHost(document);
          const pw = stamp(add(host.shadowRoot, "input", { type: "password" }), 1);
          const other = add(host.shadowRoot, "input", { type: "text" });
          // The page steals focus whenever the target is focused.
          pw.focus = () => focusOn(other);
        """
        js = build_focus_js([{"index": 1, "token": "current-password"}], nonce=NONCE)
        assert _run(setup, js)["refused"] == "focus_lost"


    def test_a_between_keys_check_does_not_pull_focus_back(self):
        """Between the keys of one field the check only asserts: re-focusing would hide that the
        page moved focus after the previous key, and the next key would be typed on a page that
        is actively redirecting input."""
        setup = """
          const pw = stamp(add(document, "input", { type: "password" }), 1);
          const other = add(document, "input", { type: "text" });
          other.focus();
        """
        fills = [{"index": 1, "token": "current-password"}]
        assert _run(setup, build_focus_js(fills, nonce=NONCE, refocus=False))["refused"] == "focus_lost"
        assert _run(setup, build_focus_js(fills, nonce=NONCE)) == {"focused": 1}


class TestReadBack:
    def test_a_select_holding_another_option_is_not_landed(self):
        """The read-back is the claim that the page kept the value: a select on the wrong option
        did not keep it."""
        setup = """
          stamp(select(document, [["US", "United States"], ["CA", "Canada"]], 1), 3);
        """
        js = build_fill_verify_js([{"index": 3, "token": "country-name", "value": "US"}], nonce=NONCE)
        assert _run(setup, js) == {"landed": 0, "reverted": 1, "missing": []}

    @pytest.mark.parametrize(("value", "selected"), [
        ("United States", 0),   # matched by visible text, case/space-insensitive like the writer
        ("07", 1),              # an expiry month "07" lands on option "7", as the writer picks it
    ])
    def test_a_select_on_the_option_the_writer_picks_is_landed(self, value, selected):
        setup = f"""
          stamp(select(document, [["US", "United States"], ["7", "7"]], {selected}), 3);
        """
        js = build_fill_verify_js([{"index": 3, "token": "country-name", "value": value}], nonce=NONCE)
        assert _run(setup, js) == {"landed": 1, "reverted": 0, "missing": []}

    def test_the_writer_and_the_read_back_agree_on_a_select(self):
        """Whatever option the fill script selects, the read-back accepts."""
        setup = """
          const s = stamp(select(document, [["", "Choose"], ["7", " July "]], 0), 3);
        """
        fills = [{"index": 3, "token": "cc-exp-month", "value": "july"}]
        expr = ("(() => { " + build_fill_js(fills, expected_origin="https://example.com", nonce=NONCE)
                + "; return " + build_fill_verify_js(fills, nonce=NONCE) + "; })()")
        assert _run(setup, expr) == {"landed": 1, "reverted": 0, "missing": []}

    @pytest.mark.parametrize(("token", "value", "shown"), [
        ("cc-number", "4111111111111111", "4111 1111 1111 1111"),
        ("cc-exp", "07/29", "07 / 29"),
        ("tel", "5551234567", "(555) 123-4567"),
    ])
    def test_a_mask_that_only_reformats_digits_is_landed(self, token, value, shown):
        setup = f'stamp(add(document, "input", {{ type: "text", value: {json.dumps(shown)} }}), 0);'
        js = build_fill_verify_js([{"index": 0, "token": token, "value": value}], nonce=NONCE)
        assert _run(setup, js)["landed"] == 1

    @pytest.mark.parametrize(("token", "value", "shown"), [
        ("cc-number", "4111111111111111", "4111 1111 1111 111"),   # a digit lost
        ("tel", "+15551234567", "(555) 123-4567"),                 # the country code dropped
        ("tel", "user@example.com", ""),                           # not a number: exact only
        ("current-password", "12 34", "1234"),                     # passwords stay exact
        ("username", "a-b", "ab"),
    ])
    def test_anything_beyond_formatting_digits_is_reverted(self, token, value, shown):
        setup = f'stamp(add(document, "input", {{ type: "{"password" if token == "current-password" else "text"}", value: {json.dumps(shown)} }}), 0);'
        js = build_fill_verify_js([{"index": 0, "token": token, "value": value}], nonce=NONCE)
        assert _run(setup, js)["reverted"] == 1
