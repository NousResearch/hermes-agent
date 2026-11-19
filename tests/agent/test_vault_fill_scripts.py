"""Behaviour of the generated vault page scripts, run in node against a minimal synthetic DOM.

The scripts are what decides whether a secret is typed and whether a fill is reported as landed, so
they are executed, not grepped: each test builds a page (stamped controls, optional open shadow
roots, focus), evaluates the generated expression and reads its verdict by value, after settling it
as an awaiting transport would. No browser, no network.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from agent.vault_login_classifier import (
    build_clear_js,
    build_fill_js,
    build_fill_verify_js,
    build_focus_js,
    build_origin_check_js,
    build_unstamp_js,
)

NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

NONCE = "n1"
ORIGIN = "https://example.com"  # the synthetic page's origin

# A DOM just large enough for the vault scripts: elements with attributes, open shadow roots,
# querySelectorAll for "*" and the stamp attribute, focus with shadow-host retargeting, <select>.
_FAKE_DOM = r"""
// The harness reads a script's answer the way CDP ``returnByValue`` does: natively, own data
// properties only, never through the page's JSON, String, toJSON or prototype accessors. Captured
// before any page setup runs, so a hostile page cannot answer for the harness either.
// It also settles the answer the way an awaiting transport does (CDP ``awaitPromise``, a plugin
// backend's evaluate): ``Promise.resolve`` treats any object with a ``then`` as a thenable and adopts
// whatever that ``then`` resolves, so a verdict that inherits a page-defined ``then`` is replaced.
const __str = JSON.stringify, __own = Object.getOwnPropertyNames, __isArray = Array.isArray;
const __settle = Promise.resolve.bind(Promise), __write = process.stdout.write.bind(process.stdout);
function __byValue(v) {
  if (v === undefined) return "null";
  if (v === null || typeof v !== "object") return __str(v);
  let s = "";
  if (__isArray(v)) {
    for (let i = 0; i < v.length; i++) s += (i ? "," : "") + __byValue(v[i]);
    return "[" + s + "]";
  }
  const keys = __own(v);
  for (let i = 0; i < keys.length; i++) s += (i ? "," : "") + __str(keys[i]) + ":" + __byValue(v[keys[i]]);
  return "{" + s + "}";
}
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


def _verdict(raw: str):
    """What the vault tool makes of a script's by-value answer (``_parse_json_result``, twice)."""
    value = json.loads(raw)
    for _ in range(2):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                break
    return value


def _node(setup: str, expression: str) -> str:
    code = (_FAKE_DOM.replace("__NONCE__", NONCE) + setup
            + "\n__settle(" + expression + ").then((v) => __write(__byValue(v)));")
    return subprocess.run([NODE, "-e", code], capture_output=True, text=True, check=True, timeout=30).stdout


def _run(setup: str, expression: str) -> dict:
    """Build the page with ``setup`` (JS using the fake-DOM helpers), evaluate ``expression``."""
    return _verdict(_node(setup, expression))


class TestFocusAssertion:
    def test_a_focused_password_inside_an_open_shadow_root_holds_focus(self):
        """Discovery and fill reach inputs inside open shadow roots; the focus check must too. At
        document level the focused element is the shadow HOST, so a shallow compare would refuse a
        field that really holds focus."""
        setup = """
          const host = shadowHost(document);
          const pw = stamp(add(host.shadowRoot, "input", { type: "password" }), 1);
        """
        js = build_focus_js([{"index": 1, "token": "current-password"}], nonce=NONCE, expected_origin=ORIGIN)
        assert _run(setup, js) == {"focused": 1}

    def test_focus_moved_to_another_shadow_input_is_refused(self):
        setup = """
          const host = shadowHost(document);
          const pw = stamp(add(host.shadowRoot, "input", { type: "password" }), 1);
          const other = add(host.shadowRoot, "input", { type: "text" });
          // The page steals focus whenever the target is focused.
          pw.focus = () => focusOn(other);
        """
        js = build_focus_js([{"index": 1, "token": "current-password"}], nonce=NONCE, expected_origin=ORIGIN)
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
        assert _run(setup, build_focus_js(fills, nonce=NONCE, expected_origin=ORIGIN, refocus=False))["refused"] == "focus_lost"
        assert _run(setup, build_focus_js(fills, nonce=NONCE, expected_origin=ORIGIN)) == {"focused": 1}

    @pytest.mark.parametrize("root", ["document", "shadowHost(document).shadowRoot"])
    def test_a_page_holding_a_frame_is_refused_even_with_focus(self, root):
        """A frame runs in its own process and can take focus after this check, before the keystroke
        lands; Input.insertText would then type into the frame's document, another origin."""
        setup = """
          const pw = stamp(add(document, "input", { type: "password" }), 1);
          add(ROOT, "iframe");
        """.replace("ROOT", root)
        js = build_focus_js([{"index": 1, "token": "current-password"}], nonce=NONCE, expected_origin=ORIGIN)
        assert _run(setup, js) == {"refused": "frame_on_page"}


class TestReadBack:
    def test_a_select_holding_another_option_is_not_landed(self):
        """The read-back is the claim that the page kept the value: a select on the wrong option
        did not keep it."""
        setup = """
          stamp(select(document, [["US", "United States"], ["CA", "Canada"]], 1), 3);
        """
        js = build_fill_verify_js([{"index": 3, "token": "country-name", "value": "US"}], nonce=NONCE, expected_origin=ORIGIN)
        assert _run(setup, js) == {"landed": 0, "reverted": 1, "missing": []}

    @pytest.mark.parametrize(("value", "selected"), [
        ("United States", 0),   # matched by visible text, case/space-insensitive like the writer
        ("07", 1),              # an expiry month "07" lands on option "7", as the writer picks it
    ])
    def test_a_select_on_the_option_the_writer_picks_is_landed(self, value, selected):
        setup = f"""
          stamp(select(document, [["US", "United States"], ["7", "7"]], {selected}), 3);
        """
        js = build_fill_verify_js([{"index": 3, "token": "country-name", "value": value}], nonce=NONCE, expected_origin=ORIGIN)
        assert _run(setup, js) == {"landed": 1, "reverted": 0, "missing": []}

    def test_the_writer_and_the_read_back_agree_on_a_select(self):
        """Whatever option the fill script selects, the read-back accepts."""
        setup = """
          const s = stamp(select(document, [["", "Choose"], ["7", " July "]], 0), 3);
        """
        fills = [{"index": 3, "token": "cc-exp-month", "value": "july"}]
        expr = ("(() => { " + build_fill_js(fills, expected_origin="https://example.com", nonce=NONCE)
                + "; return " + build_fill_verify_js(fills, nonce=NONCE, expected_origin=ORIGIN) + "; })()")
        assert _run(setup, expr) == {"landed": 1, "reverted": 0, "missing": []}

    @pytest.mark.parametrize(("token", "value", "shown"), [
        ("cc-number", "4111111111111111", "4111 1111 1111 1111"),
        ("cc-exp", "07/29", "07 / 29"),
        ("tel", "5551234567", "(555) 123-4567"),
        ("tel", "+48 600 100 200", "+48600100200"),          # the + stays where it was
        ("cc-exp", "07/29", "7/29"),                          # month 07 shown as 7: the same month
        ("cc-exp", "07/29", "0729"),                          # separator dropped, groups intact
        ("tel", "+48 600 100 200", "(+48) 600 100 200"),      # the country code in parentheses
    ])
    def test_a_mask_that_only_reformats_digits_is_landed(self, token, value, shown):
        setup = f'stamp(add(document, "input", {{ type: "text", value: {json.dumps(shown)} }}), 0);'
        js = build_fill_verify_js([{"index": 0, "token": token, "value": value}], nonce=NONCE, expected_origin=ORIGIN)
        assert _run(setup, js)["landed"] == 1

    @pytest.mark.parametrize(("token", "value", "shown"), [
        ("cc-number", "4111111111111111", "4111 1111 1111 111"),   # a digit lost
        ("tel", "+15551234567", "(555) 123-4567"),                 # the country code dropped
        ("tel", "+15551234567", "15551234567"),                    # only the + dropped: another number
        ("tel", "+48 600 100 200", "48600100200"),
        ("cc-exp", "12/03", "1/203"),                              # same digits, other month and year
        ("cc-exp", "07/29", "07/2029"),
        ("cc-exp", "07/29", "0730"),                               # separator dropped, another year
        ("tel", "48600100200", "(+48) 600 100 200"),               # a + the value never had
        ("tel", "+48 600 100 200", "((+48) 600 100 200"),         # the + leads after one "(" at most
        ("tel", "+48 600 100 200", "48 600 +100 200"),            # a + that no longer leads
        ("tel", "48600100200", "4860010020+0"),                   # a + anywhere but the front
        ("tel", "user@example.com", ""),                           # not a number: exact only
        ("current-password", "12 34", "1234"),                     # passwords stay exact
        ("username", "a-b", "ab"),
    ])
    def test_anything_beyond_formatting_digits_is_reverted(self, token, value, shown):
        setup = f'stamp(add(document, "input", {{ type: "{"password" if token == "current-password" else "text"}", value: {json.dumps(shown)} }}), 0);'
        js = build_fill_verify_js([{"index": 0, "token": token, "value": value}], nonce=NONCE, expected_origin=ORIGIN)
        assert _run(setup, js)["reverted"] == 1


_SENTINEL = "PW-SENTINEL-42"

# Pages that try to read a script's values with nothing but their own JavaScript: a replaced global
# that records its arguments, and one that walks ``caller`` up to the evaluated function and reads
# its source (in sloppy mode that is the whole script, literal values included).
_SPY_PAGES = {
    "argument-spy": """
      globalThis.__seen = [];
      const RealString = String;
      globalThis.String = function (v) { __seen.push(RealString(v)); return RealString(v); };
    """,
    "caller-walk": """
      globalThis.__seen = [];
      const realStringify = JSON.stringify;
      JSON.stringify = function spy(v) {
        if (spy.caller) __seen.push(Function.prototype.toString.call(spy.caller));
        return realStringify(v);
      };
    """,
}


def _secret_scripts():
    fills = [{"index": 1, "token": "current-password", "value": _SENTINEL}]
    return {
        "fill": build_fill_js(fills, expected_origin=ORIGIN, nonce=NONCE),
        "read-back": build_fill_verify_js(fills, nonce=NONCE, expected_origin=ORIGIN),
    }


class TestOriginGuard:
    @pytest.mark.parametrize("page", sorted(_SPY_PAGES))
    @pytest.mark.parametrize("script", ["fill", "read-back"])
    def test_a_page_on_another_origin_never_sees_the_values(self, script, page):
        """The read-back carries every expected value, the password included, just like the fill. Run
        on a page that navigated to another origin (a login that redirects on input, a retargeted tab),
        it must refuse before the values reach anything the page can hook."""
        setup = _SPY_PAGES[page] + """
          globalThis.window.location.origin = "https://other.test";
          stamp(add(document, "input", { type: "password", value: "" }), 1);
        """
        expr = ("(() => { const r = " + _secret_scripts()[script] + "; return { r, saw: __seen.some((s) => s.includes('"
                + _SENTINEL + "')) }; })()")
        out = json.loads(_node(setup, expr))
        assert _verdict(__import__("json").dumps(out["r"]))["refused"] == "origin_changed"
        assert out["saw"] is False

    def test_the_same_page_still_reads_back(self):
        setup = f'stamp(add(document, "input", {{ type: "password", value: "{_SENTINEL}" }}), 1);'
        assert _run(setup, _secret_scripts()["read-back"]) == {"landed": 1, "reverted": 0, "missing": []}

    @pytest.mark.parametrize("build", [
        lambda fills: build_focus_js(fills, nonce=NONCE, expected_origin=ORIGIN),
        lambda fills: build_clear_js(fills, nonce=NONCE, expected_origin=ORIGIN),
    ], ids=["focus", "clear"])
    def test_focus_and_clear_do_not_act_on_another_origin(self, build):
        """The focus check decides where a trusted keystroke lands; neither it nor the clear may act
        on a page that left the bound origin."""
        setup = """
          globalThis.window.location.origin = "https://other.test";
          const pw = stamp(add(document, "input", { type: "password", value: "kept" }), 1);
        """
        out = _run(setup, build([{"index": 1, "token": "current-password"}]))
        assert out["refused"] == "origin_changed"



# Pages on ANOTHER origin that try to answer a check for the checked page. The guard refuses before
# anything else runs, so the only thing such a page can still touch is how that refusal is returned:
# every global and prototype the old scripts reached on the way out.
_FORGED = '{"focused":1,"ok":true,"filled":1,"landed":1,"reverted":0,"missing":[],"cleared":1}'
_LYING_PAGES = {
    "json-stringify": "JSON.stringify = () => " + json.dumps(_FORGED) + ";",
    "string": "globalThis.String = () => " + json.dumps(_FORGED) + ";",
    "to-json": "Object.prototype.toJSON = function () { return " + _FORGED + "; };",
    "prototype-setters": """
      for (const k of ["refused", "found"]) Object.defineProperty(Object.prototype, k, { set() {}, configurable: true });
    """,
    "everything": """
      JSON.stringify = () => '__FORGED__';
      globalThis.String = () => '__FORGED__';
      Object.prototype.toJSON = function () { return JSON.parse('__FORGED__'); };
    """.replace("__FORGED__", _FORGED),
}
# A ``then`` on a prototype every plain object (or array) inherits. An awaiting transport hands the
# verdict to it, and it resolves the forged answer instead: the reviewer's real-Chromium repro. It
# steps aside while resolving so the forged answer is not itself a thenable.
_THENABLE = """
  const forged = JSON.parse('__FORGED__');
  const then = function (resolve) { delete __PROTO__.then; resolve(forged); __PROTO__.then = then; };
  __PROTO__.then = then;
""".replace("__FORGED__", _FORGED)
_LYING_PAGES.update({
    "object-then": _THENABLE.replace("__PROTO__", "Object.prototype"),
    "array-then": _THENABLE.replace("__PROTO__", "Array.prototype"),
})

_GUARDED_SCRIPTS = {
    "origin-check": lambda: build_origin_check_js(ORIGIN),
    "focus": lambda: build_focus_js([{"index": 1, "token": "current-password"}], nonce=NONCE, expected_origin=ORIGIN),
    "clear": lambda: build_clear_js([{"index": 1}], nonce=NONCE, expected_origin=ORIGIN),
    "fill": lambda: _secret_scripts()["fill"],
    "read-back": lambda: _secret_scripts()["read-back"],
}


# Lying pages that leave the scripts' own machinery (``String``) alone, so on the checked origin each
# script can still compute its answer.
_ANSWERING_PAGES = ["json-stringify", "to-json", "object-then", "array-then"]

_OWN_VERDICTS = {
    "origin-check": (_GUARDED_SCRIPTS["origin-check"], {"ok": True}),
    "focus": (_GUARDED_SCRIPTS["focus"], {"focused": 1}),
    "clear": (_GUARDED_SCRIPTS["clear"], {"cleared": 1}),
    "fill": (_GUARDED_SCRIPTS["fill"], {"filled": 1}),
    "read-back": (_GUARDED_SCRIPTS["read-back"], {"landed": 1, "reverted": 0, "missing": []}),
    "unstamp": (lambda: build_unstamp_js(NONCE), {"unstamped": True}),
}


class TestForgedVerdicts:
    @pytest.mark.parametrize("page", sorted(_LYING_PAGES))
    @pytest.mark.parametrize("script", sorted(_GUARDED_SCRIPTS))
    def test_a_page_on_another_origin_cannot_answer_for_the_checked_one(self, script, page):
        """The focus and origin checks decide whether a trusted keystroke carrying the password is
        sent. A tab that redirected to another origin before the check must get a refusal through,
        whatever its own JSON/String/prototypes say: the verdict is a value, not a page call."""
        setup = """
          globalThis.window.location.origin = "https://other.test";
          const pw = stamp(add(document, "input", { type: "password", value: "" }), 1);
          pw.focus();
        """ + _LYING_PAGES[page]
        out = _run(setup, _GUARDED_SCRIPTS[script]())
        assert isinstance(out, dict) and out.get("refused") == "origin_changed", out
        assert out.get("found") == "https://other.test"
        assert not ({"focused", "ok", "filled", "landed", "cleared"} & set(out)), out

    @pytest.mark.parametrize("script", sorted(_GUARDED_SCRIPTS))
    def test_on_the_checked_origin_the_scripts_still_answer(self, script):
        setup = f"""
          const pw = stamp(add(document, "input", {{ type: "password", value: "{_SENTINEL}" }}), 1);
          pw.focus();
        """
        out = _run(setup, _GUARDED_SCRIPTS[script]())
        assert isinstance(out, dict) and "refused" not in out, out

    @pytest.mark.parametrize("page", ["json-stringify", "everything", "object-then", "array-then"])
    def test_a_page_cannot_claim_a_focus_it_does_not_hold(self, page):
        """On the checked origin too, the focus verdict is the script's own: focus held by another
        element (an iframe's host, a decoy input) reads as lost whatever the page's JSON says."""
        setup = """
          const pw = stamp(add(document, "input", { type: "password", value: "" }), 1);
          const decoy = add(document, "iframe");
          pw.focus = () => focusOn(decoy);
        """ + _LYING_PAGES[page]
        out = _run(setup, _GUARDED_SCRIPTS["focus"]())
        assert isinstance(out, dict) and out.get("refused") == "focus_lost", out

    @pytest.mark.parametrize("page", _ANSWERING_PAGES)
    def test_a_read_back_of_a_rerendered_field_is_not_landed(self, page):
        """On the checked origin the read-back's counts are the script's own: a field that lost its
        stamp (re-rendered) is missing, not landed, whatever JSON or ``then`` the page defined."""
        setup = 'add(document, "input", { type: "password", value: "" });' + _LYING_PAGES[page]
        assert _run(setup, _secret_scripts()["read-back"]) == {"landed": 0, "reverted": 0, "missing": [1]}

    @pytest.mark.parametrize("page", _ANSWERING_PAGES)
    @pytest.mark.parametrize("script", sorted(_OWN_VERDICTS))
    def test_on_the_checked_origin_every_answer_is_the_scripts_own(self, script, page):
        """A page that keeps its origin still cannot put words in a script's mouth: each script's
        answer is exactly what it computed, not the page's JSON or ``then``."""
        setup = f"""
          const pw = stamp(add(document, "input", {{ type: "password", value: "{_SENTINEL}" }}), 1);
          pw.focus();
        """ + _LYING_PAGES[page]
        build, expected = _OWN_VERDICTS[script]
        assert _run(setup, build()) == expected

    @pytest.mark.parametrize("page", _ANSWERING_PAGES)
    @pytest.mark.parametrize(("fills", "control", "refused"), [
        ([], "password", "not_found"),                     # no field to check
        ([{"index": 1, "token": "current-password"}], "", "not_found"),  # the stamp is gone
        ([{"index": 1, "token": "current-password"}], "text", "not_password"),
    ], ids=["no-field", "stamp-gone", "not-a-password-box"])
    def test_a_focus_refusal_is_not_turned_into_focus(self, fills, control, refused, page):
        setup = ((f'stamp(add(document, "input", {{ type: "{control}" }}), 1).focus();' if control
                  else 'add(document, "input", { type: "password" }).focus();') + _LYING_PAGES[page])
        assert _run(setup, build_focus_js(fills, nonce=NONCE, expected_origin=ORIGIN)) == {"refused": refused}
