"""United-shaped synthetic DOM regression tests; never real credentials or codes."""
import json
import shutil
import subprocess

import pytest

from agent.vault_login_classifier import (
    LoginControl, build_inspection_js, build_otp_probe_js, classify_otp_controls,
    build_otp_fills, build_fill_js,
    ClassifiedLoginControl,
)

# Metadata captured from United: nested item DIVs, common container and real form;
# autocomplete off, no maxlength, numeric inputmode, positional accessible labels.
def united_fields(total=6):
    return [dict(type="text", autocomplete="off", maxLength=-1,
                 ariaLabel=f"Enter code Enter digit {i} of {total}",
                 inputmode="numeric", form=0, group=0) for i in range(1, total + 1)]


DOM_SETUP = r'''
const specs = JSON.parse(process.argv[1]);
const forms = [ {}, {} ];
const groups = new Map();
class Input {
  constructor(c) {
    this.spec = c; this.type = c.type || "text"; this.tagName = "INPUT";
    this.name = c.name || ""; this.id = ""; this.autocomplete = c.autocomplete || "";
    this.maxLength = c.maxLength ?? -1; this.inputMode = c.inputmode || "";
    this.form = c.form == null ? null : forms[c.form]; this.labels = [];
    this.disabled = !!c.disabled; this.readOnly = !!c.readOnly;
    this.attrs = {"aria-label": c.ariaLabel || ""}; this.events = []; this._value = "";
    if (c.group != null) {
      const key = String(c.form) + ":" + c.group;
      if (!groups.has(key)) groups.set(key, {tagName: "DIV", members: [], parentElement: null,
        querySelectorAll() { return this.members; }});
      const group = groups.get(key); group.members.push(this);
      this.parentElement = {tagName: "DIV", parentElement: group, querySelectorAll: () => [this]};
    } else this.parentElement = null;
  }
  get value() { return this._value; }
  set value(v) { this._value = v; }
  getAttribute(k) { return this.attrs[k] || ""; }
  setAttribute(k,v) { this.attrs[k] = v; }
  removeAttribute(k) { delete this.attrs[k]; }
  getClientRects() { return this.spec.hidden ? [] : [{}]; }
  focus() {}
  dispatchEvent(e) { this.events.push(e.type); }
}
const elements = specs.map(c => new Input(c));
global.HTMLInputElement = Input;
global.InputEvent = global.Event = class { constructor(type) { this.type = type; } };
global.window = {location: {origin: "https://otp.example.test"}};
global.document = {forms, getElementById: () => null,
  querySelectorAll: s => s === "[data-hermes-vault-slot]" ? elements.filter(e => e.attrs["data-hermes-vault-slot"]) : elements,
  querySelector: s => elements.find(e => s === '[data-hermes-vault-slot="' + e.attrs["data-hermes-vault-slot"] + '"]') || null};
global.getComputedStyle = e => ({display: e.spec.display || "block", visibility: e.spec.visibility || "visible"});
'''


def dom_eval(fields, expression, prelude=""):
    node = shutil.which("node")
    if not node:
        pytest.skip("node required for actual generated inspection/probe/fill JS")
    script = DOM_SETUP + prelude + "console.log(JSON.stringify(" + expression + "));"
    result = subprocess.run([node, "-e", script, json.dumps(fields)],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def inspect(fields):
    raw = json.loads(dom_eval(fields, build_inspection_js("synthetic-nonce")))
    return [LoginControl.from_dict(c) for c in raw]


def test_united_positional_detection_python_and_generated_probe():
    fields = united_fields()
    controls = inspect(fields)
    assert len(classify_otp_controls(controls)) == 6
    # Probe must not leave stamps or perform writes.
    assert dom_eval(fields, "(() => { const found = " + build_otp_probe_js() + "; return [found, elements.every(e => !e.attrs['data-hermes-vault-slot'] && e.value === '')]; })()") == [True, True]


def test_united_fills_one_synthetic_digit_per_box_through_generated_js():
    fields = united_fields()
    fills = build_otp_fills(classify_otp_controls(inspect(fields)), "024680")
    assert [f["value"] for f in fills] == list("024680")
    result = dom_eval(fields, "(() => { const result = JSON.parse(" +
                      build_fill_js(fills, "https://otp.example.test", "synthetic-nonce") +
                      "); return [result, elements.map(e => e.value), elements.map(e => e.events), elements.every(e => !e.attrs['data-hermes-vault-slot'])]; })()",
                      prelude=build_inspection_js("synthetic-nonce") + ";")
    assert result == [{"filled": 6}, list("024680"), [["input", "change"]] * 6, True]


@pytest.mark.parametrize("total", [4, 5, 6, 7, 8])
@pytest.mark.parametrize("context", ["Enter code Enter", "Verification code", "Authentication code"])
def test_generic_positional_widgets_with_supported_totals(total, context):
    fields = united_fields(total)
    for i, f in enumerate(fields, 1):
        f["ariaLabel"] = f"{context} digit {i} of {total}"
    controls = classify_otp_controls(inspect(fields))
    assert len(controls) == total
    assert dom_eval(fields, build_otp_probe_js()) is True
    assert [f["value"] for f in build_otp_fills(controls, "0" * total)] == ["0"] * total


def invalid_widgets():
    for text in ["Enter code", "Postal code", "Country code", "Promo code", "Enter digit", "Enter code Postal"]:
        fields = united_fields()
        for i, f in enumerate(fields, 1):
            f["ariaLabel"] = text if text == "Enter code" else f"{text} digit {i} of 6"
        yield text, fields
    for total in [3, 9]:
        yield f"unsupported total {total}", united_fields(total)
    for key, value in [("form", 1), ("group", 1), ("hidden", True), ("disabled", True),
                       ("readOnly", True), ("visibility", "hidden"), ("display", "none"),
                       ("ariaLabel", "Enter code Enter digit 5 of 6"),
                       ("ariaLabel", "Enter code Enter digit 6 of 7"),
                       ("ariaLabel", "Enter code Enter digit 0 of 6"),
                       ("ariaLabel", "Enter code Enter digit 7 of 6"),
                       ("ariaLabel", "Enter code Enter digit 6 of 6 digit 6 of 6"),
                       ("ariaLabel", "Verification code digit 6 of unknown"),
                       ("ariaLabel", "Verification code digit 6 of"),
                       ("type", "email")]:
        fields = united_fields()
        fields[-1][key] = value
        yield f"mixed {key} {value}", fields
    fields = united_fields()
    fields[2], fields[3] = fields[3], fields[2]
    yield "out of order", fields
    yield "missing", united_fields()[:-1]
    fields = united_fields()
    fields.insert(3, dict(type="text", ariaLabel="Search"))
    yield "not adjacent", fields
    fields = united_fields()
    for f in fields:
        f["group"] = f["form"] = None
    yield "no actual form or group", fields
    yield "two complete groups", united_fields() + [dict(f, group=1) for f in united_fields()]


@pytest.mark.parametrize("name,fields", list(invalid_widgets()), ids=lambda x: x if isinstance(x, str) else None)
def test_invalid_widgets_fail_closed_in_python_and_generated_js(name, fields):
    controls = inspect(fields)
    assert classify_otp_controls(controls) == [], name
    assert dom_eval(fields, build_otp_probe_js()) is False, name
    # Defense in depth: positional malformed targets must not get the WHOLE code
    # even if an upstream caller incorrectly supplies them as classified.
    if any(" of " in c.label for c in controls):
        forced = [ClassifiedLoginControl(c, 100, "one-time-code") for c in controls]
        assert build_otp_fills(forced, "024680") == [], name


@pytest.mark.parametrize("code", ["", "12345", "1234567"])
def test_positional_code_length_mismatch_never_falls_back_to_whole_code(code):
    assert build_otp_fills(classify_otp_controls(inspect(united_fields())), code) == []


def test_formless_widget_requires_actual_shared_container():
    fields = [dict(f, form=None) for f in united_fields()]
    assert len(classify_otp_controls(inspect(fields))) == 6
    assert dom_eval(fields, build_otp_probe_js()) is True


@pytest.mark.parametrize("origin,nonce,refusal", [
    ("https://other.example.test", "synthetic-nonce", {"refused": "origin_changed", "found": "https://otp.example.test"}),
    ("https://otp.example.test", "stale-nonce", {"filled": 0}),
    ("https://otp.example.test:8443", "synthetic-nonce", {"refused": "origin_changed", "found": "https://otp.example.test"}),
])
def test_split_fill_retains_exact_origin_and_nonce_guards(origin, nonce, refusal):
    fields = united_fields()
    fills = build_otp_fills(classify_otp_controls(inspect(fields)), "024680")
    result = dom_eval(fields, "(() => { const result = JSON.parse(" + build_fill_js(fills, origin, nonce) +
                      "); return [result, elements.every(e => e.value === '' && e.events.length === 0)]; })()",
                      prelude=build_inspection_js("synthetic-nonce") + ";")
    assert result == [refusal, True]


def test_enter_code_refuses_empty_fill_plan_without_secret_evaluation():
    from unittest.mock import patch
    from tools import browser_vault_tool as tool
    fields = [dict(index=i, type="text", label=f["ariaLabel"], formIndex=0, groupIndex=0)
              for i, f in enumerate(united_fields())]
    with patch.object(tool, "_focus_bound_origin"), \
         patch.object(tool, "_current_page_origin", return_value="https://otp.example.test"), \
         patch.object(tool, "_eval_js", return_value={"success": True, "result": json.dumps(fields)}), \
         patch.object(tool, "_eval_js_secret") as secret, \
         patch("agent.vault_backends.unlock.can_prompt_here", return_value=True), \
         patch("agent.vault_backends.unlock.get_code_prompt_callback", return_value=lambda *_: "12345"):
        result = json.loads(tool.browser_vault_enter_code())
    secret.assert_not_called()
    assert result["success"] is False
    assert result["error_type"] == "invalid_code_fields"


def test_maxlength_boxes_without_actual_form_or_group_are_not_split():
    fields = [dict(type="tel", autocomplete="one-time-code", maxLength=1) for _ in range(6)]
    fills = build_otp_fills(classify_otp_controls(inspect(fields)), "024680")
    assert len(fills) == 1


@pytest.mark.parametrize("form,group", [(0, None), (0, 0), (None, 0)])
def test_classic_maxlength_widget_still_splits_with_real_form_or_group(form, group):
    fields = [dict(type="tel", autocomplete="one-time-code", maxLength=1, form=form, group=group) for _ in range(6)]
    fills = build_otp_fills(classify_otp_controls(inspect(fields)), "024680")
    assert [f["value"] for f in fills] == list("024680")


@pytest.mark.parametrize("label,autocomplete", [("Verification code", "off"), ("Enter the 6-digit code", "off"), ("", "one-time-code")])
def test_normal_single_field_keeps_whole_code(label, autocomplete):
    fields = [dict(type="text", ariaLabel=label, autocomplete=autocomplete)]
    assert build_otp_fills(classify_otp_controls(inspect(fields)), "024680") == [
        {"index": 0, "token": "one-time-code", "value": "024680"}]