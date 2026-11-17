"""Nonsecret widget proof evaluated only in a supervisor-owned isolated world.

The returned remote object stays inside CDP: no page-global proof, element
selector or secret is handed to the model. This observes an authorized site's
masking behavior; it does not isolate a credential from that site's own code.
"""

import json


MASKED_PASSWORD_TAB_PROBE = """!!document.querySelector(
 'input[type=text][name*="password" i],input[type=text][id*="password" i],input[autocomplete="current-password"]'
)"""


def build_masked_password_probe(expected_origin: str, deadline_ms: int) -> str:
    return _PROBE.replace("__ORIGIN__", json.dumps(expected_origin)).replace(
        "__DEADLINE__", str(deadline_ms)
    )


_PROBE = r"""(async () => {
  const origin = __ORIGIN__;
  const deadline = __DEADLINE__;
  const refuse = reason => JSON.stringify({refused: reason});
  if (location.origin !== origin || window.top !== window) return refuse('origin_or_frame');
  const doc = document, href = location.href, title = document.title;
  const norm = text => text.replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase();
  const visible = el => {
    const s = getComputedStyle(el), r = el.getBoundingClientRect();
    return el.isConnected && el.ownerDocument === doc && !el.disabled && !el.readOnly &&
      s.display !== 'none' && s.visibility === 'visible' && Number(s.opacity) >= 0.9 &&
      r.width > 0 && r.height > 0;
  };
  const label = el => [el.name, el.id, el.autocomplete, el.getAttribute('aria-label') || '',
    el.placeholder, el.title, ...Array.from(el.labels || [], l => l.textContent || ''),
    ...(el.getAttribute('aria-labelledby') || '').split(/\s+/).filter(Boolean)
      .map(id => doc.getElementById(id)?.textContent || '')].join(' ');
  const semantics = el => {
    const text = norm(label(el));
    return el.type === 'text' && /\b(?:password|passphrase)\b/.test(text) &&
      !/\b(?:new|confirm|create|repeat|reset|change|recovery|otp|totp|code|pin)\b/.test(text) &&
      !el.autocomplete.split(/\s+/).some(t => ['new-password','one-time-code'].includes(t));
  };
  const controls = () => Array.from(doc.querySelectorAll('input')).filter(visible);
  if (controls().some(el => el.type === 'password')) return refuse('standard_password_present');
  const candidates = controls().filter(semantics);
  if (candidates.length !== 1) return refuse('ambiguous_or_unsupported_widget');
  const el = candidates[0], form = el.form;
  if (!form || !form.isConnected || form.ownerDocument !== doc || el.value !== '')
    return refuse('form_or_nonempty_widget');
  const probeLimit = el.maxLength >= 0 ? Math.min(13, el.maxLength) : 13;
  if (probeLimit < 4) return refuse('unsupported_field_limit');
  const fingerprint = () => JSON.stringify([
    el.type, el.name, el.id, el.autocomplete, el.maxLength, label(el),
    form.id, form.getAttribute('action'), form.action, form.method, form.target,
    el.getAttribute('form'),
    getComputedStyle(el).webkitTextSecurity,
  ]);
  const initial = fingerprint();
  const valid = () => location.origin === origin && location.href === href && doc === document &&
    document.title === title && el.form === form && form.isConnected &&
    visible(el) && semantics(el) && fingerprint() === initial &&
    controls().filter(semantics).length === 1;
  const hit = () => {
    const r = el.getBoundingClientRect();
    return doc.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2) === el;
  };
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set;
  const input = value => {
    setter.call(el, value);
    el.dispatchEvent(new InputEvent('input', {bubbles:true, inputType:value ? 'insertText' : 'deleteContentBackward'}));
    el.dispatchEvent(new Event('change', {bubbles:true}));
  };
  const settled = () => new Promise(resolve => {
    const timer = setTimeout(() => resolve(false), 1000);
    requestAnimationFrame(() => requestAnimationFrame(() => { clearTimeout(timer); resolve(true); }));
  });
  const masked = length => {
    const value = el.value;
    return value.length === length && /^[\u2022\u25cf]+$/.test(value) &&
      Array.from(value).every(c => c === value[0]);
  };
  el.focus();
  if (!valid() || !hit() || doc.activeElement !== el || el.value !== '') return refuse('focus_changed');
  // Two different lengths also detect widgets whose apparent clear leaves an
  // accumulating backing value. No real credential has been resolved yet.
  for (const probe of ['vault-probe-A7'.slice(0, probeLimit), 'probe-B9'.slice(0, Math.min(8, probeLimit - 1))]) {
    if (Date.now() >= deadline || !valid() || doc.activeElement !== el || el.value !== '') return refuse('probe_context_changed');
    let okay = false;
    try {
      input(probe);
      okay = await settled() && valid() && doc.activeElement === el && masked(probe.length);
    } finally {
      // Clear only the retained control and our own observed probe material.
      // Context changes must not leave text behind or retarget cleanup.
      if (el.ownerDocument === doc && location.origin === origin &&
          (el.value === probe || masked(probe.length))) {
        try { input(''); } catch (_) { /* report failed cleanup below */ }
      }
    }
    const cleared = await settled();
    if (!valid() || el.value !== '') return refuse('cleanup_failed');
    if (!okay || !cleared) return refuse('not_uniformly_masked');
  }
  const presentation = JSON.stringify([el.getAttribute('class'), el.getAttribute('style')]);
  if (Date.now() >= deadline) return refuse('probe_expired');
  let used = false, invalidated = false;
  const observer = new MutationObserver(records => { if (records.length) invalidated = true; });
  observer.observe(doc, {subtree:true, childList:true, attributes:true,
    attributeFilter:['type','name','id','autocomplete','form','readonly','disabled','hidden',
      'style','class','aria-label','aria-labelledby','placeholder','title','action','method','target','href','maxlength','minlength']});
  const expiry = setTimeout(() => {used = true; observer.disconnect();}, Math.max(0, deadline - Date.now()));
  const finish = () => {used = true; clearTimeout(expiry); observer.disconnect();};
  const ready = () => {
    if (observer.takeRecords().length) invalidated = true;
    return Date.now() < deadline && !used && !invalidated && valid() && hit() && el.value === '' &&
      JSON.stringify([el.getAttribute('class'), el.getAttribute('style')]) === presentation;
  };
  return Object.freeze({
    check() { return {ready:ready()}; },
    fill(value) {
      if (!ready()) { finish(); return {filled:0, refused:'widget_changed'}; }
      if (el.maxLength >= 0 && value.length > el.maxLength) {
        finish(); return {filled:0, refused:'password_exceeds_field_limit'};
      }
      el.focus();
      // Focus handlers run synchronously and can replace the field or change
      // its owner. Drain records and revalidate before the private write.
      if (!ready() || doc.activeElement !== el) {
        finish(); return {filled:0, refused:'focus_changed'};
      }
      finish();
      try {
        input(value);
        if (valid() && masked(value.length)) return {filled:1};
      } catch (_) { /* never return a site exception containing the value */ }
      // Clear our own retained field before an uncertain result can paint.
      // This does not claim rollback of the site's internal application state.
      if (el.ownerDocument === doc && location.origin === origin &&
          (el.value === value || masked(value.length))) {
        try { input(''); } catch (_) { /* no credential retry */ }
      }
      return {filled:0, refused:'delivery_uncertain'};
    },
    dispose() { finish(); return {disposed:true}; },
  });
})()"""
