"""Server-rendered /login page (no React, no SPA bundle, no injected token).

Providers come from the registry; an OAuth provider renders an anchor to
``/auth/login?provider=<name>``, a ``supports_password`` provider renders a
credential form wired by :data:`_PASSWORD_FORM_SCRIPT`. Styling mirrors the
hosted settings UI so sign-in and settings look like one product.

The ``class="provider-btn"`` anchor is test-stable: the suite extracts its
href to walk the OAuth flow.
"""
from __future__ import annotations

import html
from urllib.parse import quote, urlencode

from hermes_cli.dashboard_auth import list_session_providers

# Mirrors the hosted settings tokens in ``web/src/settings/settings.css``.
# Inserted as a ``str.format`` value, so its braces stay literal.
_STYLE = """\
<style>
  :root {
    --page: #f7f7f6;
    --ink: #171717;
    --ink-2: #4d4d4b;
    --muted: #8a8a86;
    --line: #ededeb;
    --field: #e0e0dd;
    --soft: #f4f4f3;
    --err: #d93025;
    --focus: #2f6feb;
  }
  *, *::before, *::after { box-sizing: border-box; }
  html, body { margin: 0; min-height: 100%; }
  body {
    display: grid; place-items: center; min-height: 100vh; padding: 24px 18px;
    background: var(--page); color: var(--ink);
    font: 14px/1.5 "Geist", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  :focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }

  main {
    width: 100%; max-width: 380px; padding: 32px 28px 28px;
    background: #fff; border: 1px solid var(--line); border-radius: 12px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, .04);
  }
  main.wide { max-width: 480px; }
  .face {
    width: 34px; height: 34px; margin-bottom: 20px; border-radius: 9px;
    background: var(--ink); color: #fff; display: grid; place-items: center;
    font-weight: 600; font-size: 15px;
  }
  h1 { margin: 0 0 24px; font-size: 20px; font-weight: 600; letter-spacing: -0.02em; }

  .provider-list { display: grid; gap: 16px; }
  .provider-form { display: grid; gap: 16px; }
  .form-title { font-weight: 500; color: var(--ink-2); }
  /* A lone form needs no "Sign in with ..." heading under the page title. */
  .provider-list > .provider-form:only-child .form-title { display: none; }
  .field { display: grid; gap: 6px; }
  .field-label { font-size: 13px; color: var(--ink-2); }
  .field-input {
    width: 100%; height: 38px; padding: 0 12px; outline: none;
    background: #fff; border: 1px solid var(--field); border-radius: 8px;
    font: inherit; color: inherit; box-shadow: 0 1px 1px rgba(0, 0, 0, .02);
    transition: border-color .12s, box-shadow .12s;
  }
  .field-input:focus { border-color: #a3a3a0; box-shadow: 0 0 0 3px #f0f0ee; }
  .form-error { color: var(--err); font-size: 13px; }

  .provider-btn {
    display: flex; align-items: center; justify-content: center; width: 100%; height: 38px; padding: 0 12px;
    background: var(--ink); color: #fff; border: 1px solid var(--ink); border-radius: 8px;
    font: inherit; font-size: 13px; font-weight: 500; text-decoration: none; cursor: pointer;
    box-shadow: 0 1px 1px rgba(0, 0, 0, .03);
  }
  .provider-btn:hover { background: #333; }
  .provider-btn:disabled { background: #ececea; border-color: #ececea; color: #a8a8a4; cursor: default; box-shadow: none; }
  .provider-form .provider-btn { margin-top: 4px; }

  p { margin: 0 0 12px; color: var(--ink-2); }
  p:last-child { margin-bottom: 0; }
  a { color: inherit; text-underline-offset: 2px; }
  code {
    padding: 1px 5px; border-radius: 5px; background: var(--soft);
    font-family: "Geist Mono", ui-monospace, Menlo, monospace; font-size: 12.5px;
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>"""

# Single curly braces are ``str.format`` placeholders.
_LOGIN_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in — Hermes</title>
{style}
</head>
<body>
<main>
  <div class="face">H</div>
  <h1>Sign in to Hermes</h1>
  <div class="provider-list">
{provider_buttons}
  </div>
</main>
{password_script}
</body>
</html>
"""

_EMPTY_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign-in unavailable — Hermes</title>
{style}
</head>
<body>
<main class="wide">
<div class="face">H</div>
<h1>Sign-in unavailable</h1>
<p>This dashboard is bound to a non-loopback host but no authentication
providers are available.</p>
<p>Configure the bundled username/password provider or an OAuth provider.
See the <a href="https://hermes-agent.nousresearch.com/docs/user-guide/features/web-dashboard#authentication-gated-mode">dashboard
authentication documentation</a> for setup instructions.</p>
<p>For auth-free local use, bind to <code>127.0.0.1</code> and connect through
an SSH tunnel or Tailscale.</p>
</main>
</body>
</html>
""".format(style=_STYLE)


# Emitted ONLY when a ``supports_password`` provider is listed, so OAuth-only
# login pages stay script-free. Plain string (not ``str.format``): braces are
# literal. One delegated submit handler covers every form; the provider name
# comes from the form's ``data-provider`` attribute.
_PASSWORD_FORM_SCRIPT = """\
<script>
(function () {
  function handle(form) {
    form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      var err = form.querySelector('.form-error');
      var btn = form.querySelector('button[type=submit]');
      if (err) { err.hidden = true; err.textContent = ''; }
      if (btn) { btn.disabled = true; }
      var body = {
        provider: form.getAttribute('data-provider') || '',
        username: (form.querySelector('input[name=username]') || {}).value || '',
        password: (form.querySelector('input[name=password]') || {}).value || '',
        next: (form.querySelector('input[name=next]') || {}).value || ''
      };
      fetch('/auth/password-login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        credentials: 'same-origin'
      }).then(function (resp) {
        if (resp.ok) {
          return resp.json().then(function (data) {
            window.location.assign((data && data.next) || '/');
          });
        }
        var msg = resp.status === 429
          ? 'Too many attempts. Please wait and try again.'
          : (resp.status === 401 ? 'Invalid username or password.'
                                 : 'Sign-in failed. Please try again.');
        if (err) { err.textContent = msg; err.hidden = false; }
        if (btn) { btn.disabled = false; }
      }).catch(function () {
        if (err) { err.textContent = 'Network error. Please try again.'; err.hidden = false; }
        if (btn) { btn.disabled = false; }
      });
    });
  }
  var forms = document.querySelectorAll('form.provider-form');
  for (var i = 0; i < forms.length; i++) { handle(forms[i]); }
})();
</script>
"""


def render_login_html(*, next_path: str = "") -> str:
    """Return the full HTML for ``GET /login``.

    ``next_path`` is threaded into each provider button/form so the OAuth round
    trip carries it end-to-end. The caller validates it same-origin; it is
    HTML-escaped here as defence in depth.
    """
    providers = list_session_providers()
    if not providers:
        return _EMPTY_HTML
    # URL-encode then HTML-escape, matching the gate's ``_safe_next_target``
    # shape so a round-tripped value is byte-identical.
    next_qs = f"&next={html.escape(quote(next_path, safe=''), quote=True)}" if next_path else ""
    buttons = [
        _render_password_form(p, next_path) if getattr(p, "supports_password", False) else
        f'      <a class="provider-btn" '
        f'href="/auth/login?provider={html.escape(p.name, quote=True)}{next_qs}">'
        f'Sign in with {html.escape(p.display_name)}</a>'
        for p in providers
    ]
    needs_password_script = any(getattr(p, "supports_password", False) for p in providers)
    return _LOGIN_HTML_TEMPLATE.format(
        style=_STYLE,
        provider_buttons="\n".join(buttons),
        password_script=_PASSWORD_FORM_SCRIPT if needs_password_script else "",
    )


def render_native_provider_choice_html(
        *, providers, authorize_path: str, code_challenge: str,
        code_challenge_method: str, redirect_uri: str, state: str) -> str:
    """Provider picker for a native authorize request with more than one interactive provider.

    Every link re-enters ``/auth/native/authorize`` with the SAME desktop PKCE inputs plus an
    explicit ``provider``, so the choice never leaves the validated native flow.
    """
    common = {"code_challenge": code_challenge, "code_challenge_method": code_challenge_method,
              "redirect_uri": redirect_uri, "state": state}
    buttons = []
    for p in providers:
        href = html.escape(f"{authorize_path}?{urlencode({**common, 'provider': p.name})}",
                           quote=True)
        buttons.append(f'      <a class="provider-btn" href="{href}">'
                       f'Sign in with {html.escape(p.display_name)}</a>')
    if not buttons:
        return _EMPTY_HTML
    return _LOGIN_HTML_TEMPLATE.format(
        style=_STYLE, provider_buttons="\n".join(buttons), password_script="")


def _render_password_form(provider, next_path: str) -> str:
    """Username/password form for a ``supports_password`` provider.

    ``next_path`` rides in a hidden field (already validated by the caller,
    HTML-escaped here). The provider name is a ``data-`` attribute so the
    script does not depend on field ordering.
    """
    pname = html.escape(provider.name, quote=True)
    plabel = html.escape(provider.display_name)
    safe_next = html.escape(next_path, quote=True) if next_path else ""
    return (
        f'      <form class="provider-form" data-provider="{pname}" '
        f'autocomplete="on">\n'
        f'        <div class="form-title">Sign in with {plabel}</div>\n'
        f'        <input type="hidden" name="next" value="{safe_next}">\n'
        f'        <label class="field">\n'
        f'          <span class="field-label">Username</span>\n'
        f'          <input class="field-input" type="text" name="username" '
        f'autocomplete="username" autocapitalize="none" '
        f'autocorrect="off" spellcheck="false" required>\n'
        f'        </label>\n'
        f'        <label class="field">\n'
        f'          <span class="field-label">Password</span>\n'
        f'          <input class="field-input" type="password" name="password" '
        f'autocomplete="current-password" required>\n'
        f'        </label>\n'
        f'        <div class="form-error" role="alert" hidden></div>\n'
        f'        <button class="provider-btn" type="submit">Sign in</button>\n'
        f'      </form>'
    )
