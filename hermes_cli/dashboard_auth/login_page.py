"""Server-rendered /login page.

No React, no JavaScript dependency. Listed providers come from the
registry; clicking a provider sends a GET to
``/auth/login?provider=<name>``.
"""
from __future__ import annotations

import html

from hermes_cli.dashboard_auth import list_providers

# Inline minimal CSS. The dashboard's full skin lives in the React
# bundle, which we deliberately do NOT load here — the login page must
# not depend on the SPA build being present or on the injected session
# token.
_LOGIN_HTML_TEMPLATE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sign in — Hermes Agent</title>
<style>
  :root {{
    --bg: #0a0a0b;
    --fg: #e5e5e7;
    --accent: #f97316;
    --border: #27272a;
  }}
  html, body {{
    margin: 0; padding: 0; height: 100%;
    background: var(--bg); color: var(--fg);
    font: 16px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  main {{
    max-width: 28rem; margin: 10vh auto; padding: 2rem;
    border: 1px solid var(--border); border-radius: 0.75rem;
    background: rgba(255,255,255,0.02);
  }}
  h1 {{ margin: 0 0 0.5rem; font-size: 1.5rem; }}
  p  {{ margin: 0 0 1.5rem; opacity: 0.7; }}
  .provider-list {{ display: grid; gap: 0.75rem; }}
  .provider-btn {{
    display: block; width: 100%; box-sizing: border-box;
    padding: 0.875rem 1rem; text-align: center;
    background: var(--accent); color: #0a0a0b;
    font-weight: 600; font-size: 1rem;
    border-radius: 0.5rem; text-decoration: none;
    border: 0; cursor: pointer;
  }}
  .provider-btn:hover {{ filter: brightness(1.1); }}
  footer {{
    margin-top: 2rem; font-size: 0.875rem;
    opacity: 0.5; text-align: center;
  }}
</style>
</head>
<body>
<main>
  <h1>Sign in to Hermes Agent</h1>
  <p>Choose a sign-in method to continue.</p>
  <div class="provider-list">
{provider_buttons}
  </div>
  <footer>This dashboard is bound to a non-loopback host.<br>
  Sign-in is required for security.</footer>
</main>
</body>
</html>
"""

_EMPTY_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sign-in unavailable — Hermes Agent</title>
</head>
<body><main style="font-family: system-ui; max-width: 36rem; margin: 10vh auto; padding: 2rem;">
<h1>Sign-in unavailable</h1>
<p>This dashboard is bound to a non-loopback host but no authentication
providers are installed.</p>
<p>Install <code>plugins/dashboard-auth-nous</code> (default) or another
auth provider, or restart with <code>--insecure</code> to bypass the
auth gate (not recommended on untrusted networks).</p>
</main></body></html>
"""


def render_login_html(*, next_path: str = "") -> str:
    """Return the full HTML for ``GET /login``.

    ``next_path`` — when set, the post-login landing path the user
    originally requested. Threaded into each provider button's ``href``
    as a ``next=`` query parameter so the OAuth round trip carries it
    end-to-end. The caller (``routes.login_page``) is responsible for
    validating ``next_path`` against the same-origin rules before we
    emit it; we still HTML-escape it as defence in depth.
    """
    providers = list_providers()
    if not providers:
        return _EMPTY_HTML

    if next_path:
        # URL-encode then HTML-escape. The URL-encode step matches the
        # gate's ``_safe_next_target`` output shape (also URL-encoded),
        # so a value that round-tripped from /login?next=... back into
        # the button href is byte-identical.
        from urllib.parse import quote
        next_qs = f"&next={html.escape(quote(next_path, safe=''), quote=True)}"
    else:
        next_qs = ""

    buttons = []
    for p in providers:
        buttons.append(
            f'    <a class="provider-btn" '
            f'href="/auth/login?provider={html.escape(p.name, quote=True)}{next_qs}">'
            f'Sign in with {html.escape(p.display_name)}</a>'
        )
    return _LOGIN_HTML_TEMPLATE.format(provider_buttons="\n".join(buttons))
