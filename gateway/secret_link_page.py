"""HTML for the secret-link page: one plain POST form, no script, no third-party resource.

The ``Referrer-Policy: no-referrer`` response header would make browsers send ``Origin: null``
on the form POST (Fetch spec), and the server needs the real origin to reject cross-site posts.
The page therefore sets ``<meta name="referrer" content="same-origin">``: same-origin requests
keep their Origin/Referer, and the page loads nothing cross-origin that could receive a referrer.
"""

from __future__ import annotations

from html import escape
from typing import Iterable

_STYLE = """
body{font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;margin:0;background:#f4f4f5;color:#18181b}
main{max-width:30rem;margin:0 auto;padding:1.5rem 1rem}
h1{font-size:1.4rem;margin:0 0 1rem}
label{display:block;font-weight:600;margin:1rem 0 .3rem}
input[type=text],input[type=password]{box-sizing:border-box;width:100%;padding:.7rem;font-size:1rem;
border:1px solid #71717a;border-radius:.4rem;background:#fff;color:#18181b}
fieldset{border:1px solid #71717a;border-radius:.4rem;margin:1rem 0;padding:.5rem .8rem}
legend{font-weight:600;padding:0 .3rem}
.check{display:flex;align-items:center;gap:.5rem;font-weight:400;margin:.4rem 0}
.check input{width:1.2rem;height:1.2rem}
button{margin-top:1.2rem;width:100%;padding:.8rem;font-size:1rem;font-weight:600;border:0;border-radius:.4rem;
background:#1d4ed8;color:#fff}
.note{border-left:4px solid #b91c1c;background:#fff;padding:.6rem .8rem;margin:0 0 1rem}
.ok{border-left-color:#15803d}
""".strip()


def _document(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<meta name=\"referrer\" content=\"same-origin\">"
        "<meta name=\"robots\" content=\"noindex,nofollow\">"
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main><h1>{escape(title)}</h1>{body}</main></body></html>")


def _note(text: str, *, ok: bool = False) -> str:
    return f"<p class=\"note{' ok' if ok else ''}\" role=\"status\">{escape(text)}</p>" if text else ""


def _check(name: str, value: str, label: str, checked: bool) -> str:
    return (f"<label class=\"check\"><input type=\"checkbox\" name=\"{name}\" value=\"{escape(value)}\""
            f"{' checked' if checked else ''}> {escape(label)}</label>")


def render_form(*, nonce: str, key: str, profiles: Iterable[str], selected: Iterable[str] = (),
                all_selected: bool = False, message: str = "", conflicts: Iterable[str] = ()) -> str:
    """The entry form. The secret field is always empty; a conflict adds the replace choice."""
    chosen = set(selected)
    conflicts = list(conflicts)
    parts = [_note(message)]
    if conflicts:
        parts.append(_note(f"{key} already exists in: {', '.join(conflicts)}. Select \"Replace existing value\" "
                           "or use a different key name. Type the secret value again."))
    parts.append(
        "<form method=\"post\" autocomplete=\"off\">"
        f"<input type=\"hidden\" name=\"nonce\" value=\"{escape(nonce)}\">"
        "<label for=\"key\">Key name</label>"
        f"<input type=\"text\" id=\"key\" name=\"key\" value=\"{escape(key)}\" required autocomplete=\"off\" "
        "autocapitalize=\"characters\" spellcheck=\"false\">"
        "<label for=\"secret\">Secret value</label>"
        "<input type=\"password\" id=\"secret\" name=\"secret\" value=\"\" required autocomplete=\"off\" "
        "autocapitalize=\"off\" spellcheck=\"false\">"
        "<fieldset><legend>Profiles</legend>"
        + _check("all_profiles", "1", "All profiles", all_selected)
        + "".join(_check("profile", name, name, name in chosen) for name in profiles)
        + "</fieldset>"
        + (_check("replace", "1", "Replace existing value", False) if conflicts else "")
        + "<button type=\"submit\">Save secret</button></form>")
    return _document("Add a secret", "".join(parts))


def render_message(title: str, message: str, *, ok: bool = False) -> str:
    return _document(title, _note(message, ok=ok))


def render_result(*, key: str, written: list[str], failed: list[str], not_tried: list[str],
                  restarting: list[str]) -> str:
    """The page after a use: key name and profile names only."""
    lines = [f"Written: {', '.join(written) or 'none'}."]
    if failed:
        lines.append(f"Failed: {', '.join(failed)}. Not tried: {', '.join(not_tried) or 'none'}.")
        lines.append("This link is now used. Ask Hermes for a new link to try again.")
    if restarting:
        lines.append(f"Restarting the gateway of: {', '.join(restarting)}.")
    head = f"Saved {key}." if not failed else f"{key} was not saved to all profiles."
    body = _note(head, ok=not failed) + "".join(f"<p>{escape(line)}</p>" for line in lines)
    return _document("Secret saved" if not failed else "Secret partly saved", body)


NOT_FOUND_PAGE = render_message(
    "Link not valid", "This link is not valid. It may have expired or been used. Ask Hermes for a new link in Telegram.")
FORBIDDEN_PAGE = render_message(
    "Request refused", "The request is not valid. Open the link again from the Telegram message.")
TOO_MANY_PAGE = render_message("Too many requests", "Too many requests. Wait one minute and try again.")
BAD_REQUEST_PAGE = render_message("Request refused", "The form is not valid. Open the link again and try again.")
ERROR_PAGE = render_message("Something went wrong", "The secret was not saved. Try again.")
