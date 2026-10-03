#!/usr/bin/env python3
"""Set a WordPress static front page over XML-RPC and verify the write stuck.

Hardened shared hosts (cPanel / Softaculous installs) filter the XML-RPC
option route: wp.setOptions answers an empty success while discarding the
write, and wp.getOptions never returns the filtered keys (#121361). The
setOptions return value is therefore never trusted -- every option is read
back and any missing or unchanged value raises FrontPageNotApplied instead
of reporting success.

Three further rules, all of them about the write being destructive:

- the page ids are checked with wp.getPage before anything is written, so a
  stale id cannot leave the site pointed at a page that does not exist;
- the current values are read *before* the write and put back when the combo
  does not verify, so a host that accepts part of the trio cannot leave the
  site in a half-applied state;
- plain http:// is refused, because the application password travels in the
  XML-RPC request body.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xmlrpc.client
from urllib.parse import urlsplit

FRONT_PAGE_OPTIONS = ("show_on_front", "page_on_front", "page_for_posts")


class FrontPageError(RuntimeError):
    """Base class for the failures reported as {"ok": false}."""


class FrontPageNotApplied(FrontPageError):
    """The static front page did not survive the write (silent host filter)."""


class UnknownPage(FrontPageError):
    """The page id the combo points at does not exist; nothing was written."""


def _describe(exc):
    """Faults carry the useful WordPress message; keep it, drop the repr noise."""
    if isinstance(exc, xmlrpc.client.Fault):
        return "XML-RPC fault {0}: {1}".format(exc.faultCode, exc.faultString)
    return "{0}: {1}".format(type(exc).__name__, exc)


def _require_https(url):
    if urlsplit(url).scheme != "https":
        raise FrontPageError(
            "--url must be https:// (got {0!r}): the WordPress application "
            "password travels in the XML-RPC request body, so a plain-http "
            "endpoint would receive it in cleartext".format(url)
        )


def _read_value(entry):
    """wp.getOptions returns {name: {"value": ...}} structs; tolerate plain values."""
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def _read_options(proxy, blog_id, user, password, names):
    """Read only the named options; keys the host filters are simply absent."""
    read_back = proxy.wp.getOptions(blog_id, user, password, list(names)) or {}
    return {name: _read_value(read_back[name]) for name in names if name in read_back}


def _require_page(proxy, blog_id, user, password, page_id):
    """A combo pointing at a missing page is no front page at all -- check first."""
    try:
        page = proxy.wp.getPage(blog_id, user, password, page_id)
    except xmlrpc.client.Fault as exc:
        if exc.faultCode != 404:
            raise  # auth / permission / transport: report it, do not call it "missing"
        page = None
    if not page:
        raise UnknownPage(
            "page id {0} does not exist; nothing was written. Create the page "
            "first (wp.newPost with post_type=page) and pass its id.".format(page_id)
        )


def _restore(proxy, blog_id, user, password, previous):
    """Put the recorded pre-image back and report what actually happened."""
    if not previous:
        return "no pre-image was readable, so nothing could be restored"
    try:
        proxy.wp.setOptions(blog_id, user, password, dict(previous))
        now = _read_options(proxy, blog_id, user, password, list(previous))
    except (xmlrpc.client.Error, OSError) as exc:
        return "restore failed: {0}".format(_describe(exc))
    drifted = sorted(k for k, v in previous.items() if str(now.get(k)) != str(v))
    if drifted:
        return "restore could not be confirmed for {0}".format(", ".join(drifted))
    return "previous values restored ({0})".format(", ".join(sorted(previous)))


def set_static_front_page(proxy, blog_id, user, password, page_id, posts_page_id=None):
    """Write the static front page combo, then read it back.

    The combo must travel together: show_on_front=page is only meaningful
    with the two page ids, and hosts that drop any member of the trio leave
    the site rendering the blog index.

    Raises UnknownPage when a page id does not exist (before writing) and
    FrontPageNotApplied when the read-back is missing or unchanged, in which
    case the recorded pre-image has been put back first.

    Returns the verified {option: value} mapping on success.
    """
    desired = {"show_on_front": "page", "page_on_front": page_id}
    if posts_page_id is not None:
        desired["page_for_posts"] = posts_page_id
    for candidate in (page_id, posts_page_id):
        if candidate is not None:
            _require_page(proxy, blog_id, user, password, candidate)
    # Pre-image, recorded before the write: the option write keeps no history of
    # its own, so this is the only way back if the host keeps part of the combo.
    previous = _read_options(proxy, blog_id, user, password, list(desired))
    try:
        # Return value deliberately untrusted: on filtered hosts it is [] either way.
        proxy.wp.setOptions(blog_id, user, password, desired)
        read_back = _read_options(proxy, blog_id, user, password, list(desired))
        problems = {}
        for name, expected in desired.items():
            if name not in read_back:
                problems[name] = "missing from read-back"
                continue
            actual = read_back[name]
            if str(actual) != str(expected):
                problems[name] = "expected {0!r}, read back {1!r}".format(expected, actual)
        if not problems:
            return dict(desired)
        detail = "; ".join("{0}: {1}".format(k, v) for k, v in problems.items())
    except (xmlrpc.client.Error, OSError) as exc:
        detail = "read-back failed: {0}".format(_describe(exc))
    undone = _restore(proxy, blog_id, user, password, previous)
    raise FrontPageNotApplied(
        "static front page not applied over XML-RPC -- {0}. {1}. On hardened "
        "shared hosts the XML-RPC option route is filtered; fall back to the "
        "mu-plugin documented in this skill and verify by fetching the "
        "homepage.".format(detail, undone)
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Set a WordPress static front page over XML-RPC with read-back verification"
    )
    parser.add_argument("--url", required=True, help="https://<site>/xmlrpc.php")
    parser.add_argument("--user", required=True, help="WordPress username")
    parser.add_argument("--page-id", required=True, type=int, help="page id to show as the front page")
    parser.add_argument("--page-for-posts", type=int, default=None, help="page id that renders the posts index")
    parser.add_argument("--blog-id", type=int, default=0, help="0 for single-site installs")
    args = parser.parse_args(argv)
    # Environment only: a --password flag would put the application password in
    # argv, visible to every process on the box through ps.
    password = os.environ.get("WP_PASSWORD", "")
    if not password:
        parser.error("set the WP_PASSWORD environment variable (argv is world-readable)")
    try:
        _require_https(args.url)
        proxy = xmlrpc.client.ServerProxy(args.url, allow_none=True)
        applied = set_static_front_page(
            proxy, args.blog_id, args.user, password, args.page_id, args.page_for_posts
        )
    except FrontPageError as exc:
        error = str(exc)
    except (xmlrpc.client.Error, OSError) as exc:
        error = _describe(exc)
    else:
        print(json.dumps({"ok": True, "options": applied}))
        return 0
    print(json.dumps({"ok": False, "error": error}), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
