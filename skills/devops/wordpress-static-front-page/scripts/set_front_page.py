#!/usr/bin/env python3
"""Set a WordPress static front page over XML-RPC and verify the write stuck.

Hardened shared hosts (cPanel / Softaculous installs) filter the XML-RPC
option route: wp.setOptions answers an empty success while discarding the
write, and wp.getOptions never returns the filtered keys (#121361). The
setOptions return value is therefore never trusted -- every option is read
back and any missing or unchanged value raises FrontPageNotApplied instead
of reporting success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xmlrpc.client

FRONT_PAGE_OPTIONS = ("show_on_front", "page_on_front", "page_for_posts")


class FrontPageNotApplied(RuntimeError):
    """The static front page did not survive the write (silent host filter)."""


def _read_value(entry):
    """wp.getOptions returns {name: {"value": ...}} structs; tolerate plain values."""
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def set_static_front_page(proxy, blog_id, user, password, page_id, posts_page_id=None):
    """Write the static front page combo, then read it back.

    The combo must travel together: show_on_front=page is only meaningful
    with the two page ids, and hosts that drop any member of the trio leave
    the site rendering the blog index.

    Raises FrontPageNotApplied when the read-back is missing or unchanged;
    returns the verified {option: value} mapping on success.
    """
    desired = {"show_on_front": "page", "page_on_front": page_id}
    if posts_page_id is not None:
        desired["page_for_posts"] = posts_page_id
    # Return value deliberately untrusted: on filtered hosts it is [] either way.
    proxy.wp.setOptions(blog_id, user, password, desired)
    read_back = proxy.wp.getOptions(blog_id, user, password, list(desired)) or {}
    problems = {}
    for name, expected in desired.items():
        if name not in read_back:
            problems[name] = "missing from read-back"
            continue
        actual = _read_value(read_back[name])
        if str(actual) != str(expected):
            problems[name] = "expected {0!r}, read back {1!r}".format(expected, actual)
    if problems:
        detail = "; ".join("{0}: {1}".format(k, v) for k, v in problems.items())
        raise FrontPageNotApplied(
            "static front page not applied over XML-RPC -- {0}. On hardened "
            "shared hosts the XML-RPC option route is filtered; fall back to "
            "the mu-plugin documented in this skill and verify by fetching "
            "the homepage.".format(detail)
        )
    return dict(desired)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Set a WordPress static front page over XML-RPC with read-back verification"
    )
    parser.add_argument("--url", required=True, help="https://<site>/xmlrpc.php")
    parser.add_argument("--user", required=True, help="WordPress username")
    parser.add_argument(
        "--password",
        default=os.environ.get("WP_PASSWORD", ""),
        help="WordPress password (defaults to the WP_PASSWORD env var)",
    )
    parser.add_argument("--page-id", required=True, type=int, help="page id to show as the front page")
    parser.add_argument("--page-for-posts", type=int, default=None, help="page id that renders the posts index")
    parser.add_argument("--blog-id", type=int, default=0, help="0 for single-site installs")
    args = parser.parse_args(argv)
    if not args.password:
        parser.error("set --password or the WP_PASSWORD environment variable")
    proxy = xmlrpc.client.ServerProxy(args.url, allow_none=True)
    try:
        applied = set_static_front_page(
            proxy, args.blog_id, args.user, args.password, args.page_id, args.page_for_posts
        )
    except FrontPageNotApplied as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "options": applied}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
