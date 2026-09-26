"""Regression for #121361: static front page over XML-RPC must not fail silently.

On hardened shared hosts (cPanel/Softaculous) `wp.setOptions` returns an empty
`[]` while discarding the write, so an agent that trusts the return value
reports "static front page set" over a blog index. The skill must send the
full `show_on_front` / `page_on_front` combo, read the options back, and
treat an empty or unchanged read-back as a failure.

No network: both hosts below are canned XML-RPC conversations.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "devops"
    / "wordpress-static-front-page"
    / "scripts"
)
sys.path.insert(0, str(SCRIPTS_DIR))

import set_front_page  # noqa: E402


class ApplyingHost:
    """Healthy WordPress: every wp.setOptions write sticks and reads back."""

    def __init__(self, initial=None):
        self.options = dict(initial or {})
        self.writes = []

    def setOptions(self, blog_id, user, password, options):
        self.writes.append(dict(options))
        self.options.update({k: str(v) for k, v in options.items()})
        return {k: {"value": self.options[k]} for k in options}

    def getOptions(self, blog_id, user, password, names=None):
        names = list(names) if names is not None else list(self.options)
        return {
            n: {"value": self.options[n], "readonly": False, "desc": n}
            for n in names
            if n in self.options
        }


class FilteredHost:
    """The #121361 host: setOptions returns [] and drops the write; the
    filtered keys never come back from getOptions either."""

    def __init__(self):
        self.writes = []

    def setOptions(self, blog_id, user, password, options):
        self.writes.append(dict(options))
        return []

    def getOptions(self, blog_id, user, password, names=None):
        return {}


def _proxy(host):
    return SimpleNamespace(wp=host)


def test_full_front_page_combo_reaches_the_wire_and_reads_back():
    """The mock must end up with the static front page actually configured —
    show_on_front, page_on_front and page_for_posts as one combo."""
    host = ApplyingHost(initial={"show_on_front": "posts"})
    applied = set_front_page.set_static_front_page(
        _proxy(host), 0, "user", "pw", page_id=7, posts_page_id=9
    )
    assert host.writes == [
        {"show_on_front": "page", "page_on_front": 7, "page_for_posts": 9}
    ]
    assert host.options["show_on_front"] == "page"
    assert host.options["page_on_front"] == "7"  # WP reads ids back as strings
    assert applied == {
        "show_on_front": "page",
        "page_on_front": 7,
        "page_for_posts": 9,
    }


def test_empty_set_options_success_is_reported_as_failure():
    """The body's silent success: setOptions answers [], getOptions comes back
    empty. That is a discarded write, not a configured site."""
    host = FilteredHost()
    with pytest.raises(set_front_page.FrontPageNotApplied):
        set_front_page.set_static_front_page(
            _proxy(host), 0, "user", "pw", page_id=7, posts_page_id=9
        )
    # even on the failing host, the full combo had to reach the wire first
    assert host.writes == [
        {"show_on_front": "page", "page_on_front": 7, "page_for_posts": 9}
    ]


def test_unchanged_read_back_is_a_failure():
    """A write that is accepted but silently dropped for one key leaves the
    read-back unchanged; that must not pass for success either."""
    host = ApplyingHost(initial={"show_on_front": "posts"})
    healthy = host.setOptions

    def drop_show_on_front(blog_id, user, password, options):
        kept = {k: v for k, v in options.items() if k != "show_on_front"}
        return healthy(blog_id, user, password, kept)

    host.setOptions = drop_show_on_front
    with pytest.raises(set_front_page.FrontPageNotApplied):
        set_front_page.set_static_front_page(
            _proxy(host), 0, "user", "pw", page_id=7, posts_page_id=9
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
