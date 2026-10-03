"""Regression for #121361: static front page over XML-RPC must not fail silently.

On hardened shared hosts (cPanel/Softaculous) `wp.setOptions` returns an empty
`[]` while discarding the write, so an agent that trusts the return value
reports "static front page set" over a blog index. The skill must send the
full `show_on_front` / `page_on_front` combo, read the options back, and
treat an empty or unchanged read-back as a failure.

No network: both hosts below are canned XML-RPC conversations.
"""

from __future__ import annotations

import json
import sys
import xmlrpc.client
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

    def __init__(self, initial=None, pages=(7, 9)):
        self.options = dict(initial or {})
        self.pages = set(pages)
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

    def getPage(self, blog_id, user, password, page_id):
        if page_id not in self.pages:
            raise xmlrpc.client.Fault(
                404, "Sorry, the post you are trying to access does not exist."
            )
        return {"page_id": str(page_id), "post_title": "Home", "post_type": "page"}


class FilteredHost:
    """The #121361 host: setOptions returns [] and drops the write; the
    filtered keys never come back from getOptions either."""

    def __init__(self, pages=(7, 9)):
        self.pages = set(pages)
        self.writes = []

    def setOptions(self, blog_id, user, password, options):
        self.writes.append(dict(options))
        return []

    def getOptions(self, blog_id, user, password, names=None):
        return {}

    def getPage(self, blog_id, user, password, page_id):
        return {"page_id": str(page_id), "post_title": "Home"}


class PartialHost(ApplyingHost):
    """Accepts show_on_front and silently drops page_on_front -- the host state
    that used to be left behind when the combo failed."""

    def setOptions(self, blog_id, user, password, options):
        kept = {k: v for k, v in options.items() if k == "show_on_front"}
        return ApplyingHost.setOptions(self, blog_id, user, password, kept)


class LyingHost:
    """Self-certifying write: setOptions echoes the requested combo while the
    site keeps the old state. Trusting that echo is the #121361 failure mode;
    the read-back has to be an independent call."""

    def __init__(self):
        self.options = {"show_on_front": "posts", "page_on_front": "0"}
        self.reads = 0

    def setOptions(self, blog_id, user, password, options):
        return {k: {"value": v} for k, v in options.items()}

    def getOptions(self, blog_id, user, password, names=None):
        self.reads += 1
        names = list(names) if names is not None else list(self.options)
        return {n: {"value": self.options[n]} for n in names if n in self.options}

    def getPage(self, blog_id, user, password, page_id):
        return {"page_id": str(page_id), "post_title": "Home"}


class AuthFaultHost:
    """The credentials are refused everywhere: a fault, not a filtered write."""

    def _refuse(self, *args, **kwargs):
        raise xmlrpc.client.Fault(403, "Incorrect username or password.")

    getPage = _refuse
    getOptions = _refuse


def _proxy(host):
    return SimpleNamespace(wp=host)


def _fake_server_proxy(monkeypatch, host):
    """Stand in for xmlrpc.client.ServerProxy and record the urls it was built with."""
    seen = []

    def factory(url, **kwargs):
        seen.append(url)
        return _proxy(host)

    monkeypatch.setattr(set_front_page.xmlrpc.client, "ServerProxy", factory)
    return seen


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


def test_read_back_is_its_own_round_trip_not_the_write_response():
    """A write that echoes the desired combo while the site keeps the old state
    must still fail: the data has to come from a getOptions call, not from the
    setOptions reply (#121361)."""
    host = LyingHost()
    with pytest.raises(set_front_page.FrontPageNotApplied):
        set_front_page.set_static_front_page(
            _proxy(host), 0, "user", "pw", page_id=7, posts_page_id=9
        )
    assert host.reads >= 1
    assert host.options["show_on_front"] == "posts"


def test_missing_page_id_is_refused_before_the_write():
    """The options can hold a page id that does not exist; a healthy host
    accepts it happily, so the page has to be checked before anything is
    written (the reviewer's 999999 case)."""
    host = ApplyingHost(pages=(9,))
    with pytest.raises(set_front_page.UnknownPage):
        set_front_page.set_static_front_page(
            _proxy(host), 0, "user", "pw", page_id=7, posts_page_id=9
        )
    assert host.writes == []


def test_dropped_option_is_rolled_back_to_the_pre_image():
    """A host that takes show_on_front=page and drops page_on_front used to be
    left in that half-applied state. The pre-image recorded before the write
    goes back."""
    host = PartialHost(
        initial={"show_on_front": "posts", "page_on_front": "3"}, pages=(7, 9)
    )
    with pytest.raises(set_front_page.FrontPageNotApplied) as excinfo:
        set_front_page.set_static_front_page(
            _proxy(host), 0, "user", "pw", page_id=7, posts_page_id=9
        )
    assert host.options == {"show_on_front": "posts", "page_on_front": "3"}
    assert "previous values restored" in str(excinfo.value)


def test_plain_http_url_is_refused_before_any_request(monkeypatch, capsys):
    """The application password rides in the request body; plain http:// would
    put it on the wire in cleartext, so main() refuses it locally."""
    monkeypatch.setenv("WP_PASSWORD", "app-password")
    monkeypatch.setattr(
        set_front_page.xmlrpc.client,
        "ServerProxy",
        lambda *args, **kwargs: pytest.fail("no request may be made over http"),
    )
    assert (
        set_front_page.main(
            ["--url", "http://example.com/xmlrpc.php", "--user", "u", "--page-id", "7"]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "Traceback" not in err
    payload = json.loads(err)
    assert payload["ok"] is False
    assert "https" in payload["error"]


def test_main_over_https_reports_the_verified_options(monkeypatch, capsys):
    """main() end to end: gate, proxy, write, read-back, JSON envelope."""
    monkeypatch.setenv("WP_PASSWORD", "app-password")
    host = ApplyingHost(initial={"show_on_front": "posts"}, pages=(7, 9))
    seen = _fake_server_proxy(monkeypatch, host)
    rc = set_front_page.main(
        [
            "--url",
            "https://example.com/xmlrpc.php",
            "--user",
            "u",
            "--page-id",
            "7",
            "--page-for-posts",
            "9",
        ]
    )
    assert rc == 0
    assert seen == ["https://example.com/xmlrpc.php"]
    assert host.options["show_on_front"] == "page"
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "options": {"show_on_front": "page", "page_on_front": 7, "page_for_posts": 9},
    }


def test_auth_fault_is_reported_as_json_not_a_traceback(monkeypatch, capsys):
    """A 403 fault used to escape as a raw xmlrpc.client.Fault traceback, which
    reads like a filtered write; it must come back in the same envelope."""
    monkeypatch.setenv("WP_PASSWORD", "wrong")
    _fake_server_proxy(monkeypatch, AuthFaultHost())
    assert (
        set_front_page.main(
            ["--url", "https://example.com/xmlrpc.php", "--user", "u", "--page-id", "7"]
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "Traceback" not in err
    payload = json.loads(err)
    assert payload["ok"] is False
    assert "403" in payload["error"]
    assert "Incorrect username or password." in payload["error"]


def test_password_is_read_from_the_environment_only(monkeypatch, capsys):
    """No --password flag: argv is visible to every process on the box (ps)."""
    monkeypatch.delenv("WP_PASSWORD", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        set_front_page.main(
            ["--url", "https://example.com/xmlrpc.php", "--user", "u", "--page-id", "7"]
        )
    assert excinfo.value.code == 2
    assert "WP_PASSWORD" in capsys.readouterr().err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
