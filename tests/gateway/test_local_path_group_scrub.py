"""
Regression tests for gateway.platforms.base._scrub_local_paths_for_group_delivery.

Covers a review finding on the original PR: the local-absolute-path guard
matched the "s:/" inside "https://", corrupting ordinary URLs in group
responses, and a bare [^\\s...]+ token stopped at the first space so a
username containing a space only got partially redacted.
"""

from gateway.platforms.base import _scrub_local_paths_for_group_delivery


def test_dm_chat_type_is_never_scrubbed():
    text = r"C:\Users\Jane Roe\secret.json"
    assert _scrub_local_paths_for_group_delivery(text, "dm") == text


def test_group_preserves_ordinary_urls():
    text = "normal https://example.com/a/b and /tmp/file.txt"
    assert _scrub_local_paths_for_group_delivery(text, "group") == text


def test_forum_preserves_ftp_url_with_credentials():
    text = "ftp://s:pass@host/file also fine?"
    assert _scrub_local_paths_for_group_delivery(text, "forum") == text


def test_group_redacts_windows_path_with_space_in_username():
    text = r"leaked C:\Users\Jane Roe\file.txt here"
    result = _scrub_local_paths_for_group_delivery(text, "group")
    assert "Roe" not in result
    assert "[file]" in result


def test_group_redacts_home_relative_path():
    result = _scrub_local_paths_for_group_delivery("path is ~/secret.json", "group")
    assert "secret.json" not in result


def test_group_redacts_unix_home_path():
    result = _scrub_local_paths_for_group_delivery("see /home/someuser/.env", "group")
    assert ".env" not in result


def test_group_redaction_does_not_consume_entire_message():
    text = (
        r"C:\Users\Jane Roe\file.txt and then normal sentence continues "
        "after this point with many more words following along nicely"
    )
    result = _scrub_local_paths_for_group_delivery(text, "group")
    assert "following along nicely" in result
