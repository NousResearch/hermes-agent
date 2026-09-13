"""Tests for hermes_cli/prompt_stash.py — prompt stash helpers."""


def test_stash_name_validation_rejects_empty():
    from hermes_cli.prompt_stash import _is_valid_stash_name
    assert _is_valid_stash_name("") is False
    assert _is_valid_stash_name("   ") is False


def test_stash_name_validation_accepts_normal():
    from hermes_cli.prompt_stash import _is_valid_stash_name
    assert _is_valid_stash_name("my-stash") is True
    assert _is_valid_stash_name("pr_review_v2") is True


def test_stash_name_validation_rejects_slashes():
    from hermes_cli.prompt_stash import _is_valid_stash_name
    assert _is_valid_stash_name("a/b") is False
