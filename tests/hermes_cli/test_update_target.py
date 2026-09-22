"""Behavioral tests for strict pinned update intent parsing."""

import argparse
from itertools import combinations
from unittest.mock import Mock

import pytest

from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_target import TargetRequest, validate_target_request


REVISION = "a" * 40
INSTALL_ID = "b" * 32
CURRENT_SHA = "c" * 40


def _parser(collaborator=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=collaborator or Mock())
    return parser


def test_absent_target_intent_is_legacy_none():
    assert validate_target_request(None, None, None) is None


@pytest.mark.parametrize(
    "fields",
    [
        *combinations(("revision", "install_id", "current_sha"), 1),
        *combinations(("revision", "install_id", "current_sha"), 2),
    ],
)
def test_every_nonempty_proper_subset_is_rejected(fields):
    values = {"revision": REVISION, "install_id": INSTALL_ID, "current_sha": CURRENT_SHA}
    supplied = {name: values[name] for name in fields}
    with pytest.raises(ValueError, match="^incomplete-target-intent$"):
        validate_target_request(
            supplied.get("revision"), supplied.get("install_id"), supplied.get("current_sha")
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (field, value, f"invalid-{field}")
        for field, good, short in (
            ("revision", REVISION, "a" * 39),
            ("install_id", INSTALL_ID, "b" * 31),
            ("current_sha", CURRENT_SHA, "c" * 39),
        )
        for value in ("main", short, good.upper(), " " + good, good + "\n")
    ],
)
def test_invalid_field_matrix_rejects_strict_values(field, value, error):
    values = {"revision": REVISION, "install_id": INSTALL_ID, "current_sha": CURRENT_SHA}
    values[field] = value
    with pytest.raises(ValueError, match=f"^{error}$"):
        validate_target_request(**values)


def test_complete_intent_is_frozen_and_preserved_field_for_field():
    request = validate_target_request(REVISION, INSTALL_ID, CURRENT_SHA)
    assert request == TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA)
    assert (request.revision, request.install_id, request.current_sha) == (
        REVISION,
        INSTALL_ID,
        CURRENT_SHA,
    )
    with pytest.raises(AttributeError):
        request.revision = "d" * 40


def test_actual_update_parser_exposes_the_three_target_flags():
    args = _parser().parse_args(
        [
            "update",
            "--revision",
            REVISION,
            "--install-id",
            INSTALL_ID,
            "--current-sha",
            CURRENT_SHA,
        ]
    )
    assert args.target_request == TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA)


@pytest.mark.parametrize("option", ["--revision", "--install-id", "--current-sha"])
def test_parser_rejects_incomplete_intent_before_collaborator(option):
    collaborator = Mock()
    with pytest.raises(ValueError, match="^incomplete-target-intent$"):
        _parser(collaborator).parse_args(["update", option, REVISION])
    collaborator.assert_not_called()


def test_parser_preserves_legacy_absence_and_does_not_invoke_collaborator():
    collaborator = Mock()
    args = _parser(collaborator).parse_args(["update", "--plan"])
    assert args.target_request is None
    collaborator.assert_not_called()


def test_parser_validates_complete_check_intent_without_invoking_collaborator():
    collaborator = Mock()
    args = _parser(collaborator).parse_args(
        [
            "update",
            "--check",
            "--revision",
            REVISION,
            "--install-id",
            INSTALL_ID,
            "--current-sha",
            CURRENT_SHA,
        ]
    )
    assert args.check is True
    assert args.target_request == TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA)
    collaborator.assert_not_called()
