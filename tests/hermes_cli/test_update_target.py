"""Behavioral tests for strict pinned update intent parsing."""

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_target import TargetRequest, validate_target_request


@pytest.fixture
def actual_cli(monkeypatch):
    """Build the production parser with an observable update collaborator."""
    import hermes_cli.main as main

    collaborator = Mock()
    monkeypatch.setattr(main, "cmd_update", collaborator)
    parser, subparsers = main._build_cli_parser()
    return parser, subparsers, collaborator


REVISION = "a" * 40
INSTALL_ID = "b" * 32
CURRENT_SHA = "c" * 40


def _parser(collaborator=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=collaborator or Mock())
    return parser


@pytest.mark.parametrize(
    ("argv", "expected", "remainder", "use_sys_argv"),
    [
        (["--chec"], {"check": True}, [], False),
        (["--bra=first", "--branch", "last"], {"branch": "last"}, [], False),
        (["--unknown=x"], {"check": False}, ["--unknown=x"], False),
        (["--", "--rev=x"], {"revision": None}, ["--", "--rev=x"], False),
        (["--check"], {"check": True}, [], True),
    ],
)
def test_update_parse_known_preserves_legacy_contract(
    actual_cli, monkeypatch, argv, expected, remainder, use_sys_argv
):
    import sys

    _, subparsers, collaborator = actual_cli
    parser = subparsers.choices["update"]
    namespace = argparse.Namespace(caller_marker="preserved")
    monkeypatch.setattr(sys, "argv", ["hermes update", *argv])
    try:
        parsed, unknown = parser.parse_known_args(
            None if use_sys_argv else argv, namespace
        )
    except SystemExit as exc:
        pytest.fail(f"legacy parse_known_args unexpectedly exited: {exc.code}")
    assert parsed is namespace
    assert parsed.caller_marker == "preserved"
    assert {key: getattr(parsed, key) for key in expected} == expected
    assert unknown == remainder
    assert parsed.target_request is None
    assert parsed.func is collaborator
    collaborator.assert_not_called()


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
        for value in (
            "", "main", short, good + "a", good.upper(), " " + good,
            good + "\n", "g" * len(good), "ａ" * len(good),
            good.encode(), 123, False, [], {},
        )
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
            "--expected-install-id",
            INSTALL_ID,
            "--expected-current-sha",
            CURRENT_SHA,
        ]
    )
    assert args.expected_install_id == INSTALL_ID
    assert args.expected_current_sha == CURRENT_SHA
    assert args.target_request == TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA)


def test_top_level_parser_accepts_exact_pinned_options(actual_cli):
    import hermes_cli.main as main

    parser, subparsers, collaborator = actual_cli
    args = main._parse_cli_args(
        parser,
        subparsers,
        [
            "update",
            "--revision=" + REVISION,
            "--expected-install-id=" + INSTALL_ID,
            "--expected-current-sha=" + CURRENT_SHA,
        ],
    )
    assert args.target_request == TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA)
    collaborator.assert_not_called()


@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize(
    ("exact_option", "abbreviated_option"),
    [
        (option, option[:length])
        for option in ("--revision", "--expected-install-id", "--expected-current-sha")
        for length in range(3, len(option))
    ],
)
def test_top_level_parser_rejects_abbreviated_pinned_options_before_handler(
    actual_cli, exact_option, abbreviated_option, equals, capsys
):
    import hermes_cli.main as main

    parser, subparsers, collaborator = actual_cli
    options = {
        "--revision": REVISION,
        "--expected-install-id": INSTALL_ID,
        "--expected-current-sha": CURRENT_SHA,
    }

    def argv(replacement):
        tokens = ["update"]
        for option, value in options.items():
            option = replacement if option == exact_option else option
            tokens.extend([f"{option}={value}"] if equals else [option, value])
        return tokens

    # The control and rejection differ in exactly one spelling, never in
    # completeness; incomplete-intent cannot masquerade as abbreviation safety.
    admitted = main._parse_cli_args(parser, subparsers, argv(exact_option))
    assert admitted.target_request == TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA)
    with pytest.raises(SystemExit) as exc:
        main._parse_cli_args(parser, subparsers, argv(abbreviated_option))
    assert exc.value.code == 2
    error = capsys.readouterr().err
    diagnostics = [f"unrecognized arguments: {abbreviated_option}"]
    if exact_option == "--revision" and abbreviated_option in {"--r", "--re"}:
        # The parent resolves these collisions before reaching the update parser.
        token = f"{abbreviated_option}={REVISION}" if equals else abbreviated_option
        diagnostics.append(
            f"ambiguous option: {token} could match --reasoning, --resume"
        )
    assert any(diagnostic in error for diagnostic in diagnostics)
    assert "incomplete-target-intent" not in error
    collaborator.assert_not_called()


def test_top_level_parser_preserves_legacy_no_intent_and_branch(actual_cli):
    import hermes_cli.main as main

    parser, subparsers, collaborator = actual_cli
    args = main._parse_cli_args(
        parser, subparsers, ["update", "--branch", "release/1.2", "--plan"]
    )
    assert args.target_request is None
    assert args.branch == "release/1.2"
    collaborator.assert_not_called()


PINNED_OPTIONS = {
    "--revision": REVISION,
    "--expected-install-id": INSTALL_ID,
    "--expected-current-sha": CURRENT_SHA,
}


def _complete_argv():
    return [f"{option}={value}" for option, value in PINNED_OPTIONS.items()]


@pytest.mark.parametrize("option", ["--install-id", "--current-sha", "--unknown"])
@pytest.mark.parametrize("equals", [False, True])
def test_parser_rejects_unknown_aliases_with_otherwise_complete_intent(
    actual_cli, option, equals, capsys
):
    import hermes_cli.main as main

    parser, subparsers, collaborator = actual_cli
    control = ["update", *_complete_argv()]
    assert main._parse_cli_args(parser, subparsers, control).target_request is not None
    unknown = [f"{option}={INSTALL_ID}"] if equals else [option, INSTALL_ID]
    with pytest.raises(SystemExit) as exc:
        main._parse_cli_args(parser, subparsers, [*control, *unknown])
    assert exc.value.code == 2
    assert f"unrecognized arguments: {option}" in capsys.readouterr().err
    collaborator.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        *combinations(PINNED_OPTIONS, 1),
        *combinations(PINNED_OPTIONS, 2),
    ],
)
@pytest.mark.parametrize("mode", ["--check", "--plan"])
def test_parser_rejects_incomplete_intent_before_collaborator(
    actual_cli, options, mode, capsys
):
    import hermes_cli.main as main

    parser, subparsers, collaborator = actual_cli
    with pytest.raises(SystemExit) as exc:
        main._parse_cli_args(parser, subparsers, [
            "update", mode, *[f"{option}={PINNED_OPTIONS[option]}" for option in options]
        ])
    assert exc.value.code == 2
    assert "incomplete-target-intent" in capsys.readouterr().err
    collaborator.assert_not_called()


@pytest.mark.parametrize(
    ("option", "value", "error"),
    [
        ("--revision", "not-a-revision", "invalid-revision"),
        ("--expected-install-id", "B" * 32, "invalid-install_id"),
        ("--expected-current-sha", "c" * 39, "invalid-current_sha"),
    ],
)
def test_top_level_parser_rejects_invalid_target_fields_before_handler(
    actual_cli, option, value, error, capsys
):
    import hermes_cli.main as main

    parser, subparsers, collaborator = actual_cli
    values = {**PINNED_OPTIONS, option: value}
    with pytest.raises(SystemExit) as exc:
        main._parse_cli_args(parser, subparsers, [
            "update", *[f"{key}={val}" for key, val in values.items()]
        ])
    assert exc.value.code == 2
    diagnostic = capsys.readouterr().err
    assert error in diagnostic
    assert "lowercase hexadecimal" in diagnostic
    assert "40/32/40" in diagnostic
    collaborator.assert_not_called()


def test_parser_preserves_legacy_absence_and_does_not_invoke_collaborator():
    collaborator = Mock()
    args = _parser(collaborator).parse_args(["update", "--plan"])
    assert args.target_request is None
    collaborator.assert_not_called()


@pytest.mark.parametrize("mode", ["--check", "--plan"])
@pytest.mark.parametrize("branch", [None, "release/1.2"])
def test_parser_validates_read_only_intent_without_invoking_collaborator(
    actual_cli, mode, branch
):
    import hermes_cli.main as main

    parser, subparsers, collaborator = actual_cli
    argv = ["update", mode, *_complete_argv()]
    if branch is not None:
        argv.extend(["--branch", branch])
    args = main._parse_cli_args(parser, subparsers, argv)
    assert getattr(args, mode[2:]) is True
    assert args.branch == branch  # T3, not parsing, owns tracking-branch admission.
    assert args.target_request == TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA)
    assert args.func is collaborator
    collaborator.assert_not_called()


def test_top_level_legacy_abbreviations_keep_handler_seam(actual_cli):
    import hermes_cli.main as main

    parser, subparsers, collaborator = actual_cli
    args = main._parse_cli_args(parser, subparsers, [
        "update", "--chec", "--bra=first", "--branch=last", "-y"
    ])
    assert args.check and args.yes
    assert args.branch == "last"
    assert args.target_request is None
    assert args.func is collaborator
    args.func(args)  # Only the injected collaborator; never the real updater.
    collaborator.assert_called_once_with(args)


@pytest.mark.parametrize("suffix", [[], ["--unknown=x"], ["--", "--rev=ignored"]])
def test_complete_intent_preserves_remainders_values_and_last_option_wins(
    actual_cli, suffix
):
    parser, _, collaborator = actual_cli
    args, remainder = parser.parse_known_args([
        "update", "--revision=" + "d" * 40, *_complete_argv(),
        "--branch=--rev", *suffix,
    ])
    assert args.target_request == TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA)
    assert args.branch == "--rev"  # A value, not an abbreviated pinned flag.
    assert remainder == suffix
    collaborator.assert_not_called()


def test_pinned_intent_dispatches_to_exact_apply_path_without_legacy_prepare(monkeypatch):
    """Pinned mode has a separate Git admission seam; legacy preparation stays untouched."""
    from types import SimpleNamespace

    from hermes_cli import update_cmd

    request = TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA)
    called = {}
    monkeypatch.setattr(
        update_cmd,
        "_cmd_pinned_update_impl",
        lambda args, gateway_mode: called.update(args=args, gateway_mode=gateway_mode),
        raising=False,
    )
    monkeypatch.setattr(
        update_cmd,
        "_resolve_update_options",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("legacy update preparation must not run for pinned intent")
        ),
    )

    args = SimpleNamespace(target_request=request, post_swap=None, gateway=False)
    update_cmd._cmd_update_impl(args, gateway_mode=False)

    assert called["args"] is args
    assert called["gateway_mode"] is False


# T3 behavioral fixtures use local Git only: no SSH, installation, or network.
def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if check and result.returncode:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


def _git_fixture(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source"
    bare = tmp_path / "origin.git"
    _git(tmp_path, "init", "--bare", str(bare))
    _git(tmp_path, "init", "-b", "main", str(source))
    _git(source, "config", "user.name", "fixture")
    _git(source, "config", "user.email", "fixture@example.test")
    (source / ".gitignore").write_text("install_id\n", encoding="utf-8")
    (source / "hermes_cli").mkdir()
    (source / "hermes_cli" / "update_rollout_protocol.json").write_text(
        json.dumps({"protocol": 1}) + "\n", encoding="utf-8"
    )
    (source / "payload.txt").write_text("A\n", encoding="utf-8")
    _git(source, "add", ".")
    _git(source, "commit", "-m", "A")
    commit_a = _git(source, "rev-parse", "HEAD").stdout.strip()
    _git(source, "remote", "add", "origin", str(bare))
    _git(source, "push", "-u", "origin", "main")
    install = tmp_path / "install"
    _git(tmp_path, "clone", str(bare), str(install))
    _git(install, "config", "user.name", "fixture")
    _git(install, "config", "user.email", "fixture@example.test")
    (install / "install_id").write_text("1" * 32 + "\n", encoding="utf-8")
    return {"source": source, "bare": bare, "install": install, "a": commit_a}


def _commit_source(fixture: dict[str, object], text: str, message: str) -> str:
    source = fixture["source"]
    assert isinstance(source, Path)
    (source / "payload.txt").write_text(text, encoding="utf-8")
    _git(source, "add", "payload.txt")
    _git(source, "commit", "-m", message)
    return _git(source, "rev-parse", "HEAD").stdout.strip()


def test_pinned_apply_lands_reviewed_b_when_origin_moves_to_c(tmp_path):
    from hermes_cli.update_target import apply_pinned_target

    fixture = _git_fixture(tmp_path)
    commit_b = _commit_source(fixture, "B\n", "B")
    source = fixture["source"]
    assert isinstance(source, Path)
    _git(source, "push", "origin", "main")
    commit_c = _commit_source(fixture, "C\n", "C")
    _git(source, "push", "origin", "main")

    install = fixture["install"]
    assert isinstance(install, Path)
    result = apply_pinned_target(
        install, TargetRequest(commit_b, "1" * 32, str(fixture["a"]))
    )

    assert result.target_sha == commit_b
    assert result.prior_sha == str(fixture["a"])
    assert result.target_sha != commit_c
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == commit_b
    assert (install / "payload.txt").read_text(encoding="utf-8") == "B\n"
    assert _git(install, "remote").stdout.strip() == "origin"


def test_pinned_apply_refuses_dirty_tree_before_movement(tmp_path):
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    source = fixture["source"]
    assert isinstance(source, Path)
    commit_b = _commit_source(fixture, "B\n", "B")
    _git(source, "push", "origin", "main")
    install = fixture["install"]
    assert isinstance(install, Path)
    (install / "payload.txt").write_text("local edit\n", encoding="utf-8")

    with pytest.raises(PinnedTargetRefused, match="dirty-checkout"):
        apply_pinned_target(install, TargetRequest(commit_b, "1" * 32, str(fixture["a"])))
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == str(fixture["a"])
    assert (install / "payload.txt").read_text(encoding="utf-8") == "local edit\n"


def test_pinned_apply_refuses_current_sha_identity_and_branch_admission(tmp_path):
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    source = fixture["source"]
    assert isinstance(source, Path)
    commit_b = _commit_source(fixture, "B\n", "B")
    _git(source, "push", "origin", "main")
    install = fixture["install"]
    assert isinstance(install, Path)

    with pytest.raises(PinnedTargetRefused, match="current-sha-mismatch"):
        apply_pinned_target(install, TargetRequest(commit_b, "1" * 32, "f" * 40))
    with pytest.raises(PinnedTargetRefused, match="branch-not-admitted"):
        apply_pinned_target(
            install, TargetRequest(commit_b, "1" * 32, str(fixture["a"])), branch="release"
        )
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == str(fixture["a"])


def test_pinned_apply_checks_protocol_before_moving_code(tmp_path):
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    source = fixture["source"]
    assert isinstance(source, Path)
    (source / "hermes_cli" / "update_rollout_protocol.json").unlink()
    _git(source, "add", "-u")
    _git(source, "commit", "-m", "pre-protocol target")
    incompatible = _git(source, "rev-parse", "HEAD").stdout.strip()
    _git(source, "push", "origin", "main")
    install = fixture["install"]
    assert isinstance(install, Path)

    with pytest.raises(PinnedTargetRefused, match="incompatible-target"):
        apply_pinned_target(
            install, TargetRequest(incompatible, "1" * 32, str(fixture["a"]))
        )
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == str(fixture["a"])
    assert (install / "hermes_cli" / "update_rollout_protocol.json").is_file()


def test_pinned_apply_refuses_target_removed_from_authorized_origin(tmp_path):
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    source = fixture["source"]
    bare = fixture["bare"]
    assert isinstance(source, Path) and isinstance(bare, Path)
    commit_b = _commit_source(fixture, "B\n", "B")
    _git(source, "push", "origin", "main")
    install = fixture["install"]
    assert isinstance(install, Path)
    # Preserve the reviewed object locally, then remove it from the only
    # authorized origin branch before the apply fetch.
    _git(install, "fetch", "origin", "main")
    # Remove B from the only authorized origin branch before the install fetches.
    _git(bare, "update-ref", "refs/heads/main", str(fixture["a"]))

    with pytest.raises(PinnedTargetRefused, match="target-not-reachable"):
        apply_pinned_target(
            install, TargetRequest(commit_b, "1" * 32, str(fixture["a"]))
        )
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == str(fixture["a"])
