"""Behavioral tests for strict pinned update intent parsing."""

import argparse
import base64
import json
import subprocess
from itertools import combinations
from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli.subcommands.update import build_update_parser
from hermes_cli.update_target import (
    TargetRequest, parse_reviewed_source, validate_target_request,
)


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


@pytest.fixture(autouse=True)
def _authoritative_install_identity(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "install_id").write_text("1" * 32 + "\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))


def _parser(collaborator=None):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=collaborator or Mock())
    return parser


def _reviewed_source(*, remote="https://example.test/hermes.git", target=REVISION):
    body = {
        "repositoryRoot": "C:/reviewed/hermes-agent",
        "originUrl": remote,
        "resolvedRef": "refs/remotes/origin/main",
        "targetSha": target,
        "assuranceProfile": "fixture-profile",
        "assuranceEvidenceSha256": "d" * 64,
        "assuranceGeneration": 7,
    }
    return base64.urlsafe_b64encode(json.dumps(body).encode()).decode().rstrip("=")


def _source():
    return parse_reviewed_source(_reviewed_source())


def _request():
    return TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA, _source())


def test_pinned_parser_requires_reviewed_source_binding():
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args([
            "update", "--revision", REVISION,
            "--expected-install-id", INSTALL_ID,
            "--expected-current-sha", CURRENT_SHA,
        ])
    assert exc.value.code == 2


def test_pinned_parser_carries_complete_reviewed_source_binding():
    args = _parser().parse_args([
        "update", "--revision", REVISION,
        "--expected-install-id", INSTALL_ID,
        "--expected-current-sha", CURRENT_SHA,
        "--reviewed-source", _reviewed_source(),
    ])
    assert args.target_request.source.origin_url == "https://example.test/hermes.git"
    assert args.target_request.source.target_sha == REVISION
    assert args.target_request.source.assurance_generation == 7


def test_reviewed_source_cannot_be_used_as_a_legacy_update_switch():
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["update", "--reviewed-source", _reviewed_source()])
    assert exc.value.code == 2


def test_reviewed_source_rejects_embedded_remote_credentials():
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args([
            "update", "--revision", REVISION,
            "--expected-install-id", INSTALL_ID,
            "--expected-current-sha", CURRENT_SHA,
            "--reviewed-source", _reviewed_source(
                remote="https://secret@example.test/hermes.git?token=secret",
            ),
        ])
    assert exc.value.code == 2


def test_update_intent_preserves_reviewed_source_binding():
    from hermes_cli.update_target import build_update_intent, parse_reviewed_source

    source = parse_reviewed_source(_reviewed_source())
    request = TargetRequest(REVISION, INSTALL_ID, CURRENT_SHA, source)
    intent = build_update_intent(request, "operation-1", "main")
    assert intent["source"] == source.to_wire()


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


def test_actual_update_parser_exposes_complete_pinned_identity():
    args = _parser().parse_args(
        [
            "update",
            "--revision",
            REVISION,
            "--expected-install-id",
            INSTALL_ID,
            "--expected-current-sha",
            CURRENT_SHA,
            "--reviewed-source",
            _reviewed_source(),
        ]
    )
    assert args.expected_install_id == INSTALL_ID
    assert args.expected_current_sha == CURRENT_SHA
    assert args.target_request == _request()


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
            "--reviewed-source=" + _reviewed_source(),
        ],
    )
    assert args.target_request == _request()
    collaborator.assert_not_called()


@pytest.mark.parametrize("equals", [False, True])
@pytest.mark.parametrize(
    ("exact_option", "abbreviated_option"),
    [
        (option, option[:length])
        for option in ("--revision", "--expected-install-id", "--expected-current-sha", "--reviewed-source")
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
        "--reviewed-source": _reviewed_source(),
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
    assert admitted.target_request == _request()
    with pytest.raises(SystemExit) as exc:
        main._parse_cli_args(parser, subparsers, argv(abbreviated_option))
    assert exc.value.code == 2
    error = capsys.readouterr().err
    diagnostics = [f"unrecognized arguments: {abbreviated_option}"]
    if abbreviated_option in {"--r", "--re"}:
        # The parent resolves these collisions before reaching the update parser.
        token = f"{abbreviated_option}={options[exact_option]}" if equals else abbreviated_option
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
    "--reviewed-source": _reviewed_source(),
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
    assert args.target_request == _request()
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
    assert args.target_request == _request()
    assert args.branch == "--rev"  # A value, not an abbreviated pinned flag.
    assert remainder == suffix
    collaborator.assert_not_called()


def test_pinned_intent_dispatches_to_exact_apply_path_without_legacy_prepare(monkeypatch):
    """Pinned mode has a separate Git admission seam; legacy preparation stays untouched."""
    from types import SimpleNamespace

    from hermes_cli import update_cmd

    request = _request()
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


def test_pinned_command_refuses_when_shared_update_lock_is_held(monkeypatch):
    """Pinned updates use the same command-boundary lock as legacy updates."""
    from types import SimpleNamespace

    import hermes_cli.main as main
    import hermes_cli.update_lock as update_lock
    import hermes_cli.update_handoff as update_handoff
    from hermes_cli import update_cmd

    request = _request()
    called = []

    class HeldLock:
        holder = SimpleNamespace(pid=1234, age_seconds=2)

        def acquire(self):
            return False

        def release(self):
            raise AssertionError("a refused lock must not be released by the claimant")

    monkeypatch.setattr(main, "_update_preflight_handled", lambda _args: False)
    monkeypatch.setattr(main, "_install_hangup_protection", lambda **_kwargs: None)
    monkeypatch.setattr(main, "_finalize_update_output", lambda _state: None)
    monkeypatch.setattr(update_handoff, "wait_for_shim_parent_exit", lambda: None)
    monkeypatch.setattr(update_lock, "UpdateLock", HeldLock)
    monkeypatch.setattr(update_cmd, "_cmd_update_impl", lambda *_args, **_kwargs: called.append(True))

    with pytest.raises(SystemExit) as exc_info:
        main.cmd_update(SimpleNamespace(target_request=request, gateway=False))

    assert exc_info.value.code == update_lock.UPDATE_EXIT_CONCURRENT
    assert called == []


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
    _git(tmp_path, "clone", "-b", "main", str(bare), str(install))
    _git(install, "config", "user.name", "fixture")
    _git(install, "config", "user.email", "fixture@example.test")
    return {"source": source, "bare": bare, "install": install, "a": commit_a}


def _commit_source(fixture: dict[str, object], text: str, message: str) -> str:
    source = fixture["source"]
    assert isinstance(source, Path)
    (source / "payload.txt").write_text(text, encoding="utf-8")
    _git(source, "add", "payload.txt")
    _git(source, "commit", "-m", message)
    return _git(source, "rev-parse", "HEAD").stdout.strip()


def _git_request(fixture, target, current):
    from hermes_cli.update_target import SourceBinding

    install = fixture["install"]
    assert isinstance(install, Path)
    source = SourceBinding(
        str(install.resolve()), _git(install, "remote", "get-url", "origin").stdout.strip(),
        "refs/remotes/origin/main", target, "fixture", "d" * 64, 1,
    )
    return TargetRequest(target, "1" * 32, current, source)


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
        install, _git_request(fixture, commit_b, str(fixture["a"]))
    )

    assert result.target_sha == commit_b
    assert result.prior_sha == str(fixture["a"])
    assert result.target_sha != commit_c
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == commit_b
    assert (install / "payload.txt").read_text(encoding="utf-8") == "B\n"
    assert _git(install, "remote").stdout.strip() == "origin"


def test_pinned_apply_refuses_missing_reviewed_source_before_fetch(tmp_path):
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    install = fixture["install"]
    assert isinstance(install, Path)
    with pytest.raises(PinnedTargetRefused, match="source-binding-required"):
        apply_pinned_target(
            install, TargetRequest(str(fixture["a"]), "1" * 32, str(fixture["a"])),
        )
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == fixture["a"]


def test_pinned_apply_refuses_changed_origin_even_when_target_object_matches(tmp_path):
    from hermes_cli.update_target import SourceBinding, PinnedTargetRefused, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    source = fixture["source"]
    install = fixture["install"]
    assert isinstance(source, Path) and isinstance(install, Path)
    commit_b = _commit_source(fixture, "B\n", "B")
    _git(source, "push", "origin", "main")
    binding = SourceBinding(
        str(install.resolve()), "https://wrong.example.test/hermes.git",
        "refs/remotes/origin/main", commit_b, "fixture", "d" * 64, 7,
    )

    with pytest.raises(PinnedTargetRefused, match="reviewed-origin-mismatch"):
        apply_pinned_target(
            install, TargetRequest(commit_b, "1" * 32, str(fixture["a"]), binding),
        )
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == fixture["a"]


def test_pinned_apply_refuses_wrong_repository_even_with_same_target_object(tmp_path):
    from hermes_cli.update_target import PinnedTargetRefused, SourceBinding, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    install = fixture["install"]
    source = fixture["source"]
    assert isinstance(install, Path) and isinstance(source, Path)
    commit_b = _commit_source(fixture, "B\n", "B")
    _git(source, "push", "origin", "main")
    bound = _git_request(fixture, commit_b, str(fixture["a"]))
    wrong = SourceBinding(
        str(source.resolve()), bound.source.origin_url, bound.source.resolved_ref,
        bound.source.target_sha, bound.source.assurance_profile,
        bound.source.assurance_evidence_sha256, bound.source.assurance_generation,
    )
    with pytest.raises(PinnedTargetRefused, match="reviewed-repository-mismatch"):
        apply_pinned_target(
            install, TargetRequest(commit_b, "1" * 32, str(fixture["a"]), wrong),
        )
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == fixture["a"]


def test_pinned_apply_refuses_origin_rebound_during_fetch(tmp_path, monkeypatch):
    from hermes_cli import update_target
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    install = fixture["install"]
    source = fixture["source"]
    assert isinstance(install, Path) and isinstance(source, Path)
    commit_b = _commit_source(fixture, "B\n", "B")
    _git(source, "push", "origin", "main")
    request = _git_request(fixture, commit_b, str(fixture["a"]))
    original_run_git = update_target._run_git

    def rebind_after_fetch(root, *args, **kwargs):
        result = original_run_git(root, *args, **kwargs)
        if args[:1] == ("fetch",) and result.returncode == 0:
            _git(install, "remote", "set-url", "origin", str(source))
        return result

    monkeypatch.setattr(update_target, "_run_git", rebind_after_fetch)
    with pytest.raises(PinnedTargetRefused, match="reviewed-origin-mismatch"):
        apply_pinned_target(install, request)
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == fixture["a"]


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
        apply_pinned_target(install, _git_request(fixture, commit_b, str(fixture["a"])))
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == str(fixture["a"])
    assert (install / "payload.txt").read_text(encoding="utf-8") == "local edit\n"


def test_pinned_apply_rechecks_clean_tree_after_fetch_before_movement(tmp_path, monkeypatch):
    from hermes_cli import update_target
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    source = fixture["source"]
    install = fixture["install"]
    assert isinstance(source, Path) and isinstance(install, Path)
    commit_b = _commit_source(fixture, "B\n", "B")
    _git(source, "push", "origin", "main")

    original_run_git = update_target._run_git

    def inject_race(root, *args, **kwargs):
        result = original_run_git(root, *args, **kwargs)
        if args[:1] == ("fetch",) and result.returncode == 0:
            (install / "race.txt").write_text("concurrent edit\n", encoding="utf-8")
        return result

    monkeypatch.setattr(update_target, "_run_git", inject_race)
    with pytest.raises(PinnedTargetRefused, match="dirty-checkout"):
        apply_pinned_target(install, _git_request(fixture, commit_b, str(fixture["a"])))
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == str(fixture["a"])


def test_pinned_apply_rechecks_clean_tree_after_protocol_before_merge(tmp_path, monkeypatch):
    from hermes_cli import update_target
    from hermes_cli.update_target import PinnedTargetRefused, apply_pinned_target

    fixture = _git_fixture(tmp_path)
    source = fixture["source"]
    install = fixture["install"]
    assert isinstance(source, Path) and isinstance(install, Path)
    commit_b = _commit_source(fixture, "B\n", "B")
    _git(source, "push", "origin", "main")
    request = _git_request(fixture, commit_b, str(fixture["a"]))
    original_run_git = update_target._run_git

    def inject_after_protocol(root, *args, **kwargs):
        result = original_run_git(root, *args, **kwargs)
        if args[:2] == ("show", f"{commit_b}:hermes_cli/update_rollout_protocol.json"):
            (install / "race.txt").write_text("concurrent edit\n", encoding="utf-8")
        return result

    monkeypatch.setattr(update_target, "_run_git", inject_after_protocol)
    with pytest.raises(PinnedTargetRefused, match="dirty-checkout"):
        apply_pinned_target(install, request)
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == str(fixture["a"])
    assert (install / "race.txt").read_text(encoding="utf-8") == "concurrent edit\n"


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
        apply_pinned_target(install, _git_request(fixture, commit_b, "f" * 40))
    with pytest.raises(PinnedTargetRefused, match="branch-not-admitted"):
        apply_pinned_target(
            install, _git_request(fixture, commit_b, str(fixture["a"])), branch="release"
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
            install, _git_request(fixture, incompatible, str(fixture["a"]))
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
            install, _git_request(fixture, commit_b, str(fixture["a"]))
        )
    assert _git(install, "rev-parse", "HEAD").stdout.strip() == str(fixture["a"])
