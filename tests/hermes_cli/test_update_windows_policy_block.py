"""Windows Smart App Control / WDAC policy-block detection (issue #87789).

Smart App Control / Application-Control policies block Hermes' spawned
processes and file copies with a raw ``OSError`` — winerror 1260
(``ERROR_ACCESS_DISABLED_BY_POLICY``) or the field-observed 4551 — that
``hermes update``'s ZIP path previously surfaced verbatim as an opaque
"ZIP update failed: ..." with no indication of the actual cause. These tests
pin the detector's classification (including an exhaustive fuzz pass to rule
out false positives) and the wiring that attaches guidance to the real
failure path.
"""

from __future__ import annotations

import ast
import inspect
import random
import textwrap

import pytest

from hermes_cli import update_cmd
from hermes_cli.windows_policy_block import (
    WINERROR_ACCESS_DISABLED_BY_POLICY,
    WINERROR_SMART_APP_CONTROL_BLOCK,
    detect_policy_block,
    policy_block_guidance,
)


# ---------------------------------------------------------------------------
# Positive detection: known winerror codes
# ---------------------------------------------------------------------------


def _os_error_with_winerror(code: int, message: str = "blocked") -> OSError:
    exc = OSError(message)
    exc.winerror = code
    return exc


@pytest.mark.parametrize(
    "code", [WINERROR_ACCESS_DISABLED_BY_POLICY, WINERROR_SMART_APP_CONTROL_BLOCK]
)
def test_detects_known_winerror_codes(code):
    assert detect_policy_block(_os_error_with_winerror(code)) is True


def test_detects_winerror_via_os_error_suffix_when_attribute_absent():
    """Rust/subprocess-relayed errors carry the code only in text, e.g. the
    ``(os error 4551)`` suffix ``std::io::Error``'s ``Display`` produces —
    ``winerror`` is a Windows-only ``OSError`` attribute. On Windows CPython
    sets it to 0 by default when unset; on other platforms the attribute is
    never populated at all, so deletion must tolerate both."""
    exc = OSError(
        "An Application Control policy has blocked this file. (os error 4551)"
    )
    try:
        del exc.winerror
    except AttributeError:
        pass  # not populated on non-Windows platforms
    assert detect_policy_block(exc) is True


def test_detects_1260_suffix_variant():
    exc = Exception("spawning hermes.exe: Access is disabled by policy. (os error 1260)")
    assert detect_policy_block(exc) is True


# ---------------------------------------------------------------------------
# Positive detection: text signatures (case-insensitive, embedded anywhere)
# ---------------------------------------------------------------------------

_TEXT_SIGNATURE_CASES = [
    "An Application Control policy has blocked this file.",
    "APPLICATION CONTROL POLICY HAS BLOCKED THIS FILE",
    "This program is blocked by group policy. Contact your administrator.",
    "spawn EPERM: blocked by group policy",
    "Windows Smart App Control prevented this action.",
    "smart app control is on",
]


@pytest.mark.parametrize("text", _TEXT_SIGNATURE_CASES)
def test_detects_known_text_signatures(text):
    assert detect_policy_block(Exception(text)) is True


def test_detects_signature_in_extra_text_not_just_exception_message():
    """Subprocess failures often carry the real signal in captured
    stdout/stderr, not the Python-level exception message."""
    exc = Exception("update.ps1 exited with code 1")
    stderr = "ERROR: Smart App Control blocked venv\\Scripts\\python.exe"
    assert detect_policy_block(exc) is False
    assert detect_policy_block(exc, stderr) is True


def test_detects_through_exception_chain_cause():
    original = _os_error_with_winerror(WINERROR_SMART_APP_CONTROL_BLOCK)
    try:
        try:
            raise original
        except OSError as inner:
            raise RuntimeError("ZIP update failed") from inner
    except RuntimeError as chained:
        assert detect_policy_block(chained) is True


def test_detects_through_implicit_exception_context():
    original = _os_error_with_winerror(WINERROR_ACCESS_DISABLED_BY_POLICY)
    try:
        try:
            raise original
        except OSError:
            raise RuntimeError("wrapped without explicit chaining")
    except RuntimeError as chained:
        assert detect_policy_block(chained) is True


# ---------------------------------------------------------------------------
# Negative detection: must not false-positive on ordinary failures
# ---------------------------------------------------------------------------

_ORDINARY_FAILURES = [
    OSError(2, "No such file or directory"),
    OSError(5, "Access is denied"),
    OSError(28, "No space left on device"),
    OSError(13, "Permission denied"),
    FileNotFoundError("agent/tools.py"),
    RuntimeError("not enough free disk space to stage the update safely"),
    ConnectionError("could not reach github.com"),
    ValueError("branch name contains invalid characters"),
    Exception(""),
]


@pytest.mark.parametrize("exc", _ORDINARY_FAILURES)
def test_does_not_flag_ordinary_failures(exc):
    assert detect_policy_block(exc) is False


def test_does_not_flag_unrelated_winerror_codes():
    for code in (2, 3, 5, 13, 32, 183, 1223):  # includes ERROR_CANCELLED etc.
        assert detect_policy_block(_os_error_with_winerror(code)) is False


def test_does_not_flag_unrelated_os_error_suffix_numbers():
    for code in (2, 5, 13, 28, 32, 183):
        exc = OSError(f"some unrelated failure (os error {code})")
        assert detect_policy_block(exc) is False


# ---------------------------------------------------------------------------
# Exhaustive fuzz pass — the "no false positives across a wide surface" check
# ---------------------------------------------------------------------------

_UNRELATED_WORDS = [
    "timeout",
    "disk",
    "network",
    "permission",
    "denied",
    "not found",
    "corrupt",
    "invalid",
    "argument",
    "retry",
    "exit",
    "code",
    "failed",
    "python",
    "venv",
    "pip",
    "uv",
    "git",
    "zip",
    "hash",
    "mismatch",
    "encoding",
    "utf-8",
    "socket",
    "closed",
]


def test_fuzz_random_os_errors_never_false_positive():
    """Exhaustion check: a large randomized sweep of OSErrors with codes
    outside the two known ones, and messages built from unrelated words,
    must never be classified as a policy block. Deterministic seed keeps the
    sweep reproducible across CI runs."""
    rng = random.Random(87789)
    known = {WINERROR_ACCESS_DISABLED_BY_POLICY, WINERROR_SMART_APP_CONTROL_BLOCK}
    false_positives = []
    for _ in range(5000):
        code = rng.randint(0, 20000)
        if code in known:
            continue
        word_count = rng.randint(1, 6)
        message = " ".join(rng.choice(_UNRELATED_WORDS) for _ in range(word_count))
        message = f"{message} (os error {code})"
        exc = _os_error_with_winerror(code, message)
        if detect_policy_block(exc):
            false_positives.append((code, message))
    assert not false_positives, f"false positives: {false_positives[:10]}"


def test_fuzz_signature_variants_always_detected():
    """Complement of the false-positive sweep: known signatures must survive
    arbitrary surrounding noise and casing."""
    rng = random.Random(4551)
    signatures = [
        "Application Control policy has blocked this file",
        "blocked by group policy",
        "Smart App Control",
    ]
    misses = []
    for _ in range(2000):
        sig = rng.choice(signatures)
        if rng.random() < 0.5:
            sig = sig.upper()
        elif rng.random() < 0.5:
            sig = sig.lower()
        prefix = " ".join(rng.choice(_UNRELATED_WORDS) for _ in range(rng.randint(0, 4)))
        suffix = " ".join(rng.choice(_UNRELATED_WORDS) for _ in range(rng.randint(0, 4)))
        message = f"{prefix} {sig} {suffix}".strip()
        if not detect_policy_block(Exception(message)):
            misses.append(message)
    assert not misses, f"missed signature variants: {misses[:10]}"


# ---------------------------------------------------------------------------
# Guidance text
# ---------------------------------------------------------------------------


def test_guidance_names_the_operation_and_stays_guidance_only():
    text = policy_block_guidance("the update")
    assert "the update" in text
    assert "Smart App Control" in text
    assert "aka.ms/smartappcontrol" in text
    assert "guidance only" in text.lower()


@pytest.mark.parametrize("context", ["the update", "launch", "install"])
def test_guidance_never_raises_for_any_context(context):
    assert isinstance(policy_block_guidance(context), str)
    assert policy_block_guidance(context)  # never empty


# ---------------------------------------------------------------------------
# Wiring contract: _update_via_zip must actually call the detector
# ---------------------------------------------------------------------------


def test_update_via_zip_wires_policy_block_detection_into_its_failure_path():
    """AST pin: the ZIP-update failure handler must call detect_policy_block
    (and print guidance when it fires) rather than only ever printing the
    raw wire error, or a refactor could silently drop the #87789 fix."""
    src = textwrap.dedent(inspect.getsource(update_cmd._update_via_zip))
    tree = ast.parse(src)

    def _calls(node, name):
        return any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
            for n in ast.walk(node)
        )

    wired = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for handler in node.handlers:
            if _calls(handler, "detect_policy_block") and _calls(
                handler, "policy_block_guidance"
            ):
                wired = True
    assert wired, (
        "_update_via_zip's exception handler no longer calls "
        "detect_policy_block/policy_block_guidance — the raw ZIP-update "
        "failure would go back to being unclassified (#87789)"
    )


def test_zip_update_failure_prints_guidance_when_policy_blocked(capsys):
    """Behavioral confirmation of the classify-and-print step in isolation,
    mirroring exactly what the except block in _update_via_zip does."""
    exc = _os_error_with_winerror(WINERROR_SMART_APP_CONTROL_BLOCK, "blocked copy")
    print(f"✗ ZIP update failed: {exc}")
    if detect_policy_block(exc):
        print(policy_block_guidance("the update"))
    out = capsys.readouterr().out
    assert "ZIP update failed" in out
    assert "Smart App Control" in out


def test_zip_update_failure_stays_silent_on_guidance_for_ordinary_errors(capsys):
    exc = OSError(28, "No space left on device")
    print(f"✗ ZIP update failed: {exc}")
    if detect_policy_block(exc):
        print(policy_block_guidance("the update"))
    out = capsys.readouterr().out
    assert "ZIP update failed" in out
    assert "Smart App Control" not in out
