"""Tool-call arguments streamed WITHOUT their leading bytes must be repaired, not destroyed.

Observed on model V4.1 (DeepSeek-class) served through an OpenAI-compatible router: the SSE
argument deltas begin mid-object, so the assembled arguments string is valid JSON minus a
prefix.  Because the repair pipeline had no prefix pass, every such call was reported
"Unrepairable tool_call arguments ... replaced with empty object", retried, and finally
surfaced to the user as a bogus "Response truncated due to output length limit".

Two shapes were captured:

  a) only the opening ``{"`` never arrived:
        ``command": "cd ~/x && git log"}``
  b) the whole ``{"command": "`` head never arrived:
        ``cd ~/x && git log"}``

The payloads in ``REAL_CAPTURED`` are byte-verbatim copies of the ``(was: ...)`` strings
logged in ``~/.hermes/logs/errors.log`` during that turn, so this test fails if the
prefix-repair pass is removed or narrowed.
"""

import json

import pytest

from agent.message_sanitization import _repair_tool_call_arguments

REAL_CAPTURED = [
    'command": "cd ~/budakorporat-pipeline && sed -n 505,535p test_grounding_contract.py; echo \\"=== the 3.037/4,5x calibration origin? ===\\"; grep -rn \\"3\\\\.037\\\\|4,5x\\" --include=*.py --include=*.md . | grep -v \\"\\\\.bak\\" | head"}',
    'command": "cd ~/budakorporat-pipeline && sed -n 510,530p test_grounding_contract.py"}',
    'cd ~/budakorporat-pipeline && date -u +%Y-%m-%d; echo \\"=== commits ===\\"; for c in 56793c4 81ba778 fd0796d 81cc462; do git log -1 --format=\\"%h %ad %s\\" --date=short $c; done; echo; echo \\"=== file mtime history? ===\\"; git log -1 --format=\\"%h %ad %s\\" --date=short HEAD"}',
    'command": "cd ~/budakorporat-pipeline && git log --oneline --all -S\'**5d.\' | head -5; echo ===; git log --oneline -S\'kok bisa gini sih\' | head -5"}',
    'cd ~/budakorporat-pipeline && for c in $(git log --format=%h -20); do n=$(git show $c:budakorporat_pipeline.py 2>/dev/null | grep -cF \'5d.\'); echo \\"$c $n\\"; done"}',
    'cd ~/budakorporat-pipeline && for c in 81cc462 35df628 81ba778; do echo \\"### $c\\"; git show $c:budakorporat_pipeline.py 2>/dev/null | grep -cF \'**5d.\'; done; echo \\"=== cari di seluruh history ===\\"; git log --oneline -S\'**5d.\' -- budakorporat_pipeline.py | head; echo \\"=== cari di backup .bak & .hermes ===\\"; grep -rlF \'**5d.\' --include=*.py . ~/.hermes/scripts/ 2>/dev/null | head"}',
    '; for c in $(git log --oneline -20 --format=%h); do n=$(git show $c:budakorporat_pipeline.py 2>/dev/null | grep -cF \'**5d.\'); [ \\"$n\\" != \\"0\\" ] && echo \\"$c = $n\\"; done; echo \\"=== grep konsumen 5d di test lain ===\\"; grep -rnF \'5d\' tests/*.py | head"}',
    'command": "cd ~/budakorporat-pipeline && git log --oneline -S\'**5d.\' --all -- budakorporat_pipeline.py | head; echo \\"=== search any commit ===\\"; for c in $(git rev-list --max-count=40 HEAD -- budakorporat_pipeline.py); do if git show $c:budakorporat_pipeline.py | grep -qF \'**5d.\'; then echo \\"FOUND $c\\"; fi; done | head -3"}',
]

# Serialisations of the recovered object, so the prepend-only invariant can be checked
# against whichever separator style the payload was rendered with.
_SEPARATORS = ((",", ":"), (",", ": "), (", ", ":"), (", ", ": "))


@pytest.mark.parametrize("raw", REAL_CAPTURED)
def test_missing_prefix_payloads_are_recovered(raw):
    out = _repair_tool_call_arguments(raw, "terminal")
    assert out != "{}", f"payload still unrepairable: {raw[:80]!r}"
    parsed = json.loads(out)
    assert isinstance(parsed, dict)
    assert parsed.get("command"), "the lost key must be restored"
    # The repair restores a prefix; it must never invent content.  Compared on the serialised
    # form because the payload's ``\"`` escapes re-appear there verbatim, while the DECODED
    # command string legitimately does not appear in the raw escaped payload.
    assert any(
        json.dumps(parsed, separators=sep).endswith(raw.strip()) for sep in _SEPARATORS
    ), f"repair altered received bytes: {raw[:80]!r}"


def test_missing_brace_only():
    """Narrowest shape: only the opening brace is gone."""
    out = _repair_tool_call_arguments('"command": "echo hi"}', "terminal")
    assert json.loads(out) == {"command": "echo hi"}


def test_key_prefix_only():
    """Shape (b): the whole ``{"command": ` head never arrived."""
    out = _repair_tool_call_arguments('echo hi"}', "terminal")
    assert json.loads(out) == {"command": "echo hi"}


def test_missing_head_with_literal_newline_in_value():
    """A dropped head also strands the value's literal newlines; that must not block repair."""
    out = _repair_tool_call_arguments('code": "line1\nline2"}', "execute_code")
    assert json.loads(out) == {"code": "line1\nline2"}


def test_true_truncation_still_rejected():
    """A payload truncated mid-string must not be passed off as recovered."""
    assert _repair_tool_call_arguments('{"command": "cd ~/x && git log', "t") == "{}"
    assert _repair_tool_call_arguments("cd ~/x && git log", "t") == "{}"


def test_intact_object_is_never_rekeyed():
    """A payload that already starts with ``{`` is never touched by the prefix pass."""
    out = _repair_tool_call_arguments('{"command": "cd ~/x", "timeout": 30}', "terminal")
    assert json.loads(out) == {"command": "cd ~/x", "timeout": 30}


def test_non_json_garbage_still_rejected():
    assert _repair_tool_call_arguments("lorem ipsum dolor", "t") == "{}"


def test_schema_prefix_recovers_tools_other_than_terminal():
    """``execute_code``/``write_file`` lost their head too and must be recovered.

    Regression: the prefix candidates only knew ``{"command": "``, so every other tool's
    headless payload was unrepairable, got written back as ``{}`` and surfaced as the bogus
    "Response truncated due to output length limit".
    """
    import run_agent  # noqa: F401  (registers the tool schemas in the registry)

    assert json.loads(
        _repair_tool_call_arguments('code": "print(1)"}', "execute_code")
    ) == {"code": "print(1)"}
    assert json.loads(
        _repair_tool_call_arguments('path": "/tmp/a", "content": "hi"}', "write_file")
    ) == {"path": "/tmp/a", "content": "hi"}


def test_schema_prefix_does_not_mis_key_a_body_that_is_a_bare_string():
    """A schema-derived head must win over the generic ``{"command": "`` fallback.

    For a non-``terminal`` call the received body is itself a valid bare string, so a
    command-shaped head would parse — and silently move the payload into the wrong key.
    """
    import run_agent  # noqa: F401

    out = _repair_tool_call_arguments('echo hi"}', "execute_code")
    assert json.loads(out) == {"code": "echo hi"}


def test_reserialised_form_is_canonical():
    """Recovered output is compact JSON so the canon-args cache stays stable."""
    out = _repair_tool_call_arguments('command": "echo  hi"}', "t")
    assert out == json.dumps({"command": "echo  hi"}, separators=(",", ":"))



def test_non_ascii_payload_is_still_recognised_as_the_tail():
    """A non-ASCII payload must survive: ``json.dumps`` defaults to ``ensure_ascii=True``.

    The prepend-only guard re-serialises the candidate and checks the received bytes are
    its tail.  With the default ASCII escaping a payload containing e.g. ``→`` is claimed
    to be ``\u2192``, the tail check fails, and a perfectly recoverable call is discarded
    (observed on ``memory``, whose Indonesian content is full of non-ASCII punctuation).
    """
    import run_agent  # noqa: F401

    import json as _json

    for ensure_ascii in (False, True):
        for obj, tool in (
            ({"operations": [{"action": "add", "content": "DITOLAK → adopt"}], "target": "memory"}, "memory"),
            ({"path": "/tmp/x.py", "content": 's = "café"\nprint(s)'}, "write_file"),
            ({"command": "grep → /tmp/f"}, "terminal"),
        ):
            full = _json.dumps(obj, ensure_ascii=ensure_ascii)
            prefix = '{"'
            assert full.startswith(prefix)
            out = _repair_tool_call_arguments(full[len(prefix):], tool)
            assert out is not None, (obj, ensure_ascii)
            assert _json.loads(out) == obj, (obj, ensure_ascii)


def test_literal_newline_inside_value_is_recoverable():
    """A dropped head can leave a LITERAL newline in the value; strict JSON rejects it."""
    import run_agent  # noqa: F401

    import json as _json

    full = _json.dumps({"path": "/tmp/a.md", "content": "line1\nline2\tend"})
    assert full.startswith('{"')
    out = _repair_tool_call_arguments(full[2:], "write_file")
    assert out is not None
    assert _json.loads(out) == {"path": "/tmp/a.md", "content": "line1\nline2\tend"}
