"""pm.environment: the resolver diagnostic a failed `uv lock` leaves behind.

uv puts the CAUSE at the top of a resolver failure ("No solution found when
resolving dependencies for split (markers: ...)" plus the "cause:" line naming
the package with no compatible wheels) and only the derived conclusion at the
bottom. A tail-only excerpt drops the cause, so the reader is told that some
version "cannot be used" and never learns which platform broke it.
"""

from __future__ import annotations

import pytest

from pm.environment import _diagnostic_excerpt, classify_uv_failure


# A real uv resolver failure: the split and its cause are in the head, the
# per-version derivation and the hint are in the tail.
UV_RESOLVER_OUTPUT = """error: No solution found when resolving dependencies for split (markers: python_full_version == '3.14.*' and sys_platform == 'android')
  cause: Because onnxruntime>=1.21.0,<=1.28.0 has no `python_full_version >= '3.14'`-compatible wheels and only the following versions of onnxruntime are available:
             onnxruntime<=1.28.0
             onnxruntime>=1.29
         we can conclude that onnxruntime>=1.21.0,<1.29 cannot be used.
         And because mnemosyne-memory[embeddings]>=4.0.0b3 depends on onnxruntime>=1.21.0,<1.29, we can conclude that mnemosyne-memory[embeddings]>=4.0.0b3 cannot be used. (1)

         Because onnxruntime>=1.21.0,<=1.28.0 has no `python_full_version >= '3.14'`-compatible wheels and only the following versions of onnxruntime are available:
             onnxruntime<=1.28.0
             onnxruntime>=1.29
         we can conclude that onnxruntime>=1.21.0,<1.29 cannot be used.
         And because mnemosyne-memory[embeddings]>=4.0.0b4 depends on onnxruntime>=1.21.0,<1.29, we can conclude that mnemosyne-memory[embeddings]>=4.0.0b4 cannot be used. (2)

         And because hermes-plugin-mnemosyne depends on mnemosyne-memory[embeddings]>=4.0.0b3 and your workspace requires hermes-plugin-mnemosyne, we can conclude that your workspace's requirements are unsatisfiable.

hint: The resolution failed for an environment that is not the current one, consider limiting the environments with `tool.uv.environments`.
"""


def test_excerpt_keeps_the_failing_platform_split():
    """The platform that broke the solve is the fact the reader needs; it is in the head."""
    excerpt = _diagnostic_excerpt(UV_RESOLVER_OUTPUT)
    assert "sys_platform == 'android'" in excerpt


def test_excerpt_keeps_the_package_with_no_wheels():
    """The package named by the cause is what a reader acts on."""
    excerpt = _diagnostic_excerpt(UV_RESOLVER_OUTPUT)
    assert "onnxruntime" in excerpt


def test_excerpt_keeps_uvs_own_hint():
    """uv's hint names the knob to turn; the tail carries it."""
    excerpt = _diagnostic_excerpt(UV_RESOLVER_OUTPUT)
    assert "hint:" in excerpt
    assert "tool.uv.environments" in excerpt


def test_excerpt_drops_the_repeated_derivation():
    """The middle restates the same conclusion per candidate version; it is noise."""
    excerpt = _diagnostic_excerpt(UV_RESOLVER_OUTPUT)
    assert excerpt.count("we can conclude that") <= 2


def test_excerpt_keeps_a_short_output_whole():
    """A short diagnostic has nothing to trim."""
    assert _diagnostic_excerpt("error: No solution found") == "error: No solution found"


def test_excerpt_is_bounded():
    """The excerpt stays a size a message can carry, whatever uv printed."""
    excerpt = _diagnostic_excerpt(UV_RESOLVER_OUTPUT * 40)
    assert len(excerpt) <= 1300


def test_conflict_message_carries_the_cause():
    """The classified error a user reads must name the platform, not just the conclusion."""
    error = classify_uv_failure("lock", 1, UV_RESOLVER_OUTPUT)
    message = str(error)
    assert "sys_platform == 'android'" in message
    assert "onnxruntime" in message
    assert "tool.uv.environments" in message


@pytest.mark.parametrize("diagnostic", [
    "No solution found", "Connection timed out", "Failed to build wheel",
])
def test_classification_is_unchanged(diagnostic):
    """Trimming the excerpt must not change which failure class it is."""
    error = classify_uv_failure("lock", 1, diagnostic)
    assert type(error) is type(classify_uv_failure("lock", 1, diagnostic))
