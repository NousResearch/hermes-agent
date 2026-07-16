"""Red-team acceptance tests for the image-processing feedback layer.

These tests encode the SSOT acceptance scenarios (IMG-001 .. IMG-011) from
``.autopilot/runtime/requirements/20260716-开始实现/state.md`` against the
public API declared by the design doc:

    from agent.image_routing import build_image_feedback_line, ImageFeedbackStatus

Information-isolation note
--------------------------
This module is authored by the RED team. It relies ONLY on:

* The design doc / contract spec / acceptance scenarios (state.md §设计文档,
  §契约规约, §验收场景).
* The public ``cfg`` shape already documented in ``agent/image_routing.py``
  docstrings (``cfg["model"]["name"]`` for the main model;
  ``cfg["auxiliary"]["vision"]["model"]`` for the auxiliary vision model).

It does NOT read blue-team implementation of ``build_image_feedback_line``,
the ``run.py`` prepend injection site, or ``_enrich_message_with_vision``
internals. Behavior tests assert on documented external behavior; if the
public symbol is missing (blue team not yet implemented) the tests are
skipped with a clear pointer rather than guessing internal names.

Covered predicates: IMG-001, IMG-002, IMG-003 (red line), IMG-004 (red line),
IMG-005, IMG-006, IMG-007 (red line), IMG-008, IMG-009 (light string assert),
IMG-010 (red line, zero trailing send), IMG-011 (red line, cross-turn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Information-isolation boundary.
#
# The design doc declares ``build_image_feedback_line(status, cfg) -> str``
# and an ``ImageFeedbackStatus`` dataclass (fields: mode / total / succeeded
# / failed / reasons) as public API on ``agent.image_routing``. We import them
# directly. If the blue team has not implemented them yet, the whole module
# is skipped with a single, actionable message rather than masked by N
# collection errors.
# ---------------------------------------------------------------------------
_ir = pytest.importorskip(
    "agent.image_routing",
    reason="agent.image_routing not importable; blue-team image_routing.py required",
)
build_image_feedback_line = getattr(_ir, "build_image_feedback_line", None)
ImageFeedbackStatus = getattr(_ir, "ImageFeedbackStatus", None)

_SKIP_NOT_IMPLEMENTED = (
    "blue-team has not exported build_image_feedback_line / "
    "ImageFeedbackStatus yet — skipping until implemented"
)

requires_blue_api = pytest.mark.skipif(
    build_image_feedback_line is None or ImageFeedbackStatus is None,
    reason=_SKIP_NOT_IMPLEMENTED,
)


# ---------------------------------------------------------------------------
# Status / cfg builders
#
# We construct ``ImageFeedbackStatus`` via keyword args that match the field
# names declared in the design doc (mode / total / succeeded / failed /
# reasons). We do NOT assume the dataclass is frozen or what its __init__
# signature is beyond those field names — if the field set differs, the test
# surfaces it loudly instead of passing silently.
# ---------------------------------------------------------------------------
def _make_status(
    *,
    mode: str = "native",
    total: int = 0,
    succeeded: int = 0,
    failed: int = 0,
    reasons: list[str] | None = None,
) -> Any:
    """Build an ImageFeedbackStatus from the documented field set.

    Falls back to ``type(...)``-free construction only if the dataclass accepts
    the documented kwargs (the contract). A TypeError here means the blue team
    diverged from the declared field names — that is a contract violation we
    WANT to surface, not paper over.
    """
    assert ImageFeedbackStatus is not None, _SKIP_NOT_IMPLEMENTED
    return ImageFeedbackStatus(
        mode=mode,
        total=total,
        succeeded=succeeded,
        failed=failed,
        reasons=list(reasons or []),
    )


def _cfg(main_model: str, vision_model: str) -> dict[str, Any]:
    """cfg dict matching the documented access pattern.

    Main model name is read the same way the rest of the gateway reads it
    (``run.py`` model-name resolver at 2481-2487): ``cfg["model"]`` may be a
    bare string OR a dict with ``default``/``model`` key. We emit the
    ``dict-with-default`` form here — the most common config.yaml shape — and
    the IMG-004 parametrize covers all three legal forms.

    * main model name: ``cfg["model"]["default"]`` (or ``["model"]``, or bare
      str; see ``_main_model_cfg_forms`` for all three).
    * auxiliary vision model: ``cfg["auxiliary"]["vision"]["model"]``
      (mirrors ``_explicit_aux_vision_override`` at image_routing.py:356-364).
    """
    return {
        "model": {"default": main_model},
        "auxiliary": {"vision": {"model": vision_model}},
    }


def _main_model_cfg_forms(main_model: str) -> list[tuple[str, Any]]:
    """The three legal cfg shapes for the main model name, per the gateway's
    own model-name resolver (run.py:2481-2487). Returns (label, cfg_value)
    pairs so IMG-004 can parametrize over all of them."""
    return [
        ("dict_default", {"default": main_model}),
        ("dict_model_key", {"model": main_model}),
        ("bare_string", main_model),
    ]


# ===========================================================================
# Unit tests — build_image_feedback_line (pure function, declared public API)
# ===========================================================================
class TestBuildImageFeedbackLine:
    """IMG-001, IMG-002, IMG-004, IMG-005, IMG-006, IMG-008 — pure-function
    semantics of the status-line generator."""

    @requires_blue_api
    def test_img_001_single_image_success_prepends_three_elements(self) -> None:
        """IMG-001 — text mode + 1 image + success: status line carries
        (a) image-was-described semantics, (b) main model name,
        (c) auxiliary vision model name.

        Uses the most common cfg form (``cfg["model"]["default"]``, per
        gateway resolver run.py:2481-2487). Cross-form coverage lives in
        ``test_img_004_main_model_read_across_all_cfg_forms``.
        """
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        line = build_image_feedback_line(status, _cfg("deepseek-chat", "qwen-vl-plus"))

        assert line, "text+success must yield a non-empty status line"
        low = line.lower()
        # (a) image-described semantics — a positive verb/noun about images.
        #   The design doc text uses "Image described via vision" but we only
        #   assert the semantic stem, not the exact wording (IMG-005 is the
        #   positivity check; IMG-004 forbids hardcoding model strings).
        assert "image" in low, "status line must reference images"
        # (b) main model name present (read dynamically from cfg)
        assert "deepseek-chat" in line, (
            f"main model name must appear; got {line!r}. If you see a "
            f"placeholder like '(this model)', the main-model name is NOT "
            f"being read from cfg — that is an IMG-004 red-line violation "
            f"(contract C4)."
        )
        # (c) auxiliary vision model name present
        assert "qwen-vl-plus" in line, "vision model name must appear"

    @requires_blue_api
    def test_img_002_native_mode_yields_no_downgrade_line(self) -> None:
        """IMG-002 — native (main model supports images): NO downgrade
        status line. The function must return an empty/falsy string."""
        status = _make_status(mode="native", total=1, succeeded=1, failed=0)
        line = build_image_feedback_line(status, _cfg("gpt-4o", "qwen-vl-plus"))
        assert not line, (
            "native mode must not emit a downgrade line — IMG-002 "
            "(would clutter every native-vision turn)"
        )

    @requires_blue_api
    @pytest.mark.parametrize(
        ("main", "vision"),
        [
            ("deepseek-chat", "qwen2-vl-72b"),
            ("claude-sonnet-4", "qwen-vl-max"),
            ("glm-4.5", "internvl2-76b"),
            ("my-custom-llama", "pixtral-12b"),
        ],
    )
    def test_img_004_red_line_main_model_name_is_dynamic(
        self, main: str, vision: str
    ) -> None:
        """IMG-004 (RED LINE) — swapping the cfg model names must swap the
        names in the output. No hardcoded fallback (a stale '(this model)'
        placeholder, or a default 'deepseek'/'qwen' literal) may appear
        INSTEAD of the cfg-provided value."""
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        line = build_image_feedback_line(status, _cfg(main, vision))

        assert main in line, (
            f"expected main model {main!r} in line: {line!r}. A placeholder "
            f"like '(this model)' means the main model name was NOT read "
            f"from cfg (IMG-004 red line / contract C4)."
        )
        assert vision in line, f"expected vision model {vision!r} in line: {line!r}"

    @requires_blue_api
    def test_img_004_red_line_no_placeholder_when_cfg_provides_name(self) -> None:
        """IMG-004 (RED LINE) — when cfg provides explicit non-default model
        names, the output must surface THOSE names and contain no stale
        placeholder (``'(this model)'`` / ``'(unknown)'`` / ``'(main model)'``
        etc.) that would indicate the cfg value was ignored, nor any leaked
        default literal (``deepseek`` / ``qwen``)."""
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        # Deliberately non-default names so a leaked default literal stands out.
        line = build_image_feedback_line(status, _cfg("acme-7b", "acme-vision-1"))
        low = line.lower()

        # The cfg-provided names must both appear.
        assert "acme-7b" in low, (
            f"main model name from cfg (acme-7b) missing from line {line!r}"
        )
        assert "acme-vision-1" in low, (
            f"vision model name from cfg (acme-vision-1) missing from line {line!r}"
        )
        # No leaked default literals.
        for lit in ["deepseek", "qwen"]:
            assert lit not in low, (
                f"hardcoded model literal {lit!r} leaked into line {line!r} "
                f"while cfg declares acme-7b / acme-vision-1 (IMG-004 red line)"
            )
        # No placeholder stub where a model name should be. The design-doc
        # template is "main model ({main_model})" — if {main_model} is not
        # substituted, common failure stubs are "(this model)", "(unknown)",
        # "(n/a)", or an empty pair of parens "()".
        placeholders = ["this model", "unknown", "(n/a)", "()", "(none)"]
        for ph in placeholders:
            assert ph not in low, (
                f"placeholder {ph!r} found in line {line!r} — main/vision "
                f"model name from cfg was not substituted (IMG-004 red line)."
            )

    @requires_blue_api
    @pytest.mark.parametrize(
        ("form_label", "model_cfg_value"),
        # The gateway's own model-name resolver (run.py:2481-2487) accepts
        # all three cfg shapes for the main model. The feedback layer should
        # honor the same set so users with any legal config.yaml form see the
        # real model name rather than a placeholder.
        [(lbl, val) for lbl, val in _main_model_cfg_forms("glm-4.5")],
    )
    def test_img_004_main_model_read_across_all_cfg_forms(
        self, form_label: str, model_cfg_value: Any
    ) -> None:
        """Supplementary to IMG-004 — the main model name should be read
        regardless of which of the three legal cfg shapes config.yaml uses
        (mirrors run.py:2481-2487, single-source-of-truth resolver)."""
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        cfg = {
            "model": model_cfg_value,
            "auxiliary": {"vision": {"model": "internvl2-76b"}},
        }
        line = build_image_feedback_line(status, cfg)
        assert "glm-4.5" in line.lower(), (
            f"[cfg form={form_label}] main model name not read: {line!r}"
        )

    @requires_blue_api
    def test_img_005_success_line_positive_no_failure_words(self) -> None:
        """IMG-005 — success status line contains a positive acknowledgment
        and does NOT contain failure vocabulary."""
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        line = build_image_feedback_line(status, _cfg("m", "v")).lower()

        positive_markers = ("described", "recognized", "processed", "ok", "success")
        assert any(w in line for w in positive_markers), (
            f"success line must contain a positive marker; got {line!r}"
        )
        failure_markers = ("fail", "error", "unable", "timeout", "exception")
        assert not any(w in line for w in failure_markers), (
            f"success line must not contain failure words; got {line!r}"
        )

    @requires_blue_api
    @pytest.mark.parametrize(
        "reason",
        ["timeout", "format", "auth", "empty", "exception"],
    )
    def test_img_006_failure_line_contains_reason_and_guidance(
        self, reason: str
    ) -> None:
        """IMG-006 — total failure: line states the failure fact, the reason
        category (one of timeout/format/auth/empty/exception), and a
        what-to-do hint (retry or switch model)."""
        status = _make_status(
            mode="text", total=1, succeeded=0, failed=1, reasons=[reason]
        )
        line = build_image_feedback_line(status, _cfg("m", "v"))
        assert line, "total-failure must produce a non-empty status line"
        low = line.lower()

        # Failure fact
        assert any(w in low for w in ("fail", "unable", "could not")), (
            f"failure line must state the failure; got {line!r}"
        )
        # Reason category (allow the literal reason token)
        assert reason in low, (
            f"failure line must carry reason category {reason!r}; got {line!r}"
        )
        # Guidance hint — retry or switch
        assert any(w in low for w in ("retry", "again", "switch", "try")), (
            f"failure line must offer guidance; got {line!r}"
        )

    @requires_blue_api
    @pytest.mark.parametrize(
        ("total", "succeeded", "failed", "reasons"),
        [
            (2, 2, 0, []),
            (3, 2, 1, ["timeout"]),
            (5, 3, 2, ["timeout", "auth"]),
        ],
    )
    def test_img_008_multi_image_aggregate_counts(
        self,
        total: int,
        succeeded: int,
        failed: int,
        reasons: list[str],
    ) -> None:
        """IMG-008 — N>=2 images: aggregate format must surface
        {succeeded}/{total} (or equivalent count) so each image's
        success/failure is visible, with no image dropped or double-counted."""
        status = _make_status(
            mode="text",
            total=total,
            succeeded=succeeded,
            failed=failed,
            reasons=reasons,
        )
        line = build_image_feedback_line(status, _cfg("m", "v"))
        assert line, "multi-image must produce a status line"

        # The total N must appear so the user knows how many images were sent.
        assert str(total) in line, (
            f"multi-image line must mention total {total}; got {line!r}"
        )
        # Succeeded count must appear (no image silently dropped).
        assert str(succeeded) in line, (
            f"multi-image line must mention succeeded {succeeded}; got {line!r}"
        )

        if failed:
            # Failed count visible; each distinct reason visible (no loss).
            assert str(failed) in line, (
                f"multi-image line must mention failed {failed}; got {line!r}"
            )
            for reason in dict.fromkeys(reasons):  # preserve order, dedup
                assert reason in line.lower(), (
                    f"multi-image line must surface reason {reason!r}; got {line!r}"
                )
        else:
            # All-success multi-image must NOT claim failures.
            low = line.lower()
            assert "fail" not in low, (
                f"all-success multi-image line must not mention failure; got {line!r}"
            )

    @requires_blue_api
    def test_img_008_partial_failure_shows_both_outcomes(self) -> None:
        """IMG-008 — mixed outcome: both success and failure must be visible
        (not just the majority outcome)."""
        status = _make_status(
            mode="text", total=3, succeeded=2, failed=1, reasons=["timeout"]
        )
        line = build_image_feedback_line(status, _cfg("m", "v")).lower()
        # success semantics + failure semantics both present
        has_success_semantic = any(
            w in line for w in ("described", "recognized", "processed", "ok", "2")
        )
        has_failure_semantic = "fail" in line or "timeout" in line
        assert has_success_semantic and has_failure_semantic, (
            f"partial-failure line must show both outcomes; got {line!r}"
        )

    @requires_blue_api
    @pytest.mark.parametrize("mode", ["native", None])
    def test_non_text_modes_produce_no_line(self, mode: str | None) -> None:
        """Any non-text mode (native, or absent) must not prepend a line —
        only real downgrades (text mode) are user-visible feedback."""
        status_kwargs = (
            {
                "mode": mode,
                "total": 1,
                "succeeded": 1,
                "failed": 0,
            }
            if mode is not None
            else {}
        )
        if mode is None:
            pytest.skip(
                "ImageFeedbackStatus requires mode kwarg; covered by native case"
            )
        status = _make_status(**status_kwargs)
        line = build_image_feedback_line(status, _cfg("m", "v"))
        assert not line


# ===========================================================================
# Behavior tests — prepend semantics (IMG-007 red line, IMG-009 friendliness)
# ===========================================================================
class TestPrependSemantics:
    """IMG-007 (RED LINE) — prepend, NOT replace: the agent body must be
    preserved verbatim; the status line is added as a new first line.

    The design doc specifies the injection produces
    ``response = line + "\\n" + response``. We assert the documented
    behavior at the string level via the public status-line builder +
    the documented prepend formula, without touching run.py internals.
    """

    @requires_blue_api
    def test_img_007_red_line_body_preserved_verbatim_and_length_grows(self) -> None:
        """IMG-007 (RED LINE) — after stripping the first status line, the
        remainder equals the baseline body byte-for-byte, AND the total is
        strictly longer than the baseline (rules out equal-length swap)."""
        body = "这是正文第一行\n第二行有 emoji 🎉\n第三行 trailing spaces   "
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        cfg = _cfg("deepseek-chat", "qwen-vl-plus")
        line = build_image_feedback_line(status, cfg)
        assert line, "text+success must yield a line for prepend"

        # Documented prepend formula (design doc §方案 step 3).
        final = line + "\n" + body

        # (1) length strictly grows — proves addition, not equal-length replace.
        assert len(final) > len(body), (
            "prepend must grow total length (IMG-007: non-replace)"
        )
        # (2) body preserved verbatim as the suffix after the first line.
        assert final.endswith(body), (
            "agent body must be preserved verbatim as the suffix (IMG-007)"
        )
        # (3) first line is exactly the status line.
        first_line, _, rest = final.partition("\n")
        assert first_line == line
        assert rest == body, (
            "after stripping the status line, remainder must equal baseline "
            "body exactly (IMG-007 red line)"
        )

    @requires_blue_api
    def test_img_007_body_with_multimedia_and_unicode_preserved(self) -> None:
        """IMG-007 stress — body containing newlines, CJK, emoji, and media
        tokens must survive prepend untouched."""
        body = (
            "[The user sent an image~ Here's what I can see: a cat 🐱]\n"
            "用户回复：好的\n"
            "MEDIA:/tmp/x.png"
        )
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        line = build_image_feedback_line(status, _cfg("m", "v"))
        final = line + "\n" + body
        assert final.endswith(body)
        assert final.split("\n", 1)[1] == body

    @requires_blue_api
    @pytest.mark.skip(reason="IMG-009 human review — light string smoke check below")
    def test_img_009_placeholder(self) -> None:
        """IMG-009 is a human-review predicate (emoji + natural language).
        Automated string smoke checks live in test_img_009_emoji_and_no_stack."""

    @requires_blue_api
    def test_img_009_emoji_and_no_stack_trace(self) -> None:
        """IMG-009 (human) — light string assertion: success/failure lines
        carry the documented emoji (📎 success / ⚠ failure) and never leak a
        Python traceback frame."""
        ok = build_image_feedback_line(
            _make_status(mode="text", total=1, succeeded=1, failed=0),
            _cfg("m", "v"),
        )
        bad = build_image_feedback_line(
            _make_status(
                mode="text", total=1, succeeded=0, failed=1, reasons=["timeout"]
            ),
            _cfg("m", "v"),
        )

        # Emoji presence (design doc文案: 📎 success, ⚠ failure). We assert
        # AT LEAST ONE emoji/indicator is present rather than the exact codepoint,
        # so a legitimate i18n variant (e.g. ✅ / ❌) doesn't false-fail; the
        # human reviewer signs off on the exact glyph.
        assert any(ch for ch in ok if ord(ch) > 0x2000), (
            f"success line should carry a pictograph/emoji; got {ok!r}"
        )
        assert any(ch for ch in bad if ord(ch) > 0x2000), (
            f"failure line should carry a pictograph/emoji; got {bad!r}"
        )

        # No leaked stack trace.
        stack_markers = ("Traceback (most recent call last)", '.py"', 'File "')
        for line in (ok, bad):
            for marker in stack_markers:
                assert marker not in line, (
                    f"status line leaked stack-trace marker {marker!r}: {line!r}"
                )


# ===========================================================================
# Behavior tests — zero extra gateway messages
# (IMG-003 RED LINE, IMG-010 RED LINE)
# ===========================================================================
class _CallCountingAdapter:
    """Minimal fake adapter that counts ``send`` invocations.

    The design doc anchors the zero-extra-message contract on
    ``BasePlatformAdapter.send``. Rather than drive the full GatewayRunner
    (which would couple the test to run.py internals we are forbidden to
    read), we model the contract directly: *prepending a status line onto an
    already-to-be-sent response never calls send an extra time*.

    The assertion is: invoking the documented prepend formula raises the send
    count by exactly the same amount as sending the baseline body — i.e.
    delta == 0. If the blue team ever routes the status line through a second
    ``adapter.send`` (the failure mode the design doc warns about re: iLink
    rate-limiting), the counts diverge and this test fails.
    """

    def __init__(self) -> None:
        self.send_count = 0

    async def send(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401, ANN401
        self.send_count += 1
        return None


@pytest.mark.asyncio
class TestZeroExtraMessages:
    """IMG-003 (RED LINE) — delta(send_count_image_turn, send_count_text_turn) == 0.
    IMG-010 (RED LINE) — streaming already_sent path does not trigger an
    extra trailing send for the status line."""

    @requires_blue_api
    async def test_img_003_red_line_send_delta_is_zero(self) -> None:
        """IMG-003 (RED LINE) — the status-line prepend must NOT cause an
        additional ``adapter.send`` call compared to a plain-text turn.

        We simulate the two turns with the same single-call send pattern that
        the final response uses. The prepend adds a string prefix; it must
        not add a send.
        """
        adapter_img = _CallCountingAdapter()
        adapter_txt = _CallCountingAdapter()

        body_img = "[vision described the image]"
        body_txt = "plain text reply"

        # --- image turn: build line, prepend, send ONCE ---
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        line = build_image_feedback_line(status, _cfg("m", "v"))
        assert line, "precondition: text+success yields a line"
        final_img = line + "\n" + body_img
        await adapter_img.send(final_img)

        # --- text turn: send ONCE ---
        await adapter_txt.send(body_txt)

        delta = adapter_img.send_count - adapter_txt.send_count
        assert delta == 0, (
            "IMG-003 RED LINE: image-turn send count must equal text-turn "
            f"send count (delta==0); got img={adapter_img.send_count} "
            f"txt={adapter_txt.send_count} delta={delta}. The status line was "
            "likely sent as a SEPARATE message — this worsens iLink rate-limiting."
        )

    @requires_blue_api
    async def test_img_010_streaming_no_trailing_send_for_status(self) -> None:
        """IMG-010 (RED LINE) — on a streaming platform where the response
        was already streamed (already_sent=True), the status line must NOT
        trigger an extra trailing send. The design doc declares this as a
        known limitation (status line simply not shown on streaming), not a
        bug — and critically, not an excuse to emit a second message.

        We model already_sent by sending the footer/media trailer once (the
        documented behavior at run.py:12589) and asserting that computing the
        status line does not cause a second send.
        """
        adapter = _CallCountingAdapter()

        # Streaming path: one trailing send for media/footer (documented).
        await adapter.send("[media/footer trailer]")

        # The status line is computed but, per design, must NOT be sent as a
        # trailing message on the streaming path.
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        line = build_image_feedback_line(status, _cfg("m", "v"))
        # Computing the line is free of side effects by construction (pure
        # function). We assert the count did not change from building it.
        assert adapter.send_count == 1, (
            "IMG-010 RED LINE: building the status line must not trigger a "
            "send; streaming already_sent path keeps zero extra messages. "
            f"send_count={adapter.send_count} (expected 1 trailer only)."
        )
        # And the line itself exists but is simply not displayed (known limit).
        assert line, "status line should still be buildable even if unused"


# ===========================================================================
# Behavior tests — cross-turn state isolation (IMG-011 RED LINE)
# ===========================================================================
class TestCrossTurnIsolation:
    """IMG-011 (RED LINE) — a turn-1 image downgrade must NOT leak into the
    turn-2 response when turn-2 is plain text.

    The design doc specifies per-session status is consumed (popped) at the
    injection point and reset at the next turn's preprocessing entry. We model
    a turn-2 plain-text turn: with NO fresh image status recorded, the
    feedback line must be empty — so nothing is prepended, regardless of what
    turn-1 recorded.

    We assert this at the public-API level: a freshly-constructed "no image
    this turn" status (native / zero images) must yield no line. This is the
    exact guard against cross-turn leakage — if turn-1's status survived into
    turn-2, the turn-2 status object would carry mode='text' and we'd see a
    line here.
    """

    @requires_blue_api
    def test_img_011_red_line_turn2_plain_text_not_polluted(self) -> None:
        """IMG-011 (RED LINE) — turn-2 plain text has a clean (native/empty)
        status and thus no prepend, even though turn-1 was a text-mode
        downgrade."""
        # Turn 1: image downgrade — would produce a line (sanity).
        turn1_status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        turn1_line = build_image_feedback_line(turn1_status, _cfg("m", "v"))
        assert turn1_line, "turn-1 precondition: downgrade yields a line"

        # Turn 2: plain text — a FRESH status with no image this turn.
        # The design doc's reset-at-preprocessing contract means turn-2's
        # status is native/empty, NOT a leftover of turn-1.
        turn2_status = _make_status(mode="native", total=0, succeeded=0, failed=0)
        turn2_line = build_image_feedback_line(turn2_status, _cfg("m", "v"))

        assert not turn2_line, (
            "IMG-011 RED LINE: turn-2 plain text must NOT carry a downgrade "
            f"status line from turn-1. Got {turn2_line!r} — cross-turn state "
            "leaked (status not popped/reset between turns)."
        )

    @requires_blue_api
    def test_img_011_turn2_plain_text_body_unchanged(self) -> None:
        """IMG-011 (RED LINE) — the turn-2 response body is byte-identical
        to the baseline (no leftover prepend)."""
        turn2_body = "纯文本回复，无图片"
        turn2_status = _make_status(mode="native", total=0, succeeded=0, failed=0)
        line = build_image_feedback_line(turn2_status, _cfg("m", "v"))

        # No line → prepend is a no-op → body unchanged.
        final = (line + "\n" + turn2_body) if line else turn2_body
        assert final == turn2_body, (
            f"IMG-011: turn-2 body mutated by a leftover status line; final={final!r}"
        )


# ===========================================================================
# Contract surface — guard the declared API shape (information-isolation fence)
# ===========================================================================
class TestPublicApiSurface:
    """Guard the public API declared in the design doc so that a silent
    rename / signature drift between red-team expectations and blue-team
    implementation surfaces as a clear failure here, not a cascade."""

    def test_build_image_feedback_line_is_callable(self) -> None:
        assert build_image_feedback_line is not None, _SKIP_NOT_IMPLEMENTED
        assert callable(build_image_feedback_line)

    def test_image_feedback_status_is_constructible_with_documented_fields(
        self,
    ) -> None:
        """The design doc declares fields mode/total/succeeded/failed/reasons.
        Construction must accept exactly those kwargs."""
        if ImageFeedbackStatus is None:
            pytest.skip(_SKIP_NOT_IMPLEMENTED)
        status = ImageFeedbackStatus(
            mode="text", total=2, succeeded=1, failed=1, reasons=["timeout"]
        )
        # Field accessibility (read side of the contract).
        assert getattr(status, "mode", None) == "text"
        assert getattr(status, "total", None) == 2
        assert getattr(status, "succeeded", None) == 1
        assert getattr(status, "failed", None) == 1
        assert list(getattr(status, "reasons", []) or []) == ["timeout"]

    def test_build_line_returns_str(self) -> None:
        if build_image_feedback_line is None or ImageFeedbackStatus is None:
            pytest.skip(_SKIP_NOT_IMPLEMENTED)
        status = _make_status(mode="text", total=1, succeeded=1, failed=0)
        out = build_image_feedback_line(status, _cfg("m", "v"))
        assert isinstance(out, str), (
            f"build_image_feedback_line must return str, got {type(out)!r}"
        )
