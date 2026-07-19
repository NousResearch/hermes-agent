"""self-verification plugin — audit AI output and tool results with confidence scoring and self-refute.

Hooks:
  ``transform_llm_output`` — audits the final LLM response before delivery.
  ``transform_tool_result`` — verifies intermediate tool call results (NEW in v0.3.0).

Features (v0.3.0):
  - Confidence scoring (0-100 continuous, not binary pass/fail)
  - Self-refute stage: adversarial disprove of each finding
  - Auto-fix retry loop: max 3 retries with context injection
  - Language-aware footnotes (zh/en, follows config.yaml display.language)
  - Output completeness check (VMAO sub-goal coverage)
  - Non-blocking warn mode by default; strict blocking via config
  - Tool result verification: write_file compile check, patch success check,
    terminal exit code check, web_search emptiness check

Architecture:
  - ``_on_transform_llm_output`` fires once per turn after the tool-calling
    loop completes. It runs verification, applies self-refutation, checks
    confidence threshold, and appends a footnote for warn/fail verdicts.
  - ``_on_transform_tool_result`` fires after each tool call and performs
    fast local checks (no LLM) on the tool result — Python syntax errors,
    JSON parse failures, patch errors, non-zero exit codes, empty searches.
  - The retry loop (max 3) is triggered when strict mode is enabled and
    verification fails — the issues are injected as context for a retry.

References:
  - Hermes Plugin API: hermes_cli/plugins.py VALID_HOOKS
  - Claude Code code-review: 0-100 scoring, threshold 80
  - Claude Code security-guidance: Investigate → Self-Refute
  - Aider reflected_message: inject failure context for retry
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

# Use importlib for robustness — the plugin directory name contains a hyphen
# ("self-verification"), and relative imports can fail when the module is
# loaded outside Hermes' native plugin loading context.
# When loaded as a package, __package__ is "plugins.self-verification".
# When loaded standalone (e.g., by tests), fall back to the full module path.
_pkg = __package__ or "plugins.self-verification"
_conf = importlib.import_module(".conf", package=_pkg)
_verifier = importlib.import_module(".verifier", package=_pkg)

logger = logging.getLogger(__name__)

# Maximum number of auto-fix retries
_MAX_RETRIES = 3


def _on_transform_llm_output(
    response_text: str = "",
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **kwargs: Any,
) -> str | None:
    """transform_llm_output hook — audit the final response and append footnote.

    Called once per turn after the tool-calling loop completes. If the
    Verifier flags issues and the confidence score is below the threshold,
    a language-aware footnote is appended to the response.

    In strict (blocking) mode, returns a retry instruction instead of the
    original response. The retry loop is capped at 3 by the caller.

    Returns the (possibly modified) response text, or None to leave unchanged.
    """
    # Check if plugin is disabled
    if _conf.is_plugin_disabled():
        return None

    # Also check legacy is_enabled() path
    if not _verifier.is_enabled():
        return None

    if not response_text or not response_text.strip():
        return None

    # Skip on messaging surfaces (Telegram, WeChat, etc.) — verification
    # footnotes would be chat noise.
    if platform and platform not in {
        "",
        "cli",
        "desktop",
        "tui",
        "gateway",
        "local",
        "tool",
    }:
        logger.debug(
            "self-verification: skipping messaging surface '%s'", platform
        )
        return None

    # Read language setting
    lang = _conf.get_language()

    # Run verification
    result = _verifier.verify_with_timeout(response_text)
    if result is None:
        # Skipped (no risk signals), timed out, or Verifier failed — silent
        return None

    verdict = result.get("verdict", "pass")
    if verdict == "pass":
        return None

    # Apply self-refute to filter false positives
    claims = result.get("claims") or []
    if claims:
        survived_claims = _verifier.self_refute(claims)
        result["claims"] = survived_claims

        # Re-check verdict after self-refute — if all claims were refuted,
        # demote the verdict to pass
        if not survived_claims:
            remaining_contradictions = result.get("contradictions") or []
            remaining_sub_goals = result.get("sub_goals") or []
            if not remaining_contradictions and not remaining_sub_goals:
                logger.debug(
                    "self-verification: all claims refuted, demoting to pass"
                )
                return None
            # If only contradictions/sub-goals remain, keep at most warn
            result["verdict"] = "warn"

    # Score the result
    confidence = _verifier.score_claims(
        result.get("claims") or [],
        result.get("contradictions") or [],
        result.get("sub_goals") or [],
    )

    # Check against threshold
    threshold = _conf.get_confidence_threshold()
    if confidence >= threshold:
        logger.debug(
            "self-verification: confidence %d >= threshold %d, no issues",
            confidence,
            threshold,
        )
        return None

    # Re-check verdict post-threshold
    verdict = result.get("verdict", "pass")
    if verdict == "pass":
        return None

    # Store confidence in result for footer
    result["confidence"] = confidence

    # Build and append footnote
    footer = _verifier.format_verification_footer(result, lang=lang, threshold=threshold)
    if not footer:
        return None

    logger.info(
        "self-verification: verdict=%s, confidence=%d, claims=%d, session=%s",
        verdict,
        confidence,
        len(result.get("claims") or []),
        session_id,
    )

    # Persist verification result for "修正" flow
    _verifier.save_last_result(result, response_text, session_id)

    # In strict mode, return a retry instruction instead of footnote
    if _conf.is_strict_mode():
        return _format_retry_message(result, lang=lang)

    return response_text.rstrip() + "\n\n" + footer


def _format_retry_message(result: dict[str, Any], lang: str = "zh") -> str:
    """Format a retry instruction message for strict (blocking) mode.

    This replaces the original response with a structured correction request
    that the agent should process in a retry loop.
    """
    i18n = _verifier._get_i18n(lang)

    claims = result.get("claims") or []
    contradictions = result.get("contradictions") or []
    sub_goals = result.get("sub_goals") or []
    confidence = result.get("confidence", 0)

    verdict_str = result.get("verdict", "")
    if lang == "zh":
        verdict_label = {"warn": "⚠️ 需关注", "fail": "❌ 不通过"}.get(verdict_str, verdict_str)
    else:
        verdict_label = {"warn": "⚠️ Attention needed", "fail": "❌ Failed"}.get(verdict_str, verdict_str)

    lines = [
        f"{i18n['title']}: {verdict_label}",
        f"{i18n['confidence_label']}: {confidence}/100",
        "",
    ]

    # List issues for correction
    for claim in claims:
        text = (claim.get("text") or "")[:150]
        note = claim.get("note") or ""
        lines.append(f"- [{claim.get('risk_level', 'unknown')}] {text}")
        if note:
            lines.append(f"  原因: {note}")

    for contra in contradictions:
        if contra.get("severity") == "critical":
            a = (contra.get("statement_a") or "")[:100]
            b = (contra.get("statement_b") or "")[:100]
            lines.append(f"- [矛盾] 「{a}」↔「{b}」")

    for goal in sub_goals:
        if goal.get("status") in ("missing", "partial"):
            g = (goal.get("goal") or "")[:100]
            lines.append(f"- [{goal.get('status')}] {g}")

    lines.append("")
    lines.append("请逐条核实上述问题，修正不准确的内容后重新回复。")

    return "\n".join(lines)


def _on_transform_tool_result(
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    **_: Any,
) -> str | None:
    """transform_tool_result hook — verify intermediate tool call results.

    Performs fast local checks (no LLM calls) on tool results. If issues
    are found, appends a warning block to the result string. Returns None
    if no issues or plugin disabled.

    Follows the same pattern as security-guidance's
    ``_on_transform_tool_result``.

    Returns the (possibly modified) result string, or None to leave unchanged.
    """
    # Check if plugin is disabled
    if _conf.is_plugin_disabled():
        return None

    # Also check legacy is_enabled() path
    if not _verifier.is_enabled():
        return None

    # Skip if no result or not a string
    if result is None or not isinstance(result, str):
        return None

    # Skip empty results
    if not result.strip():
        return None

    # Read language setting
    lang = _conf.get_language()

    # Run tool result verification
    warnings = _verifier.verify_tool_result(tool_name, args, result)
    if not warnings:
        return None

    # Build and append warning block
    warning_block = _verifier._format_tool_warning_block(warnings, lang=lang)

    logger.info(
        "self-verification: tool=%s, %d warning(s), session trace",
        tool_name,
        len(warnings),
    )

    return result + "\n" + warning_block


def register(ctx: Any) -> None:
    """Register self-verification hooks with the plugin context.

    Registers both ``transform_llm_output`` (output verification) and
    ``transform_tool_result`` (tool result verification) hooks.
    Follows the same registration pattern as security-guidance plugin.
    """
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
    logger.info(
        "self-verification plugin registered (v0.3.0): "
        "output verification + tool result verification"
    )
