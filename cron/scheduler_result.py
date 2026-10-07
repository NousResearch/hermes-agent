"""Classify cron turn results before their prose can mask a runtime hard stop."""
import logging

from agent.turn_failure_copy import is_max_iteration_handoff

logger = logging.getLogger("cron.scheduler")


def is_guardrail_halt(result: dict | None) -> bool:
    if not result:
        return False
    guardrail = result.get("guardrail")
    return result.get("turn_exit_reason") == "guardrail_halt" or (
        isinstance(guardrail, dict) and guardrail.get("action") in {"block", "halt"}
    )


def guardrail_halt_message(result: dict) -> str:
    """Use the controller's count/reason, never the assistant's account of its retries."""
    guardrail = result.get("guardrail")
    guardrail = guardrail if isinstance(guardrail, dict) else {}
    details = ", ".join(f"{key}={guardrail[key]}" for key in ("tool_name", "count") if key in guardrail)
    code = guardrail.get("code") or "unspecified"
    message = f"guardrail_halt: {code}"
    if details:
        message += f" ({details})"
    if guardrail.get("message"):
        message += f". {guardrail['message']}"
    return message


def _final_response_from_result(result: dict, job_id: str, job_name: str, AIAgent) -> str:
    """Return a deliverable response, rejecting agent and guardrail failures.

    ``run_conversation`` can synthesize a final response when a hard tool-loop guardrail stops
    the turn. That text explains the failure; it is not successful job output.
    """
    # If the agent itself reported failure (e.g. all retries exhausted on API errors, model abort, mid-run
    # interrupt), do not silently mark the job as successful. run_agent populates
    # `failed=True`/`completed=False` on these paths and may put the error into `final_response`, which
    # would otherwise be delivered as if it were the agent's reply and the job's `last_status` set to "ok".
    # A hard tool guardrail is another abnormal terminal even though the generic agent result remains
    # completed=True/failed=False for interactive surfaces. Cron must route its synthesized explanation
    # through the failure-delivery lane instead of recording completed/delivered.
    # Raise so the except handler below builds the proper failure tuple. (issue #17855)
    turn_exit_reason = str(result.get("turn_exit_reason") or "")
    final_response_text = (result.get("final_response") or "").strip()
    if is_guardrail_halt(result):
        raise RuntimeError(guardrail_halt_message(result))
    max_iteration_summary = is_max_iteration_handoff(result)
    if (
        result.get("failed") is True
        or (result.get("completed") is False and not max_iteration_summary)
    ):
        raise RuntimeError(result.get("error") or final_response_text or "agent reported failure")
    if max_iteration_summary:
        logger.warning(
            "Job '%s' reached the iteration limit but produced a final fallback response; "
            "delivering the response instead of failing the cron run",
            job_name)

    final_response = result.get("final_response", "") or ""
    # Repair model-mangled computer_use media paths before delivery (fail-open, as in gateway).
    if final_response:
        from gateway.media_repair import repair_explicit_computer_use_media_paths

        final_response = repair_explicit_computer_use_media_paths(
            final_response, result.get("messages", []))
    if final_response.strip() == "(No response generated)":
        final_response = ""
    # The "⚠️ No reply" turn-completion explainer would be delivered as a cron warning; detect it
    # via the same formatter and treat as empty so cron stays silent on abnormal empty turns.
    if final_response.strip() and turn_exit_reason:
        # Render every persistence-cause variant or cause-refined text slips through.
        _explainer_variants = []
        try:
            from hermes_state_errors import PERSISTENCE_ERROR_CAUSES as _causes
        except Exception:
            _causes = ("locked", "disk", "unknown")
        # The finalizer fills the model name into the explainer; render with the same name (and
        # the bare form) or the comparison below misses and the warning is delivered.
        _model = str(result.get("model") or "")
        for _cause in (None, *_causes):
            for _kwargs in ({"model": _model}, {}):
                try:
                    _variant = AIAgent._format_turn_completion_explanation(turn_exit_reason, _cause, **_kwargs)
                except TypeError:
                    try:
                        _variant = AIAgent._format_turn_completion_explanation(turn_exit_reason)
                    except Exception:
                        _variant = ""
                except Exception:
                    _variant = ""
                if _variant:
                    _explainer_variants.append(_variant.strip())
        if final_response.strip() in _explainer_variants:
            logger.info(
                "Job '%s': abnormal empty turn (%s) — suppressing explainer for cron delivery",
                job_id, turn_exit_reason)
            final_response = ""
    return final_response
