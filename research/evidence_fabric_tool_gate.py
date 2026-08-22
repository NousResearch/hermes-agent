"""Task-8 decision record for optional model-facing exposure."""

MODEL_TOOL_DEFERRED = "MODEL_TOOL_DEFERRED"


def model_tool_go_no_go() -> tuple[bool, str]:
    """Return the safe exposure decision for the current Hermes runtime.

    Existing model-facing tool conventions do not provide a trusted,
    runtime-owned EvidenceScope and ResearchRun context to a compact tool
    handler. Model-controlled parameters therefore cannot be accepted as
    authority, so v1 retains the service API only.
    """
    return False, MODEL_TOOL_DEFERRED
