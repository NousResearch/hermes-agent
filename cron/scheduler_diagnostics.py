"""Private run-document diagnostics, kept separate from delivery summaries."""

from traceback import format_exception

from agent.redact import redact_sensitive_text


def format_run_error(exc: BaseException) -> str:
    """Retain chained causes without capturing locals or exposing URL credentials."""
    traceback_text = redact_sensitive_text(
        "".join(format_exception(exc)), force=True, redact_url_credentials=True,
    )
    return f"## Error\n\n```\n{traceback_text}\n```\n"
