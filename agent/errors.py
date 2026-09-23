import json
import re


class SSLConfigurationError(Exception):
    """Raised when SSL/TLS certificate bundle configuration fails."""


class EmptyStreamError(RuntimeError):
    """Raised when a provider closes a stream without yielding a response."""


class MoAPresetNotFoundError(ValueError):
    """Raised when a persisted MoA preset no longer exists in config."""


# ── Provider wire-format parse errors ───────────────────────────────────
#
# The OpenAI and Anthropic SDKs both parse streaming payloads with ``jiter``
# (a Rust extension). Malformed provider bytes surface as a plain
# ``ValueError`` whose message ends with ``at line N column M`` — jiter's
# signature for *every* parse failure: ``key must be a string at line 1
# column 338``, ``expected value at line 1 column 6``, ``trailing comma at
# line 1 column 8``, and so on.
#
# Such an error is provider wire damage, not local request validation, so it
# must follow the transient retry path (and let fallback providers run)
# rather than aborting the turn. Matching the full signature instead of one
# literal message matters: hardcoding a single phrase means the next jiter
# message aborts the conversation with the message still streaming.
_JITER_ERROR_SUFFIX = re.compile(r" at line \d+ column \d+$")


def is_provider_stream_parse_error(error: BaseException) -> bool:
    """True when ``error`` is a native streaming-parser (jiter) failure.

    Tight enough to never claim a local ``ValueError``: the message must end
    with jiter's ``at line N column M`` position suffix. ``json.JSONDecodeError``
    is a ``ValueError`` subclass carrying its own position format and is
    excluded; the Unicode errors are local encoding problems, not wire damage.
    """
    if not isinstance(error, ValueError):
        return False
    if isinstance(error, (UnicodeEncodeError, json.JSONDecodeError)):
        return False
    return bool(_JITER_ERROR_SUFFIX.search(str(error).strip().lower()))
