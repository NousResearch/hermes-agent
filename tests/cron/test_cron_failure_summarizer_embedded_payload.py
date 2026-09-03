"""_summarize_cron_failure_for_delivery must classify on the error's first
line (the exception line), never on substrings embedded in later payload
lines.

Field-reported (#99988): a monitor-job creation bug stored the whole script
SOURCE in ``monitor_script`` (which must hold a filename), so every run died
with ``[Errno 63] File name too long: '/home/user/.hermes/scripts/#!/bin/bash
... Authorization: Bearer $TOKEN ...'`` — OSError embeds the whole too-long
"filename", newlines included, and the embedded curl header's word
"Authorization" tripped the auth substring match. The operator was delivered
"provider authentication error" for a crash that never opened a provider
socket and was told to debug the wrong subsystem. Replacing the single word
"Authorization" with "Xuthorization" flipped the label — the classifier was
reading the crashed script's own source code as an error signature.

The no-agent script path already guards this class via its mode gate
(test_no_agent_failure_never_blamed_on_a_provider); an agent-mode job's error
text can embed the same attacker-content-shaped strings (OSError filenames,
captured subprocess output, tool payloads in tracebacks), so the provider
branches match the FIRST line only: provider errors arrive as single-line
``str(exc)`` ("Error code: 401 - ...", "httpx.ReadTimeout: ..."), while
anything on later lines is payload, not signature.

One escape hatch remains for a first-line bound: ``str(OSError)`` embeds the
filename repr with newlines escaped to a literal ``\\n``, so the ENTIRE
payload stays on one line — ``str(OSError(63, "File name too long",
script_source))`` is a single line carrying every word of the embedded
script. The scheduler's capture is ``f"{type(e).__name__}: {str(e)}"``, so
the delivered error text is that single line. OSError signature lines
(``[Errno N]``, with or without a type-name prefix) are therefore exempt
from the provider scan altogether: provider errors come from the HTTP
stack and never carry ``[Errno N]``, while an OSError names the
system/IO layer by construction (#99988 review).
"""

from cron.scheduler import _summarize_cron_failure_for_delivery


def _agent_job():
    return {"name": "watch-manifest", "id": "m111", "no_agent": False}


def test_file_name_too_long_embedding_auth_wording_not_provider_auth():
    """#99988 replay: an OSError whose embedded filename is a whole script
    whose source contains "Authorization" must not be labeled a provider
    authentication error."""
    error = (
        "[Errno 63] File name too long: '/home/user/.hermes/scripts/#!/bin/bash\n"
        "# watch the deployment manifest\n"
        'TOKEN=$(curl -s "https://auth.example/token")\n'
        'MANIFEST=$(curl -s -H "Authorization: Bearer $TOKEN" '
        "https://registry.example/manifests/latest)"
    )
    msg = _summarize_cron_failure_for_delivery(_agent_job(), error)
    assert "provider authentication error" not in msg
    assert "provider" not in msg.lower()
    # The generic cleaner names what actually failed.
    assert "File name too long" in msg


def test_embedded_timeout_wording_in_payload_not_provider_timeout():
    """A crash whose payload lines merely mention "timeout" must not be
    rewritten into "provider timeout / fallback chain" prose."""
    error = (
        "OSError: [Errno 5] Input/output error: '/var/lib/hermes/state.db'\n"
        "# retry loop: sleep 30 then retry on timeout after 30s"
    )
    msg = _summarize_cron_failure_for_delivery(_agent_job(), error)
    assert "provider timeout" not in msg
    assert "fallback chain" not in msg.lower()
    assert "Input/output error" in msg


def test_embedded_rate_limit_wording_in_payload_not_rate_limit():
    """A script's captured log line "HTTP 429" on a later line is payload,
    not the crash's signature."""
    error = (
        "RuntimeError: script produced invalid JSON for manifest\n"
        "upstream log: HTTP 429 Too Many Requests (backoff applied)"
    )
    msg = _summarize_cron_failure_for_delivery(_agent_job(), error)
    assert "provider rate limit" not in msg
    assert "provider" not in msg.lower()
    assert "invalid JSON" in msg


def test_first_line_signature_in_a_multiline_error_still_classifies():
    """Genuine provider errors can carry payload on later lines; the
    signature on the FIRST line must still classify."""
    error = (
        "Error code: 429 - {'error': {'message': 'rate limit exceeded'}}\n"
        "request-id: req_abc123\n"
        "retry-after: 60"
    )
    msg = _summarize_cron_failure_for_delivery(_agent_job(), error)
    assert "provider rate limit" in msg


def test_first_line_401_still_classifies_as_provider_auth():
    error = "Error code: 401 - {'error': {'message': 'Invalid API key'}}"
    msg = _summarize_cron_failure_for_delivery(_agent_job(), error)
    assert "provider authentication error" in msg


def _script_source():
    return (
        "#!/bin/bash\n"
        'TOKEN=$(curl -s "https://auth.example/token")\n'
        'MANIFEST=$(curl -s -H "Authorization: Bearer $TOKEN" '
        "https://registry.example/manifests/latest)"
    )


def test_oserror_str_repr_single_line_not_provider_auth():
    """Real ``str(OSError)`` replay: the filename repr keeps newlines
    escaped on ONE line, so the first-line bound alone cannot keep the
    embedded "Authorization" out of the scan — OSError signature lines are
    exempt from the provider branches (#99988 review)."""
    # Built from a real OSError, not hand-typed text: str(OSError) embeds
    # the filename repr, which escapes newlines to a literal "\n".
    error = f"OSError: {OSError(63, 'File name too long', _script_source())}"
    # Pin the premise: this is the single-line repr form. If this ever
    # becomes a multi-line string, the fixture stopped reproducing the
    # reported shape and the regression it guards is gone.
    assert len(error.splitlines()) == 1
    assert "Authorization" in error
    msg = _summarize_cron_failure_for_delivery(_agent_job(), error)
    assert "provider authentication error" not in msg
    assert "provider" not in msg.lower()
    assert "File name too long" in msg


def test_oserror_str_without_type_prefix_same_treatment():
    """A caller storing bare ``str(e)`` (no type-name prefix) gets the same
    exemption: the signature is the leading ``[Errno N]`` token."""
    error = str(OSError(63, "File name too long", _script_source()))
    assert len(error.splitlines()) == 1
    assert "Authorization" in error
    msg = _summarize_cron_failure_for_delivery(_agent_job(), error)
    assert "provider" not in msg.lower()
    assert "File name too long" in msg


def test_oserror_connect_timeout_verbatim_not_provider_timeout():
    """An OS-level connect timeout names the socket layer, not a provider
    response: no "provider timeout / fallback chain exhausted" claim, the
    verbatim error text carries the signal instead."""
    error = "TimeoutError: [Errno 110] Connection timed out"
    msg = _summarize_cron_failure_for_delivery(_agent_job(), error)
    assert "provider timeout" not in msg
    assert "fallback chain" not in msg.lower()
    assert "Connection timed out" in msg
