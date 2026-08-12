"""Behavior tests for Photon raw-URL outbound routing (issue: markdown 500s).

The iMessage markdown builder enables data detection inside spectrum-ts. On
some IMAgentKit sends, that path returns a 500 when the message contains a raw
URL. The sidecar keeps markdown rendering for URL-free messages, but must use
plain text for messages containing URLs so iMessage can auto-link them without
hitting the data-detection failure path.

The routing decision lives in
``plugins/platforms/photon/sidecar/send-format.mjs`` (imported by index.mjs's
``/send`` handler). These tests *execute* that real module under node and
assert the chosen builder for representative payloads — they do not read the
sidecar source.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Tuple

import pytest

_MODULE = Path("plugins/platforms/photon/sidecar/send-format.mjs").resolve()

_CASES: Dict[str, Tuple[str, str, str]] = {
    # name: (format, text, expected builder)
    "markdown_without_url_keeps_markdown": (
        "markdown", "**bold** and `code`", "markdown",
    ),
    "markdown_with_https_url_falls_back_to_text": (
        "markdown", "see **this**: https://example.com/a?b=1", "text",
    ),
    "markdown_with_http_url_falls_back_to_text": (
        "markdown", "http://example.com", "text",
    ),
    "markdown_link_syntax_also_falls_back_to_text": (
        "markdown", "[docs](https://example.com/docs)", "text",
    ),
    "markdown_with_uppercase_scheme_falls_back_to_text": (
        "markdown", "HTTPS://EXAMPLE.COM is loud", "text",
    ),
    "markdown_bare_domain_without_scheme_keeps_markdown": (
        # Only scheme'd URLs trip iMessage's data-detection 500; a bare domain
        # is ordinary text and must keep markdown rendering.
        "markdown", "ask example.com about *this*", "markdown",
    ),
    "text_format_stays_text": ("text", "plain message", "text"),
    "text_format_with_url_stays_text": (
        "text", "https://example.com", "text",
    ),
}


@pytest.fixture(scope="module")
def verdicts() -> Dict[str, str]:
    """Run every case through the real send-format module in one node call."""
    harness = (
        f"import {{ chooseSendFormat }} from {json.dumps(_MODULE.as_uri())};\n"
        "const chunks = [];\n"
        "process.stdin.on('data', (c) => chunks.push(c));\n"
        "process.stdin.on('end', () => {\n"
        "  const cases = JSON.parse(Buffer.concat(chunks).toString('utf-8'));\n"
        "  const out = {};\n"
        "  for (const [name, [format, text]] of Object.entries(cases)) {\n"
        "    out[name] = chooseSendFormat(format, text);\n"
        "  }\n"
        "  process.stdout.write(JSON.stringify(out));\n"
        "});\n"
    )
    payload = {name: [fmt, text] for name, (fmt, text, _) in _CASES.items()}
    run = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        input=json.dumps(payload),
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_send_builder_selection(name: str, verdicts: Dict[str, str]) -> None:
    _, _, expected = _CASES[name]
    assert verdicts[name] == expected
