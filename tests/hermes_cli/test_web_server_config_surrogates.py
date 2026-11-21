"""``GET /api/config`` must render a config value that YAML decoded into lone surrogate halves.

YAML's ``\\u`` escape takes four hex digits, so an emoji written as ``"\\uD83E\\uDD19"`` in a
double-quoted scalar parses to the two *lone* surrogates U+D83E U+DD19 rather than U+1F919. That is a
valid Python ``str`` and the file itself is valid UTF-8, but ``JSONResponse`` renders with
``json.dumps(..., ensure_ascii=False).encode("utf-8")``, which refuses them — so ``/api/config``
answered 500 for as long as the value sat in ``config.yaml``, and every config-reading pane in the
Desktop retried on a loop (#150918). The pairing halves are recombined here; a half that stands alone
cannot be encoded, so it becomes U+FFFD instead of failing the whole response.
"""

import pytest


ESCAPED_YAML = (
    "agent:\n"
    "  personalities:\n"
    '    surfer: "Cowabunga! \\uD83E\\uDD19"\n'
    '    broken: "half an emoji \\uD83E"\n'
)

# The same value with the emoji spelled as one code point: the control case, so a 500 on the fixture
# above can only come from the escaped spelling.
PLAIN_YAML = (
    "agent:\n"
    "  personalities:\n"
    '    surfer: "Cowabunga! \U0001F919"\n'
)


@pytest.fixture()
def client(_isolate_hermes_home):
    from starlette.testclient import TestClient

    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    # ``raise_server_exceptions=False`` is the point: the regression is a rendered 500, and asserting
    # the status code is what names the user-visible failure.
    test_client = TestClient(app, raise_server_exceptions=False)
    test_client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return test_client


def _write_config(text: str) -> None:
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(text, encoding="utf-8")


def test_get_config_renders_a_plain_emoji_unchanged(client):
    _write_config(PLAIN_YAML)

    resp = client.get("/api/config")

    assert resp.status_code == 200, resp.text
    assert resp.json()["agent"]["personalities"]["surfer"] == "Cowabunga! \U0001F919"


def test_get_config_renders_a_surrogate_escaped_value(client):
    _write_config(ESCAPED_YAML)

    resp = client.get("/api/config")

    assert resp.status_code == 200, resp.text
    personalities = resp.json()["agent"]["personalities"]
    # The pair spells U+1F919 again, so the value the user wrote survives the round trip.
    assert personalities["surfer"] == "Cowabunga! \U0001F919"
    # A lone half has no code point to recombine into; the response still renders.
    assert personalities["broken"] == "half an emoji \ufffd"
