"""The dashboard bundle's MIME types must not come from the host's registry.

On a host whose registry has no ``.js`` mapping (seen on Windows), Python's ``mimetypes`` reports
``text/plain`` for the dashboard's ES modules, Starlette's ``StaticFiles`` serves them with that
type, and the browser refuses to execute the module ("Expected a JavaScript module script but the
server responded with a MIME type of 'text/plain'") — :9119 renders blank. The contract pinned
here: the extensions of the served bundle answer with their real types whatever the host said.
"""

import mimetypes

import pytest

# Hardcoded on purpose: this is the contract with the browser, not a copy of the implementation.
EXPECTED = (
    (".js", "application/javascript"),
    (".mjs", "application/javascript"),
    (".css", "text/css"),
)


@pytest.fixture
def host_without_bundle_types(monkeypatch):
    """Pretend the host knows nothing useful: every bundle extension answers ``text/plain``."""
    mimetypes.init()
    for ext, _mime in EXPECTED:
        monkeypatch.setitem(mimetypes.types_map, ext, "text/plain")


def test_importing_the_spa_module_registers_the_types(host_without_bundle_types):
    """The wiring: the module that mounts the bundle must register its types itself.

    Registration is process-global and idempotent, so this is the first test in the file — the
    import has to happen after the host has been made unhelpful.
    """
    try:
        from hermes_cli import web_server_dashboard  # noqa: F401
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    for ext, mime in EXPECTED:
        assert mimetypes.guess_type("app" + ext)[0] == mime


def test_registration_replaces_a_broken_host_registry(host_without_bundle_types):
    """The helper itself stays correct no matter when it runs (order-independent)."""
    from hermes_cli import static_mime

    static_mime.register_static_mime_types()

    for ext, mime in EXPECTED:
        assert mimetypes.guess_type("app" + ext)[0] == mime
