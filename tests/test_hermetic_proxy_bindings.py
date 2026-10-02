"""Every by-value ``getproxies`` binding must answer env-only after the hermetic blank.

trust-env clients bind ``urllib.request.getproxies`` by value at import time, so the
hermetic fixture's attribute patch on ``urllib.request`` cannot reach them —
``tests/conftest.py`` patches each such module explicitly instead. The imports below
run at collection, BEFORE any fixture executes, so each module's binding still holds
the original stdlib function — the same state a test module that imports the client
for real sees. The test then re-creates the leak shape with a synthetic system layer
(env reads blank, the macOS system reader returns the developer's proxy) and asserts
every installed binding answers env-only anyway. A client that binds ``getproxies``
without a conftest entry fails here instead of leaking into the fixtures around it.

The list below must stay in lockstep with the patch loop in ``tests/conftest.py``;
to re-derive it after adding a proxy-capable client, run
``grep -rn "import getproxies" <venv>/lib/python*/site-packages``.
"""

import urllib.request

import pytest

try:
    import httpx._utils as _httpx_utils
except ImportError:  # pragma: no cover - core dependency, absent only in stripped venvs
    _httpx_utils = None
try:
    import requests.utils as _requests_utils
except ImportError:  # pragma: no cover
    _requests_utils = None
try:
    import requests.compat as _requests_compat
except ImportError:  # pragma: no cover
    _requests_compat = None
try:
    import aiohttp.helpers as _aiohttp_helpers
except ImportError:  # pragma: no cover - homeassistant/sms/teams extra
    _aiohttp_helpers = None
try:
    import anthropic._utils._httpx as _anthropic_httpx
except ImportError:  # pragma: no cover - optional provider extra
    _anthropic_httpx = None
try:
    import httpx2._utils as _httpx2_utils
except ImportError:  # pragma: no cover - mcp/computer-use extra
    _httpx2_utils = None
try:
    import botocore.utils as _botocore_utils
except ImportError:  # pragma: no cover - bedrock extra (via boto3)
    _botocore_utils = None

# httpx and requests are core dependencies, so the loop must never come up empty.
_BINDING_MODULES = (
    ("httpx._utils", _httpx_utils),
    ("requests.utils", _requests_utils),
    ("requests.compat", _requests_compat),
    ("aiohttp.helpers", _aiohttp_helpers),
    ("anthropic._utils._httpx", _anthropic_httpx),
    ("httpx2._utils", _httpx2_utils),
    ("botocore.utils", _botocore_utils),
)


def test_every_by_value_getproxies_binding_is_env_only(monkeypatch):
    monkeypatch.setattr(urllib.request, "getproxies_environment", lambda: {})
    monkeypatch.setattr(
        urllib.request, "getproxies_macosx_sysconf",
        lambda: {"http": "http://127.0.0.1:7890"})

    # Sanity: the hermetic fixture's attribute patch is active here — without it the
    # loop below proves nothing about the fixture, only about the venv.
    assert urllib.request.getproxies() == {}

    installed = [(name, module) for name, module in _BINDING_MODULES
                 if module is not None]
    assert installed  # core clients; the check must not pass vacuously

    for name, module in installed:
        binding = getattr(module, "getproxies", None)
        assert binding is not None, (
            f"{name} no longer binds getproxies — re-derive the conftest patch list")
        leaked = binding()
        assert leaked == {}, (
            f"{name} binds getproxies by value and no conftest patch reaches "
            f"it — it still resolves the system proxy: {leaked}")
