"""Solstice invariants under an httpx-free runtime: provider discovery imports every bundled
plugin, and the pm runtime (``pm/launch.py``) runs its CLI in a venv that deliberately ships
without ``httpx`` — a bundled plugin whose module import pulls ``httpx`` fails to load there
("Failed to load bundled provider plugin solstice: No module named 'httpx'")."""

from __future__ import annotations

import sys

import providers


class _HttpxBlocker:
    """Meta-path blocker that makes ``httpx`` unimportable, as in the pm runtime's venv."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "httpx" or fullname.startswith("httpx."):
            raise ModuleNotFoundError(f"No module named {fullname!r} (blocked: httpx-free runtime)", name=fullname)
        return None


def test_discovery_registers_solstice_without_httpx_and_the_client_builds_on_first_use():
    blocker = _HttpxBlocker()
    for name in [n for n in sys.modules if n == "httpx" or n.startswith(
            ("httpx.", "agent.gemini_native_adapter", "plugins.model_providers.solstice"))]:
        del sys.modules[name]
    providers._REGISTRY.pop("solstice", None)
    providers._discovered = False  # a fresh sweep imports every bundled plugin again
    sys.meta_path.insert(0, blocker)
    try:
        assert "solstice" in [p.name for p in providers.list_providers()]
    finally:
        sys.meta_path.remove(blocker)
    # httpx back: the deferred client class builds and carries the per-user-quota methods.
    client = providers.get_provider_profile("solstice").create_client(api_key="ya29.test")
    assert client.GENERATE_METHOD == "generateContentPerUserQuota"
    assert client.STREAM_METHOD == "streamGenerateContentPerUserQuota"
