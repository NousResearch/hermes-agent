"""Profile-scope probe plugin (test fixture).

Backend ``plugin_api.py`` for a dashboard/desktop plugin that reads a
profile-scoped credential — the shape every memory-provider panel has
(``hermes-memory-ui`` reads Honcho's config through
``plugins.memory.honcho.client``, which resolves ``HERMES_HONCHO_HOST`` and
the API key via ``agent.secret_scope.get_secret``).

Lives under ``tests/fixtures/plugins/`` so it is never shipped; the
``_install_probe_plugin`` fixture in
``tests/hermes_cli/test_plugin_api_profile_scope.py`` copies it into the
per-test ``HERMES_HOME`` and enables it, so tests can hit
``/api/plugins/profile-scope-probe/probe``.
"""

from fastapi import APIRouter

from agent.secret_scope import current_secret_scope, get_secret, is_multiplex_active

router = APIRouter()


@router.get("/probe")
async def probe():
    """Resolve a credential exactly the way a plugin backend does."""
    return {
        "secret": get_secret("PLUGIN_PROBE_SECRET", "unset"),
        "multiplex_active": is_multiplex_active(),
        "scoped": current_secret_scope() is not None,
    }
