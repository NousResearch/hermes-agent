"""Profile surface of ``PluginContext``: ``profile_name``, and ``create_profile`` (a plugin creates a
profile that can answer its first turn, through the same body as the ``profiles.create`` RPC,
``hermes_cli.profile_mirror``).

Catalog rule 11 bars a plugin from reading ``.env`` / ``auth.json`` itself, so without this a
plugin-created profile either has NO provider or the plugin copies credentials around Hermes."""

from __future__ import annotations

from typing import Optional

from hermes_cli.plugins_loader import _plugin_home_scope


class PluginProfileCtxMixin:
    """Mixed into :class:`hermes_cli.plugins.PluginContext` (``_manager``)."""

    @property
    def profile_name(self) -> str:
        """Active profile name (``"default"``, the ``~/.hermes/profiles/<name>`` id, or ``"custom"``),
        derived from ``HERMES_HOME`` — not ``_cli_ref``, which is None outside the interactive CLI —
        so gateway and kanban workers get it too."""
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name()
        except Exception:
            return "default"

    def create_profile(
        self, name: str, *, description: Optional[str] = None, soul: Optional[str] = None,
        model: Optional[str] = None, provider: Optional[str] = None,
        mirror_credentials: bool = True, share_auth: bool = False, no_alias: bool = False,
    ) -> dict:
        """Create a fresh profile ``name`` and return the RPC's receipt:
        ``{name, path, soul_written, model_set, mirrored: {env, auth, model_inherited, voice}}``.

        ``mirror_credentials`` (default True) copies this plugin's profile's ``.env`` (messaging
        channel tokens stripped), ``auth.json`` (single-use OAuth grants stripped; skipped with
        ``share_auth``) and voice sections into the new profile once, and inherits its model unless
        ``model`` + ``provider`` pin one. The receipt carries booleans only, never a value. Raises
        ``ValueError`` / ``FileExistsError`` exactly as ``hermes profile create`` would (invalid or
        taken name, live-multiplexer refusal); nothing is created then."""
        from hermes_cli.profile_mirror import create_profile_with_credentials
        # Source = the profile that loaded this plugin, not whichever home the calling thread holds.
        with _plugin_home_scope(self._manager.home_path):
            return create_profile_with_credentials(
                name, description=description, soul=soul, model=model or "", provider=provider or "",
                mirror_credentials=mirror_credentials, share_auth=share_auth, no_alias=no_alias)
