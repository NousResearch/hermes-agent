"""Provision the webauthn extra for a configured dashboard passkey block.

Fresh installs carry PM's base extras only, so a ``dashboard.webauthn`` block
configured AFTER the install came up (the normal order: install, then set
``rp_id``/``rp_name``/``origins``) left the passkey provider skipping with
``No module named 'fido2'`` on the next dashboard start — the classic
chicken-and-egg: the config activates the feature, but the feature's
dependency is not installed.

This unions the `webauthn` extra into PM's recorded dependency graph at
dashboard start, before web_server's import graph registers providers — the
same self-installing-extra shape the platform SDKs use (cf.
``dingtalk_auth``). The update-repair path (#124228,
``_configured_features_missing_deps``) covers the same need across updates.
"""

from __future__ import annotations

import importlib.util

def _logger():
    import logging

    return logging.getLogger("hermes_cli.dashboard_webauthn_extra")


def _webauthn_configured(section: dict) -> bool:
    """The provider's own activation gate (fail-loud trio from `pm` config:
    rp_id + rp_name + origins — anything missing and it stays skipped)."""
    return bool(
        str(section.get("rp_id") or "").strip()
        and str(section.get("rp_name") or "").strip()
        and section.get("origins")
    )


def maybe_sync_webauthn_extra() -> None:
    """No-op unless dashboard.webauthn is fully configured and fido2 is absent.

    Never raises: a dashboard start must proceed on the password gate when
    anything here fails, with the provider's own skip hint still in play.
    """
    log = _logger()
    try:
        from hermes_cli.config import load_config_readonly

        section = ((load_config_readonly().get("dashboard") or {}).get("webauthn")) or {}
        if not isinstance(section, dict) or not _webauthn_configured(section):
            return  # not configured (or deliberately incomplete): stays inert
        if importlib.util.find_spec("fido2") is not None:
            return  # extra already present
        import pm

        print("→ dashboard.webauthn is configured; installing its 'webauthn' "
              "extra (fido2)…", flush=True)
        pm.sync_venv(["webauthn"], explicit=True)
        print("  ✓ passkey support ready", flush=True)
    except Exception as exc:  # health: allow BLE001 -- startup must not hard-fail on the extra
        log.warning("could not provision the dashboard.webauthn extra: %s; "
                    "the passkey provider will skip with its own hint", exc)