"""
Multi-account Weixin (personal WeChat) support.

A single Hermes gateway can already sign in to one personal WeChat
account via the iLink Bot API (configured through the ``weixin`` key
in ``config.yaml`` or the ``WEIXIN_TOKEN`` + ``WEIXIN_ACCOUNT_ID``
env vars).  This module adds support for **additional** WeChat
accounts that share the same iLink endpoint.

Each additional account is identified by a stable ``account_id``
string and persists its credentials under
``~/.hermes/weixin/accounts/<account_id>.json`` (see
``gateway.platforms.weixin.save_weixin_account``).  At gateway start
time we discover all such files and:

1. Register one extra ``BasePlatformAdapter`` per persisted account
   via the ``platform_registry`` (named ``weixin:<account_id>``).
2. Ensure the corresponding ``Platform`` enum pseudo-member exists
   (via ``Platform._missing_``, which already accepts registry-known
   values).
3. Add a matching ``PlatformConfig`` to ``config.platforms`` so the
   gateway runner iterates over every account like any other enabled
   platform.

Design choices (cross-checked against ``AGENTS.md``):

* **Extend, don't duplicate.**  We reuse ``WeixinAdapter`` and the
  ``platform_registry`` machinery; we do not introduce a new
  ``WeixinMultiAccountAdapter`` wrapper class.
* **No new non-secret env vars.**  All multi-account configuration
  flows through the existing ``~/.hermes/weixin/accounts/`` JSON
  files (which are already the persistence path for ``qr_login``).
* **Cache- and invariant-safe.**  Each registered adapter runs its
  own long-poll loop; nothing mutates the system prompt or message
  history of a sibling adapter.
* **E2E testable.**  ``register_persisted_weixin_accounts`` only
  depends on the filesystem and ``platform_registry``; tests build
  a real ``HERMES_HOME`` tree under ``tmp_path`` and verify the
  registry entries and ``PlatformConfig`` rows that are created.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Reuse the on-disk layout from gateway.platforms.weixin without
# importing the module at import time (avoids the heavy aiohttp /
# cryptography imports for tools that just want the discovery helper).
_ACCOUNT_SUBDIR = Path("weixin") / "accounts"


def _account_dir(hermes_home: str) -> Path:
    path = Path(hermes_home) / _ACCOUNT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_persisted_weixin_accounts(hermes_home: str) -> List[str]:
    """Return all Weixin account ids persisted under ``hermes_home``.

    Reads ``~/.hermes/weixin/accounts/*.json`` and returns the
    list of account ids (file stem).  Skips companion files such
    as ``<id>.context-tokens.json`` and silently drops malformed
    JSON files (logging a warning).
    """
    accounts_dir = Path(hermes_home) / _ACCOUNT_SUBDIR
    if not accounts_dir.is_dir():
        return []
    ids: List[str] = []
    for child in sorted(accounts_dir.iterdir()):
        if not child.is_file() or child.suffix != ".json":
            continue
        try:
            data = json.loads(child.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "weixin-multi: skipping malformed account file %s: %s",
                child.name,
                exc,
            )
            continue
        if isinstance(data, dict) and data.get("token"):
            ids.append(child.stem)
    return ids


def _load_persisted_account(
    hermes_home: str, account_id: str
) -> Optional[Dict[str, Any]]:
    """Return the persisted account payload for ``account_id`` or None."""
    path = Path(hermes_home) / _ACCOUNT_SUBDIR / f"{account_id}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(
            "weixin-multi: failed to read account %s: %s", account_id, exc
        )
        return None
    return data if isinstance(data, dict) else None


def _build_extra_platform_config(
    account_id: str,
    persisted: Dict[str, Any],
    base_config: Any,
) -> Any:
    """Build a fresh ``PlatformConfig`` for one persisted account.

    Mirrors the single-account ``WeixinAdapter.__init__`` contract:
    ``token`` and ``base_url`` come from the persisted JSON; every
    other field inherits from ``base_config`` (e.g. ``dm_policy``,
    ``allow_from``, etc.).  The result is what we hand to
    ``WeixinAdapter(config)``.
    """
    from gateway.config import PlatformConfig

    token = str(persisted.get("token") or "").strip()
    base_url = str(persisted.get("base_url") or "").strip().rstrip("/")

    extra: Dict[str, Any] = {}
    if hasattr(base_config, "extra") and isinstance(base_config.extra, dict):
        # Copy shared runtime knobs (timeouts, policies, allowlists)
        # from the base config; never carry over ``account_id`` or
        # ``token`` because those differ per account.
        for key, value in base_config.extra.items():
            if key in {"account_id", "token", "base_url", "cdn_base_url"}:
                continue
            extra[key] = value
    extra["account_id"] = account_id
    if base_url:
        extra["base_url"] = base_url

    return PlatformConfig(
        enabled=True,
        token=token,
        home_channel=getattr(base_config, "home_channel", None),
        reply_to_mode=getattr(base_config, "reply_to_mode", "first"),
        gateway_restart_notification=getattr(
            base_config, "gateway_restart_notification", True
        ),
        extra=extra,
    )


def register_persisted_weixin_accounts(
    config: Any,
    hermes_home: str,
    *,
    primary_account_id: str = "",
) -> List[str]:
    """Register one extra Weixin account per persisted JSON file.

    ``config`` is the live ``GatewayConfig`` instance owned by the
    gateway runner.  This function mutates it in two places:

    * ``config.platforms[<entry_name>] = PlatformConfig(...)`` - the
      gateway runner iterates ``config.platforms`` and creates an
      adapter per enabled entry; without this row the runner would
      skip the extra account.
    * ``platform_registry.register(PlatformEntry(...))`` - so
      ``_create_adapter`` finds a factory and produces a working
      adapter for the dynamic ``weixin:<account_id>`` value.

    Returns the list of platform names registered (e.g.
    ``["weixin:work", "weixin:personal"]``).  The primary account
    (``primary_account_id`` or the one configured in
    ``base_config.extra['account_id']``) is **not** re-registered;
    it is already served by the built-in ``Platform.WEIXIN`` adapter.
    """
    try:
        from gateway.config import Platform  # local import keeps import-cheap
        from gateway.platform_registry import platform_registry, PlatformEntry
    except Exception as exc:  # pragma: no cover - registry missing
        logger.debug("weixin-multi: platform registry unavailable: %s", exc)
        return []

    try:
        from gateway.platforms.weixin import WeixinAdapter, check_weixin_requirements
    except Exception as exc:  # pragma: no cover - weixin adapter missing
        logger.debug("weixin-multi: weixin adapter unavailable: %s", exc)
        return []

    if not check_weixin_requirements():
        logger.debug("weixin-multi: skipping, weixin requirements not met")
        return []

    # Determine which account the gateway already serves via the
    # built-in WEIXIN platform.
    base_config = config.platforms.get(Platform.WEIXIN)
    if base_config is None and not primary_account_id:
        # No primary weixin platform at all - nothing to clone from.
        # Still allow registration of extras, but they'll inherit
        # from an empty base.
        base_config = None

    if not primary_account_id and base_config is not None:
        primary_account_id = str(
            (base_config.extra or {}).get("account_id") or ""
        ).strip()

    account_ids = list_persisted_weixin_accounts(hermes_home)
    registered: List[str] = []
    for account_id in account_ids:
        if account_id == primary_account_id:
            # Already served by Platform.WEIXIN; no duplicate.
            continue
        persisted = _load_persisted_account(hermes_home, account_id)
        if not persisted or not persisted.get("token"):
            continue

        # Inherit shared knobs from the primary config when available,
        # otherwise build a barebones config (token + account_id only).
        if base_config is not None:
            extra_cfg = _build_extra_platform_config(
                account_id, persisted, base_config
            )
        else:
            extra_cfg = _build_extra_platform_config(
                account_id, persisted, _empty_platform_config()
            )
        entry_name = f"weixin:{account_id}"

        def _factory(
            _cfg: Any = extra_cfg, _account_id: str = account_id
        ) -> Any:
            # WeixinAdapter always reads its account_id out of
            # config.extra; we already set it on ``extra_cfg`` above.
            return WeixinAdapter(_cfg)

        # Register FIRST so Platform(entry_name) -> _missing_ -> registry
        # lookup succeeds and the dynamic enum member exists.
        platform_registry.register(
            PlatformEntry(
                name=entry_name,
                label=f"Weixin ({account_id})",
                adapter_factory=_factory,
                check_fn=check_weixin_requirements,
                validate_config=lambda c: bool(getattr(c, "token", None)),
                required_env=[],
                install_hint="",
                source="builtin",
                platform_hint="",
            )
        )
        # Make the dynamic Platform enum member so config.platforms
        # can use it as a key.
        plat = Platform(entry_name)
        config.platforms[plat] = extra_cfg
        registered.append(entry_name)
        logger.info(
            "weixin-multi: registered extra Weixin account %s",
            account_id,
        )

    return registered


def unregister_persisted_weixin_accounts(
    config: Any, names: List[str]
) -> None:
    """Best-effort cleanup helper for tests / restart paths.

    Removes both the ``platform_registry`` entry and the matching
    ``PlatformConfig`` row so a subsequent call to
    ``register_persisted_weixin_accounts`` starts from a clean slate.
    """
    if not names:
        return
    try:
        from gateway.config import Platform
        from gateway.platform_registry import platform_registry
    except Exception:
        return
    for name in names:
        platform_registry.unregister(name)
        try:
            plat = Platform(name)
        except Exception:
            continue
        config.platforms.pop(plat, None)


def _empty_platform_config() -> Any:
    """Minimal fallback used when no primary weixin ``PlatformConfig`` exists."""
    from gateway.config import PlatformConfig

    return PlatformConfig(
        enabled=True,
        token="",
        reply_to_mode="first",
        gateway_restart_notification=True,
        extra={},
    )