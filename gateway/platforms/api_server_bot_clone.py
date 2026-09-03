"""Authenticated API handlers for cloning bot profiles between gateways."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

try:
    from aiohttp import web
except ImportError:
    web = None  # type: ignore[assignment]

logger = logging.getLogger("gateway.platforms.api_server")


def _http_routes(self) -> list[tuple[str, str, Any]]:
    return [
        ("GET", "/v1/bots/{profile_name}/clone", self._handle_bot_clone_download),
        ("POST", "/v1/bots/clone", self._handle_bot_clone_upload),
    ]


def _capabilities() -> dict[str, dict[str, str]]:
    return {
        "bot_clone_download": {
            "method": "GET",
            "path": "/v1/bots/{profile_name}/clone",
        },
        "bot_clone_upload": {"method": "POST", "path": "/v1/bots/clone"},
    }


def _bot_push_enabled(*, coerce_bool: Callable[..., bool]) -> bool:
    """Read the gateway owner's explicit remote-push policy."""
    try:
        from hermes_cli.config import cfg_get, load_config

        raw = cfg_get(
            load_config(), "gateway", "bot_sharing", "allow_push", default=False
        )
        return coerce_bool(raw, default=False)
    except Exception:
        return False


async def _handle_bot_clone_download(
    self,
    request: "web.Request",
    *,
    error_factory: Callable[..., dict[str, Any]],
    web_module: Any,
) -> "web.Response":
    """Return a bounded, credential-free clone of an owner-enabled bot."""
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    from hermes_cli.bot_transfer import export_bot_profile, profile_is_cloneable
    from hermes_cli.profiles import normalize_profile_name

    try:
        name = normalize_profile_name(request.match_info.get("profile_name", ""))
        if not profile_is_cloneable(name):
            return web_module.json_response(
                error_factory(
                    "Bot is not available for cloning.", code="bot_clone_unavailable"
                ),
                status=404,
            )

        def _export() -> tuple[bytes, str]:
            with tempfile.TemporaryDirectory(prefix="hermes_api_bot_pull_") as tmpdir:
                archive, bot_id = export_bot_profile(
                    name, str(Path(tmpdir) / f"{name}.tar.gz")
                )
                return archive.read_bytes(), bot_id

        payload, bot_id = await asyncio.to_thread(_export)
    except FileNotFoundError:
        return web_module.json_response(
            error_factory("Bot is not available for cloning.", code="bot_clone_unavailable"),
            status=404,
        )
    except ValueError as exc:
        return web_module.json_response(
            error_factory(str(exc), code="invalid_bot_clone"), status=400
        )
    except Exception:
        logger.exception("[%s] bot clone export failed", self.name)
        return web_module.json_response(
            error_factory("Bot clone export failed.", code="bot_clone_export_failed"),
            status=500,
        )

    return web_module.Response(
        body=payload,
        content_type="application/gzip",
        headers={
            "Content-Disposition": f'attachment; filename="{name}.tar.gz"',
            "X-Hermes-Bot-Id": bot_id,
            "X-Hermes-Profile-Name": name,
        },
    )


async def _handle_bot_clone_upload(
    self,
    request: "web.Request",
    *,
    error_factory: Callable[..., dict[str, Any]],
    web_module: Any,
) -> "web.Response":
    """Install an uploaded bot clone without overwriting any local bot."""
    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err
    if not self._bot_push_enabled():
        return web_module.json_response(
            error_factory(
                "This gateway does not accept pushed bot clones.",
                code="bot_clone_push_disabled",
            ),
            status=403,
        )
    if request.content_type not in {"application/gzip", "application/x-gzip"}:
        return web_module.json_response(
            error_factory(
                "Content-Type must be application/gzip.", code="invalid_content_type"
            ),
            status=415,
        )

    from hermes_cli.bot_transfer import import_bot_profile

    try:
        payload = await request.read()
        if not payload:
            raise ValueError("Bot clone archive is empty.")

        def _import() -> tuple[Path, str]:
            with tempfile.TemporaryDirectory(prefix="hermes_api_bot_push_") as tmpdir:
                archive = Path(tmpdir) / "bot.tar.gz"
                archive.write_bytes(payload)
                return import_bot_profile(
                    str(archive), name=(request.query.get("name") or "").strip() or None
                )

        profile_dir, bot_id = await asyncio.to_thread(_import)
        try:
            from hermes_cli.profiles import check_alias_collision, create_wrapper_script

            if not check_alias_collision(profile_dir.name):
                await asyncio.to_thread(create_wrapper_script, profile_dir.name)
        except Exception:
            logger.exception("Creating wrapper for cloned bot %s failed", profile_dir.name)
    except FileExistsError as exc:
        return web_module.json_response(
            error_factory(str(exc), code="bot_clone_conflict"), status=409
        )
    except (ValueError, FileNotFoundError) as exc:
        return web_module.json_response(
            error_factory(str(exc), code="invalid_bot_clone"), status=400
        )
    except Exception:
        logger.exception("[%s] bot clone import failed", self.name)
        return web_module.json_response(
            error_factory("Bot clone import failed.", code="bot_clone_import_failed"),
            status=500,
        )

    return web_module.json_response(
        {"object": "hermes.bot_clone", "name": profile_dir.name, "bot_id": bot_id},
        status=201,
    )
