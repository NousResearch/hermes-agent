"""Discord media delivery methods for :class:`DiscordAdapter`."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.base import BasePlatformAdapter, SendResult
from tools.url_safety import async_is_safe_url

from .outbound_image_fetch import (
    _DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES,
    _DISCORD_IMAGE_DOWNLOAD_BUDGET_CONTEXT,
    _DISCORD_IMAGE_DOWNLOAD_MAX_BYTES,
    _DISCORD_IMAGE_MAX_REDIRECTS,
    _DISCORD_IMAGE_REDIRECT_STATUSES,
    _DiscordImageDownloadBudget,
    _discord_image_extension_from_bytes,
    _read_response_bytes_bounded,
)


logger = logging.getLogger("plugins.platforms.discord.adapter")

discord = sys.modules.get("discord")

def _create_discord_image_http_client(proxy_url: Optional[str] = None) -> Any:
    """Forward the default client factory without binding adapter.py early."""
    from . import adapter

    return adapter._create_discord_image_http_client(proxy_url)


async def _read_url_image_with_redirect_guard(
    client: Any,
    url: str,
    *,
    timeout: Any,
    request_kwargs: Dict[str, Any],
    download_budget: Any = None,
) -> Tuple[int, bytes, Dict[str, str]]:
    """Delegate fetching while retaining moved-function patch globals."""
    from .outbound_image_fetch import (
        _read_url_image_with_redirect_guard as read_impl,
    )

    return await read_impl(
        client,
        url,
        timeout=timeout,
        request_kwargs=request_kwargs,
        download_budget=download_budget,
        async_is_safe_url_fn=_patchable_dependency("async_is_safe_url"),
        max_bytes=_patchable_dependency("_DISCORD_IMAGE_DOWNLOAD_MAX_BYTES"),
        read_response_bytes_fn=_patchable_dependency("_read_response_bytes_bounded"),
        redirect_statuses=_patchable_dependency("_DISCORD_IMAGE_REDIRECT_STATUSES"),
        max_redirects=_patchable_dependency("_DISCORD_IMAGE_MAX_REDIRECTS"),
    )

class DiscordMediaSendMixin:
    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local image file natively as a Discord file attachment."""
        try:
            return await self._send_file_attachment(chat_id, image_path, caption)
        except FileNotFoundError:
            return SendResult(success=False, error=f"Image file not found: {image_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send local image, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_image_file(chat_id, image_path, caption, reply_to, metadata=metadata)

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image natively as a Discord file attachment."""
        discord = _patchable_dependency("discord")
        async_is_safe_url = _patchable_dependency("async_is_safe_url")
        create_image_http_client = _patchable_dependency(
            "_create_discord_image_http_client"
        )
        read_image_with_redirect_guard = _patchable_dependency(
            "_read_url_image_with_redirect_guard"
        )
        image_extension_from_bytes = _patchable_dependency(
            "_discord_image_extension_from_bytes"
        )

        if not self._client:
            return SendResult(success=False, error="Not connected")

        if not await async_is_safe_url(image_url):
            logger.warning("[%s] Blocked unsafe image URL during Discord send_image", self.name)
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata=metadata)

        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")

            # Download the image and send as a Discord file attachment
            # (Discord renders attachments inline, unlike plain URLs)
            from gateway.platforms.base import resolve_proxy_url
            _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
            download_budget = _patchable_dependency(
                "_DISCORD_IMAGE_DOWNLOAD_BUDGET_CONTEXT"
            ).get()
            async with create_image_http_client(_proxy) as client:
                status, image_data, _headers = await read_image_with_redirect_guard(
                    client,
                    image_url,
                    timeout=30.0,
                    request_kwargs={},
                    download_budget=download_budget,
                )
                if status != 200:
                    raise Exception(f"Failed to download image: HTTP {status}")

                ext = image_extension_from_bytes(image_data)
                if ext is None:
                    raise ValueError("Downloaded response is not a supported image")

                import io
                file = discord.File(io.BytesIO(image_data), filename=f"image.{ext}")

                if self._is_forum_parent(channel):
                    return await self._forum_post_file(
                        channel,
                        content=(caption or "").strip(),
                        file=file,
                    )

                msg = await channel.send(
                    content=caption if caption else None,
                    file=file,
                )
                return SendResult(success=True, message_id=str(msg.id))

        except ImportError:
            logger.warning(
                "[%s] httpx not installed, falling back to URL. Run: pip install httpx",
                self.name,
                exc_info=True,
            )
            return await super().send_image(chat_id, image_url, caption, reply_to)
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send image attachment, falling back to URL: %s",
                self.name,
                e,
                exc_info=True,
            )
            return await super().send_image(chat_id, image_url, caption, reply_to)

    async def send_animation(
        self,
        chat_id: str,
        animation_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an animated GIF natively as a Discord file attachment."""
        discord = _patchable_dependency("discord")
        async_is_safe_url = _patchable_dependency("async_is_safe_url")
        create_image_http_client = _patchable_dependency(
            "_create_discord_image_http_client"
        )
        read_image_with_redirect_guard = _patchable_dependency(
            "_read_url_image_with_redirect_guard"
        )
        image_extension_from_bytes = _patchable_dependency(
            "_discord_image_extension_from_bytes"
        )

        if not self._client:
            return SendResult(success=False, error="Not connected")

        if not await async_is_safe_url(animation_url):
            logger.warning("[%s] Blocked unsafe animation URL during Discord send_animation", self.name)
            return await super().send_animation(chat_id, animation_url, caption, reply_to, metadata=metadata)

        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                return SendResult(success=False, error=f"Channel {chat_id} not found")

            # Download the GIF and send as a Discord file attachment
            # (Discord renders .gif attachments as auto-playing animations inline)
            from gateway.platforms.base import resolve_proxy_url
            _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
            download_budget = _patchable_dependency(
                "_DISCORD_IMAGE_DOWNLOAD_BUDGET_CONTEXT"
            ).get()
            async with create_image_http_client(_proxy) as client:
                status, animation_data, _headers = await read_image_with_redirect_guard(
                    client,
                    animation_url,
                    timeout=30.0,
                    request_kwargs={},
                    download_budget=download_budget,
                )
                if status != 200:
                    raise Exception(f"Failed to download animation: HTTP {status}")

                ext = image_extension_from_bytes(animation_data)
                if ext != "gif":
                    logger.warning(
                        "[%s] Downloaded animation response is not a GIF: %s",
                        self.name,
                        animation_url[:80],
                    )
                    return await BasePlatformAdapter.send_image(
                        self,
                        chat_id,
                        animation_url,
                        caption,
                        reply_to,
                        metadata=metadata,
                    )

                import io
                file = discord.File(io.BytesIO(animation_data), filename="animation.gif")

                if self._is_forum_parent(channel):
                    return await self._forum_post_file(
                        channel,
                        content=(caption or "").strip(),
                        file=file,
                    )

                msg = await channel.send(
                    content=caption if caption else None,
                    file=file,
                )
                return SendResult(success=True, message_id=str(msg.id))

        except ImportError:
            logger.warning(
                "[%s] httpx not installed, falling back to URL. Run: pip install httpx",
                self.name,
                exc_info=True,
            )
            return await BasePlatformAdapter.send_image(
                self,
                chat_id,
                animation_url,
                caption,
                reply_to,
                metadata=metadata,
            )
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(
                "[%s] Failed to send animation attachment, falling back to URL: %s",
                self.name,
                e,
                exc_info=True,
            )
            return await BasePlatformAdapter.send_image(
                self,
                chat_id,
                animation_url,
                caption,
                reply_to,
                metadata=metadata,
            )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a local video file natively as a Discord attachment."""
        try:
            return await self._send_file_attachment(chat_id, video_path, caption)
        except FileNotFoundError:
            return SendResult(success=False, error=f"Video file not found: {video_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send local video, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_video(chat_id, video_path, caption, reply_to, metadata=metadata)

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an arbitrary file natively as a Discord attachment."""
        try:
            return await self._send_file_attachment(chat_id, file_path, caption, file_name=file_name)
        except FileNotFoundError:
            return SendResult(success=False, error=f"File not found: {file_path}")
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error("[%s] Failed to send document, falling back to base adapter: %s", self.name, e, exc_info=True)
            return await super().send_document(chat_id, file_path, caption, file_name, reply_to, metadata=metadata)

    async def _send_multiple_images_via_base_with_budget(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]],
        human_delay: float,
        download_budget: Any,
    ) -> None:
        """Run the per-image fallback without escaping the batch budget."""
        budget_context = _patchable_dependency("_DISCORD_IMAGE_DOWNLOAD_BUDGET_CONTEXT")
        token = budget_context.set(download_budget)
        try:
            await super().send_multiple_images(chat_id, images, metadata, human_delay)
        finally:
            budget_context.reset(token)

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images as a single Discord message with multiple attachments.

        Discord permits up to 10 file attachments per message. Batches are
        chunked accordingly. URL images are downloaded into memory and
        uploaded as inline attachments (same pattern as ``send_image`` so
        they render inline, not as bare links). Local files are opened
        directly. On per-chunk failure the remaining images in that chunk
        fall back to the base per-image loop.
        """
        if not self._client:
            return
        if not images:
            return

        image_download_budget_cls = _patchable_dependency("_DiscordImageDownloadBudget")
        batch_download_max_bytes = _patchable_dependency(
            "_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES"
        )
        image_download_budget = image_download_budget_cls(batch_download_max_bytes)

        try:
            import discord as _discord_mod
            import io as _io
            from urllib.parse import unquote as _unquote

            resolved_discord = _patchable_dependency("discord")
            local_discord_default = _DEPENDENCY_DEFAULTS["discord"][0]
            if resolved_discord is not local_discord_default:
                _discord_mod = resolved_discord
        except Exception:  # pragma: no cover
            await self._send_multiple_images_via_base_with_budget(
                chat_id,
                images,
                metadata,
                human_delay,
                image_download_budget,
            )
            return

        async_is_safe_url = _patchable_dependency("async_is_safe_url")
        create_image_http_client = _patchable_dependency(
            "_create_discord_image_http_client"
        )
        read_image_with_redirect_guard = _patchable_dependency(
            "_read_url_image_with_redirect_guard"
        )
        image_extension_from_bytes = _patchable_dependency(
            "_discord_image_extension_from_bytes"
        )

        try:
            channel = self._client.get_channel(int(chat_id))
            if not channel:
                channel = await self._client.fetch_channel(int(chat_id))
            if not channel:
                logger.warning("[%s] Channel %s not found for multi-image send", self.name, chat_id)
                return
        except Exception as e:
            logger.warning("[%s] Failed to resolve channel for multi-image send: %s", self.name, e)
            await self._send_multiple_images_via_base_with_budget(
                chat_id,
                images,
                metadata,
                human_delay,
                image_download_budget,
            )
            return

        CHUNK = 10
        chunks = [images[i:i + CHUNK] for i in range(0, len(images), CHUNK)]

        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)

            files: List[Any] = []
            captions: List[str] = []
            image_http_client = None
            try:
                for image_url, alt_text in chunk:
                    if alt_text:
                        captions.append(alt_text)
                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        if not os.path.exists(local_path):
                            logger.warning("[%s] Skipping missing image: %s", self.name, local_path)
                            continue
                        files.append(_discord_mod.File(local_path, filename=os.path.basename(local_path)))
                    else:
                        if not await async_is_safe_url(image_url):
                            logger.warning("[%s] Blocked unsafe image URL in batch", self.name)
                            continue
                        # Download to BytesIO so it renders inline
                        try:
                            from gateway.platforms.base import resolve_proxy_url
                            if image_http_client is None:
                                _proxy = resolve_proxy_url(platform_env_var="DISCORD_PROXY")
                                image_http_client = create_image_http_client(_proxy)
                            status, data, headers = await read_image_with_redirect_guard(
                                image_http_client,
                                image_url,
                                timeout=30.0,
                                request_kwargs={},
                                download_budget=image_download_budget,
                            )
                            if status != 200:
                                logger.warning(
                                    "[%s] Failed to download image (HTTP %d) in batch: %s",
                                    self.name, status, image_url[:80],
                                )
                                continue
                            ext = image_extension_from_bytes(data)
                            if ext is None:
                                raise ValueError("Downloaded response is not a supported image")
                            files.append(_discord_mod.File(_io.BytesIO(data), filename=f"image_{len(files)}.{ext}"))
                        except Exception as dl_err:
                            logger.warning("[%s] Download failed for %s: %s", self.name, image_url[:80], dl_err)
                            continue

                if not files:
                    continue

                # Use the first caption if any (Discord only has one message body for the group)
                content = captions[0] if captions else None
                logger.info(
                    "[%s] Sending %d image(s) as single Discord message (chunk %d/%d)",
                    self.name, len(files), chunk_idx + 1, len(chunks),
                )

                if self._is_forum_parent(channel):
                    await self._forum_post_file(
                        channel,
                        content=(content or "").strip(),
                        files=files,
                    )
                else:
                    await channel.send(content=content, files=files)
            except Exception as e:
                logger.warning(
                    "[%s] Multi-image Discord send failed (chunk %d/%d), falling back to per-image: %s",
                    self.name, chunk_idx + 1, len(chunks), e,
                    exc_info=True,
                )
                await self._send_multiple_images_via_base_with_budget(
                    chat_id,
                    chunk,
                    metadata,
                    human_delay,
                    image_download_budget,
                )
            finally:
                if image_http_client is not None:
                    try:
                        await image_http_client.aclose()
                    except Exception:
                        pass


def _snapshot_dependency_defaults(adapter: Any) -> Dict[str, Tuple[Any, Any]]:
    return {
        "discord": (discord, adapter.discord),
        "async_is_safe_url": (async_is_safe_url, adapter.async_is_safe_url),
        "_create_discord_image_http_client": (
            _create_discord_image_http_client,
            adapter._create_discord_image_http_client,
        ),
        "_read_url_image_with_redirect_guard": (
            _read_url_image_with_redirect_guard,
            adapter._read_url_image_with_redirect_guard,
        ),
        "_discord_image_extension_from_bytes": (
            _discord_image_extension_from_bytes,
            adapter._discord_image_extension_from_bytes,
        ),
        "_read_response_bytes_bounded": (
            _read_response_bytes_bounded,
            adapter._read_response_bytes_bounded,
        ),
        "_DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES": (
            _DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES,
            adapter._DISCORD_IMAGE_BATCH_DOWNLOAD_MAX_BYTES,
        ),
        "_DISCORD_IMAGE_DOWNLOAD_BUDGET_CONTEXT": (
            _DISCORD_IMAGE_DOWNLOAD_BUDGET_CONTEXT,
            adapter._DISCORD_IMAGE_DOWNLOAD_BUDGET_CONTEXT,
        ),
        "_DISCORD_IMAGE_DOWNLOAD_MAX_BYTES": (
            _DISCORD_IMAGE_DOWNLOAD_MAX_BYTES,
            adapter._DISCORD_IMAGE_DOWNLOAD_MAX_BYTES,
        ),
        "_DISCORD_IMAGE_MAX_REDIRECTS": (
            _DISCORD_IMAGE_MAX_REDIRECTS,
            adapter._DISCORD_IMAGE_MAX_REDIRECTS,
        ),
        "_DISCORD_IMAGE_REDIRECT_STATUSES": (
            _DISCORD_IMAGE_REDIRECT_STATUSES,
            adapter._DISCORD_IMAGE_REDIRECT_STATUSES,
        ),
        "_DiscordImageDownloadBudget": (
            _DiscordImageDownloadBudget,
            adapter._DiscordImageDownloadBudget,
        ),
    }


_adapter_module = sys.modules.get("plugins.platforms.discord.adapter")
_DEPENDENCY_DEFAULTS: Optional[Dict[str, Tuple[Any, Any]]] = (
    _snapshot_dependency_defaults(_adapter_module)
    if _adapter_module is not None
    else None
)


def _patchable_dependency(name: str) -> Any:
    """Resolve a moved method's local seam before the adapter-level seam."""
    global _DEPENDENCY_DEFAULTS
    from . import adapter

    if _DEPENDENCY_DEFAULTS is None:
        _DEPENDENCY_DEFAULTS = _snapshot_dependency_defaults(adapter)

    local_value = globals()[name]
    local_default, adapter_default = _DEPENDENCY_DEFAULTS[name]
    if local_value is not local_default:
        return local_value
    adapter_value = getattr(adapter, name)
    if adapter_value is not adapter_default:
        return adapter_value
    return local_value
