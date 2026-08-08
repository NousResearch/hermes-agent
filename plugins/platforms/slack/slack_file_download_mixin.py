"""Slack file-download mixin.

Extracted from plugins/platforms/slack/adapter.py -- R5 slice C5 (file
download) of the god-file kill campaign (epic #78647, target #78638).

Provenance:
- Source window: adapter.py lines 8034-8229 (196 lines, 4 members
  contiguous): _is_slack_cdn_url (classmethod), _resolve_download_token,
  _download_slack_file, _download_slack_file_bytes.
- Golden sha (window bytes at origin/main aaf9688519): with-NL
  2a7edfc04118cf5c3300f5d25ae70e029f3c02ce7a9709263e2ce78cf6ee6e34,
  no-NL 170385ed34a9353b10e69427dc2ec0033262fc919294d37d8a861664e64510d2.
- Move: byte-verbatim. Module-level import intersection: logging (logger),
  re, asyncio. The window's remaining dependencies (urllib.parse.urlparse,
  httpx, the gateway.platforms.base helpers, tools.url_safety helpers) are
  function-local lazy imports that moved verbatim with the methods.
- Class line after slice:
    class SlackAdapter(SlackFileDownloadMixin, BasePlatformAdapter):
- No module-level import of adapter (circular-import guard). The logger keeps
  the qualified adapter name so log records retain their original identity.
"""

import asyncio
import logging
import re

logger = logging.getLogger("plugins.platforms.slack.adapter")


class SlackFileDownloadMixin:
    """File-download behavior for the Slack adapter.

    MRO contract: this mixin must precede ``BasePlatformAdapter`` in
    ``SlackAdapter``'s bases so its ``_download_slack_file`` /
    ``_download_slack_file_bytes`` / ``_is_slack_cdn_url`` /
    ``_resolve_download_token`` methods resolve through the adapter
    (mixin-first, mirroring ``SlackGateMixin``). The adapter owns the
    instance state the methods touch (``self._team_clients``,
    ``self.config``) plus the ``_SLACK_CDN_*`` class attributes; MRO makes
    ``self`` / ``cls`` resolve to ``SlackAdapter``, so the mixin stores
    nothing itself.
    """

    @classmethod
    def _is_slack_cdn_url(cls, url: str) -> bool:
        """Return True when *url* is an https URL on a Slack CDN host."""
        from urllib.parse import urlparse

        try:
            parsed = urlparse(url)
        except ValueError:
            return False
        if parsed.scheme != "https":
            return False
        host = (parsed.hostname or "").lower().rstrip(".")
        if not host:
            return False
        return host in cls._SLACK_CDN_EXACT_HOSTS or host.endswith(
            cls._SLACK_CDN_HOST_SUFFIXES
        )

    def _resolve_download_token(self, url: str, team_id: str = "") -> str:
        """Pick the correct bot token for a Slack file download.

        Order of preference:
        1. Explicit team_id that maps to a known workspace client.
        2. team_id parsed from the file URL itself — Slack private file URLs
           embed the workspace id as ``files-pri/<TEAM_ID>-<FILE_ID>/...`` so
           we can route to the right workspace even when the triggering event
           carried no team info (thread replies / mentions in multi-workspace
           installs). This prevents defaulting to the primary workspace token,
           which makes Slack return an HTML login page instead of file bytes.
        3. Primary workspace token as a last resort.
        """
        if team_id and team_id in self._team_clients:
            return self._team_clients[team_id].token
        try:
            m = re.search(r"/files-pri/(T[A-Z0-9]+)-", url or "")
            if m:
                url_team = m.group(1)
                if url_team in self._team_clients:
                    return self._team_clients[url_team].token
        except Exception:  # pragma: no cover - defensive
            pass
        return self.config.token or ""

    async def _download_slack_file(
        self, url: str, ext: str, audio: bool = False, team_id: str = ""
    ) -> str:
        """Download a Slack file using the bot token for auth, with retry."""
        import httpx
        from gateway.platforms.base import _ssrf_redirect_guard, safe_url_for_log
        from tools.url_safety import create_ssrf_safe_async_client, is_safe_url

        # SSRF guard: the download attaches the bot token, so a URL that
        # resolves to (or 3xx-redirects into) a private/internal address would
        # both leak the token and let the server reach internal services
        # (CWE-918). The outbound send_image() path is already guarded; this
        # is the inbound sibling that was missing the same protection.
        if not is_safe_url(url):
            raise ValueError(
                f"Blocked unsafe Slack file URL (SSRF protection): {safe_url_for_log(url)}"
            )

        # Tighter than the generic SSRF check: these URLs come from Slack file
        # objects (``url_private`` / ``url_private_download``) and legitimately
        # only ever point at the Slack CDN. Refusing everything else stops a
        # forged file object from steering the Bearer-token download at an
        # arbitrary public host (token exfiltration), which the private-IP
        # check alone cannot prevent.
        if not self._is_slack_cdn_url(url):
            raise ValueError(
                "Blocked non-Slack-CDN file URL (token-exfiltration protection): "
                f"{safe_url_for_log(url)}"
            )

        bot_token = self._resolve_download_token(url, team_id)

        # DNS-pinned client: resolve + validate once, dial the vetted IP
        # (closes the DNS-rebinding TOCTOU window between is_safe_url and
        # TCP connect — the redirect hook still re-validates every hop).
        async with create_ssrf_safe_async_client(
            timeout=30.0,
            follow_redirects=True,
            event_hooks={"response": [_ssrf_redirect_guard]},
        ) as client:
            for attempt in range(3):
                try:
                    response = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {bot_token}"},
                    )
                    response.raise_for_status()

                    # Slack may return an HTML sign-in/redirect page
                    # instead of actual media bytes (e.g. expired token,
                    # restricted file access).  Detect this early so we
                    # don't cache bogus data and confuse downstream tools.
                    ct = response.headers.get("content-type", "")
                    if "text/html" in ct:
                        raise ValueError(
                            "Slack returned HTML instead of media "
                            f"(content-type: {ct}); "
                            "check bot token scopes and file permissions"
                        )

                    if audio:
                        from gateway.platforms.base import cache_audio_from_bytes

                        return cache_audio_from_bytes(response.content, ext)
                    else:
                        from gateway.platforms.base import cache_image_from_bytes

                        return cache_image_from_bytes(response.content, ext)
                except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code < 429
                    ):
                        raise
                    if attempt < 2:
                        logger.debug(
                            "Slack file download retry %d/2 for %s: %s",
                            attempt + 1,
                            url[:80],
                            exc,
                        )
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise

    async def _download_slack_file_bytes(self, url: str, team_id: str = "") -> bytes:
        """Download a Slack file and return raw bytes, with retry."""
        import httpx
        from gateway.platforms.base import _ssrf_redirect_guard, safe_url_for_log
        from tools.url_safety import create_ssrf_safe_async_client, is_safe_url

        # SSRF guard (CWE-918): see _download_slack_file. This sibling path
        # also attaches the bot token and must validate the destination plus
        # every redirect hop.
        if not is_safe_url(url):
            raise ValueError(
                f"Blocked unsafe Slack file URL (SSRF protection): {safe_url_for_log(url)}"
            )

        # Slack-CDN allowlist — see _download_slack_file for the rationale.
        if not self._is_slack_cdn_url(url):
            raise ValueError(
                "Blocked non-Slack-CDN file URL (token-exfiltration protection): "
                f"{safe_url_for_log(url)}"
            )

        bot_token = self._resolve_download_token(url, team_id)

        # DNS-pinned client: resolve + validate once, dial the vetted IP
        # (closes the DNS-rebinding TOCTOU window between is_safe_url and
        # TCP connect — the redirect hook still re-validates every hop).
        async with create_ssrf_safe_async_client(
            timeout=30.0,
            follow_redirects=True,
            event_hooks={"response": [_ssrf_redirect_guard]},
        ) as client:
            for attempt in range(3):
                try:
                    response = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {bot_token}"},
                    )
                    response.raise_for_status()
                    ct = response.headers.get("content-type", "")
                    if "text/html" in ct:
                        raise ValueError(
                            "Slack returned HTML instead of file bytes "
                            f"(content-type: {ct}); "
                            "check bot token scopes and file permissions"
                        )
                    return response.content
                except (
                    httpx.TimeoutException,
                    httpx.HTTPStatusError,
                    ValueError,
                ) as exc:
                    if (
                        isinstance(exc, httpx.HTTPStatusError)
                        and exc.response.status_code < 429
                    ):
                        raise
                    if isinstance(exc, ValueError):
                        raise
                    if attempt < 2:
                        logger.debug(
                            "Slack file download retry %d/2 for %s: %s",
                            attempt + 1,
                            url[:80],
                            exc,
                        )
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise
