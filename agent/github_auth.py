"""Shared GitHub App authentication (JWT signing + installation tokens).

Used by ``tools/skills_hub.py`` (skills-hub fetches) and the bundled
``plugins/github`` connector so the GitHub App flow lives in exactly one
place.

Credentials (profile-scoped secrets via ``agent.secret_scope.get_secret``):

- ``GITHUB_APP_ID`` — the GitHub App ID.
- ``GITHUB_APP_PRIVATE_KEY_PATH`` — path to the app's ``.pem`` private key.
- ``GITHUB_APP_INSTALLATION_ID`` — the installation id the app was
  approved for. Optional for read-only flows that can resolve an
  installation by repository, but required for token minting here.

Installation tokens live ~1 hour; this module caches the token and mints
a fresh one shortly before expiry. All network access is lazy (importing
this module never hits the network).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Installation tokens last 1 hour; refresh ~50 min after mint to stay well
# inside the validity window (mirrors the 3500s expiry used by skills_hub).
_APP_TOKEN_TTL = 3600
_APP_TOKEN_REFRESH_AFTER = 3000


class GitHubAppAuth:
    """GitHub App JWT + installation-token minting, with caching.

    Stateless enough to construct per call; token cache is per-instance.
    Any failure (missing credentials, bad key, network error) returns
    ``None`` instead of raising, so callers can fall through to PAT/gh CLI.
    """

    def __init__(self) -> None:
        self._cached_token: Optional[str] = None
        self._token_minted_at: float = 0.0
        self._cached_slug: Optional[str] = None

    # ------------------------------------------------------------------
    # Credential discovery
    # ------------------------------------------------------------------

    def credentials_configured(self) -> bool:
        """True when all three GitHub App env secrets are present."""
        app_id, key_path, installation_id = self._credentials()
        return bool(app_id and key_path and installation_id)

    def _credentials(self):
        from agent.secret_scope import get_secret

        return (
            get_secret("GITHUB_APP_ID"),
            get_secret("GITHUB_APP_PRIVATE_KEY_PATH"),
            get_secret("GITHUB_APP_INSTALLATION_ID"),
        )

    # ------------------------------------------------------------------
    # Installation token
    # ------------------------------------------------------------------

    def installation_token(self) -> Optional[str]:
        """Return a valid installation token, minting one if needed.

        Returns ``None`` when the app is not configured or any step fails
        (missing PyJWT, unreadable key, non-201 from GitHub).
        """
        if self._cached_token and (time.time() - self._token_minted_at) < _APP_TOKEN_REFRESH_AFTER:
            return self._cached_token

        token = self._mint_installation_token()
        if token:
            self._cached_token = token
            self._token_minted_at = time.time()
        return token

    def _mint_installation_token(self) -> Optional[str]:
        app_id, key_path, installation_id = self._credentials()
        if not all([app_id, key_path, installation_id]):
            return None

        assert app_id is not None and key_path is not None  # for type checkers
        jwt_token = self._sign_jwt(app_id, key_path)
        if not jwt_token:
            return None

        try:
            import httpx

            resp = httpx.post(
                f"https://api.github.com/app/installations/{installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {jwt_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=10,
            )
            if resp.status_code == 201:
                return resp.json().get("token")
            logger.debug("GitHub App access_tokens returned %s", resp.status_code)
        except Exception as e:
            logger.debug("GitHub App token mint failed: %s", e)
        return None

    def _sign_jwt(self, app_id: str, key_path: str) -> Optional[str]:
        """Sign a 10-minute RS256 JWT for the GitHub App (``iss`` = App ID)."""
        try:
            import jwt  # PyJWT
        except ImportError:
            logger.debug("PyJWT not installed, skipping GitHub App auth")
            return None

        try:
            key_file = Path(key_path)
            if not key_file.exists():
                return None
            private_key = key_file.read_text(encoding="utf-8")

            now = int(time.time())
            payload = {
                "iat": now - 60,
                "exp": now + (10 * 60),
                "iss": app_id,
            }
            return jwt.encode(payload, private_key, algorithm="RS256")
        except Exception as e:
            logger.debug("GitHub App JWT signing failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Bot identity (attribution)
    # ------------------------------------------------------------------

    def app_slug(self) -> Optional[str]:
        """The GitHub App slug (e.g. ``jarpis-bot``), fetched via ``GET /app``.

        Cached for the life of the instance. Returns ``None`` when the app
        is not configured or the call fails — callers must fall back to a
        generic label, never crash.
        """
        if self._cached_slug is not None:
            return self._cached_slug

        app_id, key_path, _ = self._credentials()
        if not (app_id and key_path):
            return None

        jwt_token = self._sign_jwt(app_id, key_path)
        if not jwt_token:
            return None

        try:
            import httpx

            resp = httpx.get(
                "https://api.github.com/app",
                headers={
                    "Authorization": f"Bearer {jwt_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                self._cached_slug = resp.json().get("slug")
        except Exception as e:
            logger.debug("GitHub App slug lookup failed: %s", e)
        return self._cached_slug

    def bot_login(self) -> Optional[str]:
        """The bot's GitHub login: ``<slug>[bot]`` (e.g. ``jarpis-bot[bot]``).

        This is the login that appears on comments/reviews made with the
        installation token — the whole point of App identity: an agent
        action is attributed to the bot, never to the account owner.
        """
        slug = self.app_slug()
        if slug:
            return f"{slug}[bot]"
        return None
