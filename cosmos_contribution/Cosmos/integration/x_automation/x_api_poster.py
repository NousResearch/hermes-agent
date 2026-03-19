"""
X/Twitter API v2 Poster - OAuth 2.0 with PKCE
"""
import os
import json
import base64
import hashlib
import secrets
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Load .env file (cross-platform)
def load_env():
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / ".env",  # Project root
        Path(__file__).parent.parent.parent / ".env",
        Path("/workspace/Cosmos/.env"),  # Docker/cloud
    ]

    for env_path in possible_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip()
            break

load_env()

# OAuth 2.0 Configuration
CONFIG = {
    "client_id": os.environ.get("X_CLIENT_ID", "OUJSQ3BEX0Npc3pxZm1HcmxxWDc6MTpjaQ"),
    "client_secret": os.environ.get("X_CLIENT_SECRET", "3-7lG5ethJte5qPpk4H-PoT8V1gOVtMcMUZUrrK1AdxRQWciVV"),
    "redirect_uri": "https://ai.cosmos.cloud/callback",
    "token_file": Path(__file__).parent / "oauth2_tokens.json",  # Same dir as this file
}

# OAuth 1.0a Configuration (for media upload - requires separate credentials)
# To enable media upload, add these to .env:
#   X_API_KEY=your_consumer_key
#   X_API_SECRET=your_consumer_secret
#   X_OAUTH1_ACCESS_TOKEN=your_access_token
#   X_OAUTH1_ACCESS_SECRET=your_access_token_secret
OAUTH1_CONFIG = {
    "api_key": os.environ.get("X_API_KEY"),
    "api_secret": os.environ.get("X_API_SECRET"),
    "access_token": os.environ.get("X_OAUTH1_ACCESS_TOKEN"),
    "access_secret": os.environ.get("X_OAUTH1_ACCESS_SECRET"),
}

# X API v2 Endpoints (updated per docs.x.com)
AUTHORIZE_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"
TWEET_URL = "https://api.x.com/2/tweets"
# Media upload uses v1.1 endpoint (requires OAuth 1.0a for authentication)
MEDIA_UPLOAD_URL = "https://upload.twitter.com/1.1/media/upload.json"


def create_oauth1_signature(method: str, url: str, params: dict, oauth1_config: dict) -> str:
    """Create OAuth 1.0a signature for media upload"""
    import hmac
    import time
    import random
    import string

    # OAuth 1.0a params
    oauth_params = {
        "oauth_consumer_key": oauth1_config["api_key"],
        "oauth_token": oauth1_config["access_token"],
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_nonce": ''.join(random.choices(string.ascii_letters + string.digits, k=32)),
        "oauth_version": "1.0",
    }

    # Combine all params for signature base
    all_params = {**oauth_params, **params}
    sorted_params = sorted(all_params.items())
    param_string = "&".join(f"{urllib.parse.quote(str(k), safe='')}={urllib.parse.quote(str(v), safe='')}"
                           for k, v in sorted_params)

    # Create signature base string
    base_string = f"{method.upper()}&{urllib.parse.quote(url, safe='')}&{urllib.parse.quote(param_string, safe='')}"

    # Create signing key
    signing_key = f"{urllib.parse.quote(oauth1_config['api_secret'], safe='')}&{urllib.parse.quote(oauth1_config['access_secret'], safe='')}"

    # Generate signature
    signature = base64.b64encode(
        hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
    ).decode()

    oauth_params["oauth_signature"] = signature

    # Create Authorization header
    auth_header = "OAuth " + ", ".join(
        f'{urllib.parse.quote(k, safe="")}="{urllib.parse.quote(v, safe="")}"'
        for k, v in sorted(oauth_params.items())
    )

    return auth_header


def has_oauth1_credentials() -> bool:
    """Check if OAuth 1.0a credentials are configured"""
    return all([
        OAUTH1_CONFIG.get("api_key"),
        OAUTH1_CONFIG.get("api_secret"),
        OAUTH1_CONFIG.get("access_token"),
        OAUTH1_CONFIG.get("access_secret"),
    ])


class XOAuth2Poster:
    """Posts tweets using X API v2 with OAuth 2.0 PKCE"""

    def __init__(self):
        self.client_id = CONFIG["client_id"]
        self.client_secret = CONFIG["client_secret"]
        self.redirect_uri = CONFIG["redirect_uri"]
        self.token_file = CONFIG["token_file"]
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
        self.posts_today = 0
        self.daily_limit = 50  # Increased for mega threads + hackathon engagement
        self._load_tokens()

    def _load_tokens(self):
        """Load saved tokens from file or env vars"""
        # First try env vars (use distinct names to avoid conflict with OAuth 1.0a)
        env_access = os.environ.get("X_OAUTH2_ACCESS_TOKEN")
        env_refresh = os.environ.get("X_OAUTH2_REFRESH_TOKEN")
        if env_access:
            self.access_token = env_access
            self.refresh_token = env_refresh
            self.token_expires_at = datetime.now() + timedelta(hours=2)
            logger.info("Loaded OAuth2 tokens from environment")
            return

        # Then try token file (primary method)
        if self.token_file.exists():
            try:
                with open(self.token_file) as f:
                    data = json.load(f)
                    self.access_token = data.get("access_token")
                    self.refresh_token = data.get("refresh_token")
                    expires_at = data.get("expires_at")
                    if expires_at:
                        self.token_expires_at = datetime.fromisoformat(expires_at)
                    logger.info("Loaded OAuth2 tokens from file")
            except Exception as e:
                logger.error(f"Failed to load tokens: {e}")

    def _save_tokens(self):
        """Save tokens to file"""
        try:
            self.token_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.token_file, "w") as f:
                json.dump({
                    "access_token": self.access_token,
                    "refresh_token": self.refresh_token,
                    "expires_at": self.token_expires_at.isoformat() if self.token_expires_at else None,
                }, f)
            logger.info("Saved OAuth2 tokens to file")
        except Exception as e:
            logger.error(f"Failed to save tokens: {e}")

    def is_configured(self) -> bool:
        """Check if we have valid tokens"""
        return bool(self.access_token)

    def is_token_expired(self) -> bool:
        """Check if token is expired"""
        if not self.token_expires_at:
            return True
        return datetime.now() >= self.token_expires_at - timedelta(minutes=5)

    def get_authorization_url(self) -> tuple:
        """Generate OAuth2 authorization URL with PKCE"""
        # Generate PKCE code verifier and challenge
        code_verifier = secrets.token_urlsafe(64)[:128]
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip("=")

        state = secrets.token_urlsafe(32)

        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "tweet.read tweet.write users.read media.write offline.access",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }

        url = f"{AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
        return url, code_verifier, state

    async def exchange_code_for_tokens(self, code: str, code_verifier: str) -> bool:
        """Exchange authorization code for access tokens"""
        try:
            import httpx

            # Basic auth header
            credentials = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()

            headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "code_verifier": code_verifier,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(TOKEN_URL, headers=headers, data=data, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    self.access_token = result["access_token"]
                    self.refresh_token = result.get("refresh_token")
                    expires_in = result.get("expires_in", 7200)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    self._save_tokens()
                    logger.info("OAuth2 tokens obtained successfully")
                    return True
                else:
                    logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Token exchange error: {e}")
            return False

    async def refresh_access_token(self) -> bool:
        """Refresh the access token using refresh token"""
        if not self.refresh_token:
            logger.error("No refresh token available")
            return False

        try:
            import httpx

            credentials = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()

            headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(TOKEN_URL, headers=headers, data=data, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    self.access_token = result["access_token"]
                    self.refresh_token = result.get("refresh_token", self.refresh_token)
                    expires_in = result.get("expires_in", 7200)
                    self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                    self._save_tokens()
                    logger.info("OAuth2 token refreshed successfully")
                    return True
                else:
                    logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    def can_post(self) -> bool:
        if self.posts_today >= self.daily_limit:
            logger.warning(f"Daily limit reached ({self.daily_limit} posts)")
            return False
        return True

    async def upload_media(self, image_bytes: bytes, media_type: str = "image/png") -> Optional[str]:
        """
        Upload media using Twitter API.

        Tries multiple methods:
        1. OAuth 2.0 Bearer token with simple upload (newer method)
        2. OAuth 1.0a with chunked upload (fallback)

        Returns media_id on success.
        """
        if not self.is_configured():
            logger.error("X API not configured")
            return None

        if self.is_token_expired():
            if not await self.refresh_access_token():
                return None

        # Try OAuth 2.0 simple upload first (for images < 5MB)
        media_id = await self._upload_media_oauth2(image_bytes, media_type)
        if media_id:
            return media_id

        # Fallback to OAuth 1.0a if available
        if has_oauth1_credentials():
            return await self._upload_media_oauth1(image_bytes, media_type)

        logger.warning("Media upload failed. OAuth 2.0 method unsuccessful and no OAuth 1.0a credentials available.")
        return None

    async def _upload_media_oauth2(self, image_bytes: bytes, media_type: str) -> Optional[str]:
        """
        Try simple media upload with OAuth 2.0 Bearer token.
        Uses base64 encoded media_data in a single POST.
        """
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.access_token}",
            }

            # Simple upload - single POST with base64 media
            media_b64 = base64.b64encode(image_bytes).decode()

            upload_data = {
                "media_data": media_b64,
                "media_category": "tweet_image",
            }

            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    MEDIA_UPLOAD_URL,
                    headers=headers,
                    data=upload_data
                )

                if response.status_code in [200, 201, 202]:
                    result = response.json()
                    media_id = result.get("media_id_string") or str(result.get("media_id", ""))
                    if media_id:
                        logger.info(f"Media upload (OAuth2) success: {media_id}")
                        return media_id

                logger.debug(f"OAuth2 media upload failed: {response.status_code} - {response.text[:200]}")
                return None

        except Exception as e:
            logger.debug(f"OAuth2 media upload error: {e}")
            return None

    async def _upload_media_oauth1(self, image_bytes: bytes, media_type: str) -> Optional[str]:
        """
        Chunked media upload with OAuth 1.0a using requests-oauthlib.
        Requires X_API_KEY, X_API_SECRET, X_OAUTH1_ACCESS_TOKEN, X_OAUTH1_ACCESS_SECRET in .env
        """
        try:
            from requests_oauthlib import OAuth1Session
            import asyncio

            # Create OAuth1 session with proper credentials
            oauth = OAuth1Session(
                OAUTH1_CONFIG["api_key"],
                client_secret=OAUTH1_CONFIG["api_secret"],
                resource_owner_key=OAUTH1_CONFIG["access_token"],
                resource_owner_secret=OAUTH1_CONFIG["access_secret"],
            )

            loop = asyncio.get_event_loop()

            # Auto-detect actual image format from magic bytes for INIT
            detected_type = media_type
            if image_bytes[:3] == b'\xff\xd8\xff':
                detected_type = "image/jpeg"
            elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                detected_type = "image/png"
            elif image_bytes[:4] == b'GIF8':
                detected_type = "image/gif"
            elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
                detected_type = "image/webp"

            # Step 1: INIT
            init_params = {
                "command": "INIT",
                "media_type": detected_type,
                "total_bytes": len(image_bytes),
                "media_category": "tweet_image"
            }

            init_resp = await loop.run_in_executor(
                None,
                lambda: oauth.post(MEDIA_UPLOAD_URL, data=init_params)
            )

            if init_resp.status_code not in [200, 202]:
                logger.error(f"Media INIT failed: {init_resp.status_code} - {init_resp.text}")
                return None

            init_result = init_resp.json()
            media_id = init_result.get("media_id_string") or str(init_result.get("media_id", ""))

            if not media_id:
                logger.error(f"No media_id in INIT response: {init_result}")
                return None

            logger.info(f"Media INIT success: media_id={media_id}")

            # Step 2: APPEND - send media as multipart file
            # Auto-detect actual image format from magic bytes
            actual_type = media_type
            filename = "media.png"
            if image_bytes[:3] == b'\xff\xd8\xff':
                actual_type = "image/jpeg"
                filename = "media.jpg"
            elif image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
                actual_type = "image/png"
                filename = "media.png"
            elif image_bytes[:4] == b'GIF8':
                actual_type = "image/gif"
                filename = "media.gif"
            elif image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
                actual_type = "image/webp"
                filename = "media.webp"

            logger.info(f"Media type detected: {actual_type} (requested: {media_type})")

            append_data = {
                "command": "APPEND",
                "media_id": media_id,
                "segment_index": 0,
            }
            files = {
                "media": (filename, image_bytes, actual_type)
            }

            append_resp = await loop.run_in_executor(
                None,
                lambda: oauth.post(MEDIA_UPLOAD_URL, data=append_data, files=files)
            )

            # APPEND returns 204 No Content on success
            if append_resp.status_code not in [200, 202, 204]:
                logger.error(f"Media APPEND failed: {append_resp.status_code} - {append_resp.text}")
                return None

            logger.info("Media APPEND success")

            # Step 3: FINALIZE - complete the upload
            finalize_params = {
                "command": "FINALIZE",
                "media_id": media_id
            }

            finalize_resp = await loop.run_in_executor(
                None,
                lambda: oauth.post(MEDIA_UPLOAD_URL, data=finalize_params)
            )

            if finalize_resp.status_code not in [200, 201, 202]:
                logger.error(f"Media FINALIZE failed: {finalize_resp.status_code} - {finalize_resp.text}")
                return None

            finalize_result = finalize_resp.json()
            logger.info(f"Media upload complete: {media_id}")

            # Check if processing is needed (for video/gif)
            processing_info = finalize_result.get("processing_info")
            if processing_info:
                logger.info(f"Media processing: {processing_info}")

            return media_id

        except Exception as e:
            logger.error(f"Media upload error: {e}")
            return None

    async def post_tweet_with_media(self, text: str, image_bytes: bytes) -> Optional[Dict]:
        """Post a tweet with an attached image"""
        # Upload media first
        media_id = await self.upload_media(image_bytes)
        if not media_id:
            logger.warning("Media upload failed, posting text only")
            return await self.post_tweet(text)

        if not self.can_post():
            return None

        # X Premium allows up to 4000 characters, X Premium+ up to 25000
        # Use 4000 as default for verified accounts
        max_chars = int(os.environ.get("X_MAX_CHARS", "4000"))
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "text": text,
                "media": {
                    "media_ids": [media_id]
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TWEET_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 201:
                    self.posts_today += 1
                    result = response.json()
                    tweet_id = result.get("data", {}).get("id")
                    logger.info(f"Tweet with media posted: {tweet_id}")
                    return result
                else:
                    logger.error(f"Tweet with media failed: {response.status_code} - {response.text}")
                    # Fallback to text-only
                    return await self.post_tweet(text)

        except Exception as e:
            logger.error(f"Tweet with media error: {e}")
            return None

    async def post_tweet(self, text: str, image_bytes: Optional[bytes] = None) -> Optional[Dict]:
        """Post a tweet using OAuth 2.0, optionally with an image"""
        # If image provided, use the media upload flow
        if image_bytes:
            return await self.post_tweet_with_media(text, image_bytes)

        if not self.is_configured():
            logger.error("X API not configured - run authorization first")
            return None

        # Refresh token if expired
        if self.is_token_expired():
            if not await self.refresh_access_token():
                logger.error("Failed to refresh token")
                return None

        if not self.can_post():
            return None

        # X Premium allows up to 4000 characters, X Premium+ up to 25000
        # Use 4000 as default for verified accounts
        max_chars = int(os.environ.get("X_MAX_CHARS", "4000"))
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TWEET_URL,
                    headers=headers,
                    json={"text": text},
                    timeout=30
                )

                if response.status_code == 201:
                    self.posts_today += 1
                    result = response.json()
                    tweet_id = result.get("data", {}).get("id")
                    logger.info(f"Tweet posted successfully: {tweet_id}")
                    return result
                elif response.status_code == 401:
                    # Token might be invalid, try refresh
                    logger.warning("401 error, attempting token refresh")
                    if await self.refresh_access_token():
                        return await self.post_tweet(text)
                    return None
                else:
                    logger.error(f"Tweet failed: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Tweet error: {e}")
            return None

    async def upload_video(self, video_bytes: bytes, media_type: str = "video/mp4") -> Optional[str]:
        """
        Upload video using chunked media upload (INIT → APPEND → FINALIZE → STATUS).

        Videos require OAuth 1.0a for the media upload endpoint.

        Args:
            video_bytes: Raw video file bytes
            media_type: MIME type (video/mp4, video/quicktime, etc.)

        Returns:
            media_id string on success, None on failure
        """
        if not has_oauth1_credentials():
            logger.error("Video upload requires OAuth 1.0a credentials (X_API_KEY, X_API_SECRET, X_OAUTH1_ACCESS_TOKEN, X_OAUTH1_ACCESS_SECRET)")
            return None

        try:
            from requests_oauthlib import OAuth1Session
            import asyncio

            oauth = OAuth1Session(
                OAUTH1_CONFIG["api_key"],
                client_secret=OAUTH1_CONFIG["api_secret"],
                resource_owner_key=OAUTH1_CONFIG["access_token"],
                resource_owner_secret=OAUTH1_CONFIG["access_secret"],
            )

            loop = asyncio.get_event_loop()
            total_bytes = len(video_bytes)

            # Step 1: INIT
            init_params = {
                "command": "INIT",
                "media_type": media_type,
                "total_bytes": total_bytes,
                "media_category": "tweet_video"
            }

            logger.info(f"Video upload INIT: {total_bytes} bytes, type={media_type}")

            init_resp = await loop.run_in_executor(
                None,
                lambda: oauth.post(MEDIA_UPLOAD_URL, data=init_params)
            )

            if init_resp.status_code not in [200, 202]:
                logger.error(f"Video INIT failed: {init_resp.status_code} - {init_resp.text}")
                return None

            init_result = init_resp.json()
            media_id = init_result.get("media_id_string") or str(init_result.get("media_id", ""))

            if not media_id:
                logger.error(f"No media_id in INIT response: {init_result}")
                return None

            logger.info(f"Video INIT success: media_id={media_id}")

            # Step 2: APPEND - send in chunks (max 5MB per chunk)
            chunk_size = 5 * 1024 * 1024  # 5MB
            segment_index = 0

            for i in range(0, total_bytes, chunk_size):
                chunk = video_bytes[i:i + chunk_size]

                append_data = {
                    "command": "APPEND",
                    "media_id": media_id,
                    "segment_index": segment_index,
                }
                files = {
                    "media": ("video.mp4", chunk, media_type)
                }

                logger.info(f"Video APPEND segment {segment_index}: {len(chunk)} bytes")

                append_resp = await loop.run_in_executor(
                    None,
                    lambda: oauth.post(MEDIA_UPLOAD_URL, data=append_data, files=files)
                )

                if append_resp.status_code not in [200, 202, 204]:
                    logger.error(f"Video APPEND failed: {append_resp.status_code} - {append_resp.text}")
                    return None

                segment_index += 1

            logger.info("Video APPEND complete")

            # Step 3: FINALIZE
            finalize_params = {
                "command": "FINALIZE",
                "media_id": media_id
            }

            finalize_resp = await loop.run_in_executor(
                None,
                lambda: oauth.post(MEDIA_UPLOAD_URL, data=finalize_params)
            )

            if finalize_resp.status_code not in [200, 201, 202]:
                logger.error(f"Video FINALIZE failed: {finalize_resp.status_code} - {finalize_resp.text}")
                return None

            finalize_result = finalize_resp.json()
            logger.info(f"Video FINALIZE response: {finalize_result}")

            # Step 4: STATUS - poll until processing complete
            processing_info = finalize_result.get("processing_info")
            if processing_info:
                check_after_secs = processing_info.get("check_after_secs", 5)
                state = processing_info.get("state", "")

                max_checks = 60  # Max 5 minutes of polling
                checks = 0

                while state in ["pending", "in_progress"] and checks < max_checks:
                    await asyncio.sleep(check_after_secs)

                    status_params = {
                        "command": "STATUS",
                        "media_id": media_id
                    }

                    status_resp = await loop.run_in_executor(
                        None,
                        lambda: oauth.get(MEDIA_UPLOAD_URL, params=status_params)
                    )

                    if status_resp.status_code == 200:
                        status_result = status_resp.json()
                        processing_info = status_result.get("processing_info", {})
                        state = processing_info.get("state", "succeeded")
                        check_after_secs = processing_info.get("check_after_secs", 5)
                        progress = processing_info.get("progress_percent", 0)
                        logger.info(f"Video processing: state={state}, progress={progress}%")

                        if state == "failed":
                            error = processing_info.get("error", {})
                            logger.error(f"Video processing failed: {error}")
                            return None
                    else:
                        logger.warning(f"Video STATUS check failed: {status_resp.status_code}")

                    checks += 1

                if state not in ["succeeded", ""]:
                    logger.error(f"Video processing did not complete: state={state}")
                    return None

            logger.info(f"Video upload complete: media_id={media_id}")
            return media_id

        except Exception as e:
            logger.error(f"Video upload error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def post_tweet_with_video(self, text: str, video_bytes: bytes) -> Optional[Dict]:
        """
        Post a tweet with an attached video.

        Args:
            text: Tweet text
            video_bytes: Raw video file bytes (MP4 recommended)

        Returns:
            Tweet response dict on success, None on failure
        """
        # Upload video first
        media_id = await self.upload_video(video_bytes)
        if not media_id:
            logger.error("Video upload failed, cannot post tweet with video")
            return None

        if not self.can_post():
            return None

        # X Premium allows up to 4000 characters, X Premium+ up to 25000
        # Use 4000 as default for verified accounts
        max_chars = int(os.environ.get("X_MAX_CHARS", "4000"))
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."

        try:
            import httpx

            # Ensure token is fresh
            if self.is_token_expired():
                if not await self.refresh_access_token():
                    return None

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "text": text,
                "media": {
                    "media_ids": [media_id]
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TWEET_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 201:
                    self.posts_today += 1
                    result = response.json()
                    tweet_id = result.get("data", {}).get("id")
                    logger.info(f"Tweet with video posted: {tweet_id}")
                    return result
                else:
                    logger.error(f"Tweet with video failed: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Tweet with video error: {e}")
            return None

    async def post_reply(self, text: str, reply_to_id: str) -> Optional[Dict]:
        """Post a reply to a specific tweet"""
        if not self.is_configured():
            logger.error("X API not configured")
            return None

        if self.is_token_expired():
            if not await self.refresh_access_token():
                return None

        if not self.can_post():
            return None

        # X Premium allows up to 4000 characters, X Premium+ up to 25000
        # Use 4000 as default for verified accounts
        max_chars = int(os.environ.get("X_MAX_CHARS", "4000"))
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "text": text,
                "reply": {
                    "in_reply_to_tweet_id": reply_to_id
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TWEET_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 201:
                    self.posts_today += 1
                    result = response.json()
                    tweet_id = result.get("data", {}).get("id")
                    logger.info(f"Reply posted: {tweet_id}")
                    return result
                else:
                    logger.error(f"Reply failed: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Reply error: {e}")
            return None

    async def post_reply_with_media(
        self,
        text: str,
        image_bytes: bytes,
        reply_to_id: str
    ) -> Optional[Dict]:
        """Post a reply with an image attachment."""
        if not self.is_configured():
            logger.error("X API not configured")
            return None

        if self.is_token_expired():
            if not await self.refresh_access_token():
                return None

        if not self.can_post():
            return None

        # Upload media first
        media_id = await self.upload_media(image_bytes, "image/png")
        if not media_id:
            logger.warning("Media upload failed for reply, falling back to text-only")
            return await self.post_reply(text, reply_to_id)

        # X Premium allows up to 4000 characters, X Premium+ up to 25000
        # Use 4000 as default for verified accounts
        max_chars = int(os.environ.get("X_MAX_CHARS", "4000"))
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "text": text,
                "reply": {
                    "in_reply_to_tweet_id": reply_to_id
                },
                "media": {
                    "media_ids": [media_id]
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TWEET_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 201:
                    self.posts_today += 1
                    result = response.json()
                    tweet_id = result.get("data", {}).get("id")
                    logger.info(f"Reply with media posted: {tweet_id}")
                    return result
                else:
                    logger.error(f"Reply with media failed: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Reply with media error: {e}")
            return None

    async def post_reply_with_video(
        self,
        text: str,
        video_bytes: bytes,
        reply_to_id: str
    ) -> Optional[Dict]:
        """Post a reply with a video attachment."""
        if not self.is_configured():
            logger.error("X API not configured")
            return None

        if self.is_token_expired():
            if not await self.refresh_access_token():
                return None

        if not self.can_post():
            return None

        # Upload video first (uses chunked upload)
        media_id = await self.upload_video(video_bytes)
        if not media_id:
            logger.error("Failed to upload video for reply")
            return None

        # X Premium allows up to 4000 characters, X Premium+ up to 25000
        # Use 4000 as default for verified accounts
        max_chars = int(os.environ.get("X_MAX_CHARS", "4000"))
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."

        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            payload = {
                "text": text,
                "reply": {
                    "in_reply_to_tweet_id": reply_to_id
                },
                "media": {
                    "media_ids": [media_id]
                }
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    TWEET_URL,
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                if response.status_code == 201:
                    self.posts_today += 1
                    result = response.json()
                    tweet_id = result.get("data", {}).get("id")
                    logger.info(f"Reply with video posted: {tweet_id}")
                    return result
                else:
                    logger.error(f"Reply with video failed: {response.status_code} - {response.text}")
                    return None

        except Exception as e:
            logger.error(f"Reply with video error: {e}")
            return None


# Global instance
_x_oauth2_poster = None

def get_x_api_poster() -> XOAuth2Poster:
    global _x_oauth2_poster
    if _x_oauth2_poster is None:
        _x_oauth2_poster = XOAuth2Poster()
    return _x_oauth2_poster

async def post_to_x_api(text: str) -> Optional[Dict]:
    poster = get_x_api_poster()
    return await poster.post_tweet(text)


# CLI for authorization setup
if __name__ == "__main__":
    import sys

    poster = XOAuth2Poster()

    if len(sys.argv) > 1 and sys.argv[1] == "auth":
        # Generate authorization URL
        url, code_verifier, state = poster.get_authorization_url()
        print("\n" + "="*60)
        print("X/TWITTER OAUTH 2.0 AUTHORIZATION")
        print("="*60)
        print("\n1. Open this URL in your browser:\n")
        print(url)
        print("\n2. Log in and authorize the app")
        print("3. You'll be redirected - copy the 'code' parameter from the URL")
        print(f"\n4. Run: python x_api_poster.py callback <code> {code_verifier}")
        print("\n" + "="*60)

    elif len(sys.argv) > 3 and sys.argv[1] == "callback":
        # Exchange code for tokens
        import asyncio
        code = sys.argv[2]
        code_verifier = sys.argv[3]

        async def exchange():
            success = await poster.exchange_code_for_tokens(code, code_verifier)
            if success:
                print("\n✓ Authorization successful! Tokens saved.")
                print("You can now post tweets automatically.")
            else:
                print("\n✗ Authorization failed. Check the error above.")

        asyncio.run(exchange())

    elif len(sys.argv) > 2 and sys.argv[1] == "post":
        # Post a tweet
        import asyncio
        text = " ".join(sys.argv[2:])

        async def post():
            result = await poster.post_tweet(text)
            if result:
                print(f"\n✓ Tweet posted: {result}")
            else:
                print("\n✗ Tweet failed")

        asyncio.run(post())

    else:
        print("Usage:")
        print("  python x_api_poster.py auth              - Start authorization")
        print("  python x_api_poster.py callback <code> <verifier> - Complete auth")
        print("  python x_api_poster.py post <text>       - Post a tweet")
        print(f"\nConfigured: {poster.is_configured()}")
