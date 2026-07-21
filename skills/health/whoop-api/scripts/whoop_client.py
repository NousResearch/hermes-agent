"""Whoop API HTTP client with auto-refresh and rate limit handling.

Provides WhoopClient that manages authentication, automatic token refresh,
and rate limit backoff for all API interactions.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

from whoop_endpoints import ENDPOINTS, Endpoint, all_endpoints
from whoop_storage import load_tokens, save_tokens, load_client_credentials, clear_tokens

API_BASE = "https://api.prod.whoop.com"
RATE_LIMIT = 100  # requests per minute
RATE_WINDOW = 60  # seconds
DAILY_RATE_LIMIT = 10000
MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds, doubles each retry


class TokenExpiredError(Exception):
    """Raised when token refresh fails."""
    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded after retries."""
    pass


class WhoopClient:
    """HTTP client for the Whoop API with auto-refresh and rate limiting."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else Path("whoop_data")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "hermes-whoop-skill/1.0",
        })
        self._request_timestamps: list[float] = []
        self._load_auth()

    def _load_auth(self) -> None:
        """Load tokens from storage and set Authorization header."""
        tokens = load_tokens()
        if tokens is None:
            sys.exit(
                "ERROR: No tokens found. Run `whoop_sync.py setup` first."
            )
        self.session.headers["Authorization"] = f"Bearer {tokens['access_token']}"
        self._tokens = tokens

    def _refresh_if_needed(self) -> None:
        """Refresh access token if expired or about to expire (5min buffer)."""
        expires_at = self._tokens.get("expires_at", 0)
        if time.time() < expires_at - 300:
            return  # Still valid for 5+ minutes

        print("Access token expired, refreshing...")
        creds = load_client_credentials()
        if not creds:
            sys.exit("ERROR: No client credentials found. Run `whoop_sync.py setup` first.")
        client_id = creds["client_id"]
        client_secret = creds["client_secret"]

        response = requests.post(
            f"{API_BASE}/oauth/oauth2/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._tokens["refresh_token"],
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=30,
        )

        if response.status_code == 401:
            # Refresh token is invalid — clear stored tokens so status
            # reports a clean re-auth requirement instead of silent stale state
            clear_tokens()
            raise TokenExpiredError(
                "Refresh token expired. Run `whoop_sync.py setup` to re-authenticate."
            )

        # Catch OAuth error responses (e.g. invalid_grant) even on non-401 status
        token_data = response.json()
        if "error" in token_data:
            clear_tokens()
            raise TokenExpiredError(
                f"Token refresh failed: {token_data['error']}. Run `whoop_sync.py setup` to re-authenticate."
            )

        response.raise_for_status()

        new_expires_at = time.time() + token_data.get("expires_in", 3600)
        save_tokens(
            access_token=token_data["access_token"],
            refresh_token=token_data["refresh_token"],
            expires_at=new_expires_at,
        )
        self.session.headers["Authorization"] = f"Bearer {token_data['access_token']}"
        self._tokens = {**token_data, "expires_at": new_expires_at}
        print("Token refreshed successfully.")

    def _rate_limit_wait(self) -> None:
        """Enforce per-minute and daily rate limits."""
        now = time.time()
        # Remove timestamps older than the per-minute rate window
        self._request_timestamps = [
            ts for ts in self._request_timestamps
            if ts > now - RATE_WINDOW
        ]
        if len(self._request_timestamps) >= RATE_LIMIT:
            oldest = self._request_timestamps[0]
            sleep_time = oldest + RATE_WINDOW - now + 0.1
            if sleep_time > 0:
                print(f"Per-minute rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        # Check daily limit (86400s window)
        day_ago = now - 86400
        daily_count = sum(1 for ts in self._request_timestamps if ts > day_ago)
        if daily_count >= DAILY_RATE_LIMIT:
            print(f"Daily rate limit ({DAILY_RATE_LIMIT}) reached. Stopping.")
            raise RateLimitError(f"Daily rate limit of {DAILY_RATE_LIMIT} requests exceeded")
        self._request_timestamps.append(time.time())

    def _request(self, method: str, endpoint: str, params: dict | None = None) -> dict:
        """Make an authenticated API request with retry and rate limiting."""
        self._refresh_if_needed()

        for attempt in range(MAX_RETRIES):
            self._rate_limit_wait()
            try:
                response = self.session.request(
                    method, f"{API_BASE}{endpoint}",
                    params=params, timeout=30,
                )
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    print(f"Request failed ({e}), retrying in {RETRY_BACKOFF * (attempt + 1)}s...")
                    time.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                raise

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", RETRY_BACKOFF))
                print(f"Rate limited (429), waiting {retry_after}s...")
                time.sleep(retry_after)
                continue
            elif response.status_code == 401:
                # Token might have just expired, try refresh
                try:
                    self._refresh_if_needed()
                    continue
                except TokenExpiredError:
                    raise
            else:
                response.raise_for_status()

        raise RateLimitError(f"Failed after {MAX_RETRIES} retries for {endpoint}")

    def pull_endpoint(self, endpoint: Endpoint, date: str | None = None) -> list[dict]:
        """Pull data from a single endpoint, handling pagination if needed."""
        params: dict[str, str] = {}
        if endpoint.requires_pagination:
            if date:
                params["start"] = f"{date}T00:00:00.000Z"
                params["end"] = f"{date}T23:59:59.999Z"
            else:
                # Default: yesterday (Whoop data has next-day availability)
                from datetime import datetime, timedelta, timezone
                yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
                params["start"] = f"{yesterday}T00:00:00.000Z"
                params["end"] = f"{yesterday}T23:59:59.999Z"

        records = []
        params_copy = dict(params)

        while True:
            data = self._request("GET", endpoint.path, params_copy)
            # API returns list or dict with 'records' key
            items = data if isinstance(data, list) else data.get("records", [])
            records.extend(items)

            # Check for pagination
            next_token = data.get("next_token") if isinstance(data, dict) else None
            if not next_token:
                break
            params_copy["nextToken"] = next_token

        return records

    def pull_all(self, date: str | None = None) -> dict[str, list[dict]]:
        """Pull data from all registered endpoints. Returns {endpoint_name: records}."""
        results: dict[str, list[dict]] = {}
        for endpoint in all_endpoints():
            print(f"  Pulling {endpoint.name}...")
            try:
                records = self.pull_endpoint(endpoint, date)
                results[endpoint.name] = records
                print(f"    {len(records)} records")
            except Exception as e:
                print(f"    Error: {e}")
                results[endpoint.name] = []
        return results

    def save_data(self, data: dict[str, list[dict]], date: str | None = None) -> Path:
        """Save pulled data to JSON files organized by date."""
        from datetime import datetime, timedelta, timezone
        date_str = date or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        date_dir = self.data_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        for endpoint_name, records in data.items():
            file_path = date_dir / f"{endpoint_name}.json"
            file_path.write_text(json.dumps(records, indent=2))
            print(f"  Saved {file_path}")

        return date_dir