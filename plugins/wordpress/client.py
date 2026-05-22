"""Small WordPress REST API client for Hermes tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable
from urllib import error, parse, request

from .auth import WordPressCredentials, get_credentials
from .errors import WordPressAPIError


def normalize_base_url(raw: str) -> str:
    value = raw.strip()
    if not value:
        return value
    if not value.startswith(("http://", "https://")):
        value = f"https://{value}"
    return value.rstrip("/")


@dataclass
class WordPressClient:
    credentials: WordPressCredentials
    timeout: float = 20.0
    opener: Callable[..., Any] = request.urlopen

    @classmethod
    def from_env(cls, *, base_url: str | None = None) -> "WordPressClient":
        return cls(credentials=get_credentials(base_url=base_url))

    @property
    def base_url(self) -> str:
        return normalize_base_url(self.credentials.base_url)

    @property
    def api_root(self) -> str:
        return f"{self.base_url}/wp-json"

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        authenticated: bool = True,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.api_root}{path}"
        if query:
            encoded = parse.urlencode(
                {key: value for key, value in query.items() if value is not None},
                doseq=True,
            )
            if encoded:
                url = f"{url}?{encoded}"

        headers = {"Accept": "application/json"}
        payload = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        if authenticated:
            headers["Authorization"] = self.credentials.authorization_header()
        req = request.Request(url, data=payload, headers=headers, method=method)

        try:
            with self.opener(req, timeout=self.timeout) as response:
                payload = response.read().decode("utf-8")
                return json.loads(payload) if payload else {}
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            payload = None
            message = body or exc.reason or "WordPress API request failed"
            try:
                payload = json.loads(body) if body else None
                if isinstance(payload, dict):
                    message = payload.get("message") or payload.get("code") or message
            except json.JSONDecodeError:
                payload = None
            raise WordPressAPIError(
                message,
                status_code=exc.code,
                payload=payload,
            ) from exc
        except error.URLError as exc:
            raise WordPressAPIError(
                f"WordPress API request failed: {exc.reason}",
            ) from exc
        except json.JSONDecodeError as exc:
            raise WordPressAPIError("WordPress API returned invalid JSON") from exc

    def get_site_info(self) -> dict[str, Any]:
        root = self._request_json("", authenticated=False)
        if not isinstance(root, dict):
            raise WordPressAPIError("WordPress API root returned an unexpected payload")
        auth_state = {
            "authenticated": False,
            "current_user": None,
        }
        try:
            user = self._request_json("/wp/v2/users/me", authenticated=True, query={"context": "edit"})
            if not isinstance(user, dict):
                raise WordPressAPIError("WordPress users/me returned an unexpected payload")
            auth_state = {
                "authenticated": True,
                "current_user": {
                    "id": user.get("id"),
                    "slug": user.get("slug"),
                    "name": user.get("name"),
                },
            }
        except WordPressAPIError as exc:
            if exc.status_code not in {401, 403}:
                raise

        namespaces = root.get("namespaces")
        return {
            "site": {
                "base_url": self.base_url,
                "name": root.get("name"),
                "description": root.get("description"),
                "url": root.get("url"),
                "home": root.get("home"),
            },
            "api": {
                "root": self.api_root,
                "namespace_count": len(namespaces or []),
                "namespaces": namespaces or [],
            },
            "auth": auth_state,
        }

    def list_posts(self, *, query: dict[str, Any] | None = None) -> Any:
        params = {"context": "edit"}
        if query:
            params.update(query)
        return self._request_json("/wp/v2/posts", query=params)

    def get_post(self, post_id: int, *, context: str = "edit") -> Any:
        return self._request_json(f"/wp/v2/posts/{int(post_id)}", query={"context": context})

    def create_post(self, payload: dict[str, Any]) -> Any:
        return self._request_json("/wp/v2/posts", method="POST", body=payload)

    def update_post(self, post_id: int, payload: dict[str, Any]) -> Any:
        return self._request_json(f"/wp/v2/posts/{int(post_id)}", method="POST", body=payload)
