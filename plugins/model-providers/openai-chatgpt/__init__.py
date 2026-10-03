"""Official Sign in with ChatGPT, using the account's public Responses API."""

import json
import logging
from urllib.request import Request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)
BASE_URL = "https://api.openai.com/v1"


def _auth_handler(action, args):
    from hermes_cli.auth_chatgpt import auth_handler
    return auth_handler(action, args)


def _refresh_credential(entry):
    from hermes_cli.auth_chatgpt import refresh_credential
    return refresh_credential(entry)


def _classify_api_error(error, **kwargs):
    from agent.chatgpt_responses import classify_chatgpt_error
    return classify_chatgpt_error(error, **kwargs)


def _clear_credential(entry):
    from hermes_cli.auth_chatgpt import clear_credential
    return clear_credential(entry)


class ChatGPTProfile(ProviderProfile):
    def credential_is_eligible(self, entry):
        from hermes_cli.auth_chatgpt import credential_is_eligible
        return credential_is_eligible(entry)

    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        """This OAuth catalog uses models/slug, not the API-key data/id shape."""
        rows = self._catalog_rows(api_key=api_key, base_url=base_url, timeout=timeout)
        return [row["slug"] for row in rows] if rows is not None else None

    def discover_models(self, **kwargs):
        from agent.credential_pool import load_pool
        entry = load_pool(self.name).select()
        if entry is None:
            return None
        rows = self._catalog_rows(api_key=entry.runtime_api_key, base_url=entry.runtime_base_url)
        return [{"id": row["slug"], "note": row.get("display_name") or row["slug"]}
                for row in rows] if rows is not None else None

    def _catalog_rows(self, *, api_key=None, base_url=None, timeout=8.0):
        if not api_key or (base_url and base_url.rstrip("/") != BASE_URL):
            return None
        from hermes_cli.urllib_security import open_credentialed_url

        request = Request(f"{BASE_URL}/models", headers={
            "Authorization": f"Bearer {api_key}", "Accept": "application/json",
            "User-Agent": _profile_user_agent(),
        })
        try:
            with open_credentialed_url(request, timeout=timeout) as response:
                payload = json.load(response)
            return [item for item in payload.get("models", [])
                    if isinstance(item, dict) and item.get("visibility") == "list"
                    and isinstance(item.get("slug"), str) and item["slug"]]
        except Exception as exc:
            logger.debug("ChatGPT model catalog unavailable: %s", type(exc).__name__)
            return None


register_provider(ChatGPTProfile(
    name="openai-chatgpt", display_name="OpenAI — Sign in with ChatGPT",
    description="Use the selected ChatGPT account through the official Responses API",
    signup_url="https://chatgpt.com", auth_type="oauth_external",
    api_mode="codex_responses", base_url=BASE_URL,
    auth_handler=_auth_handler, refresh_credential=_refresh_credential,
    classify_api_error=_classify_api_error, clear_credential=_clear_credential,
    supports_health_check=False, requires_streaming=True, fixed_api_mode=True,
))
