"""Modal provider profile (Modal Endpoints — inference.<region>.modal.direct).

The wire is OpenAI-compatible chat completions; auth is the combined
``wk-<id>.ws-<secret>`` proxy token sent as Bearer. The ``/models`` catalog is
account-scoped and carries per-model ``reasoning_options`` and
``context_length``, so ``fetch_models()`` stashes both for the hot-path hooks
(``supported_reasoning_efforts`` / ``get_model_context_length``). Model IDs are
endpoint-scoped and rotate with deployments, so there are no ``fallback_models``
or static ``model_capabilities`` — the live catalog is the source of truth.
"""

import json
import logging
import urllib.request
from typing import Any

from hermes_cli.version_info import hermes_user_agent
from providers import register_provider
from providers.base import LazyHeaders, ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

# model id -> {"efforts": tuple[str, ...], "context_length": int}; filled by fetch_models().
_CATALOG: dict[str, dict[str, Any]] = {}


class ModalProfile(ProviderProfile):
    """Cache the Modal catalog's per-model metadata alongside the id list."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        effective_base = (base_url or "").strip() or self.base_url
        if not effective_base:
            return None
        url = effective_base.rstrip("/") + "/models"
        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        for k, v in self.default_headers.items():
            req.add_header(k, v)
        try:
            from hermes_cli.urllib_security import open_credentialed_url

            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            for item in items:
                if not isinstance(item, dict) or not item.get("id"):
                    continue
                entry: dict[str, Any] = {}
                for opt in item.get("reasoning_options") or ():
                    if (
                        isinstance(opt, dict)
                        and opt.get("type") == "effort"
                        and opt.get("values")
                    ):
                        entry["efforts"] = tuple(opt["values"])
                if isinstance(item.get("context_length"), int):
                    entry["context_length"] = item["context_length"]
                _CATALOG[item["id"]] = entry
            return [
                item["id"]
                for item in items
                if isinstance(item, dict) and item.get("id")
            ]
        except Exception as exc:
            # catalog probe is best-effort: any network/parse failure -> None
            logger.debug("fetch_models(%s): %s", self.name, exc, exc_info=True)
            return None

    def supported_reasoning_efforts(self, model: str | None) -> tuple[str, ...] | None:
        """Per-model effort vocabulary from the live catalog. ``()`` when the
        catalog listed the model with no reasoning options (omit reasoning
        fields); ``None`` while the catalog is cold."""
        entry = _CATALOG.get(model or "")
        if entry is None:
            return None
        return entry.get("efforts", ())

    def get_model_context_length(self, model: str) -> int | None:
        entry = _CATALOG.get(model or "")
        return entry.get("context_length") if entry else None


modal = ModalProfile(
    name="modal",
    aliases=("modal-labs", "modal-direct", "modal-endpoints"),
    display_name="Modal",
    description="Modal Endpoints — OpenAI-compatible inference on Modal",
    signup_url="https://modal.com/settings",
    env_vars=("MODAL_PROXY_TOKEN", "MODAL_BASE_URL"),
    base_url="https://inference.us-west.modal.direct/v1",
    metadata_hostnames=("modal.direct", "modal.run"),
    default_headers=LazyHeaders(
        {
            "User-Agent": lambda: hermes_user_agent("HermesAgent"),
        }
    ),
)

register_provider(modal)
