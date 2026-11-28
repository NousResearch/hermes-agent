"""Infersia provider profile.

Infersia serves open-weight models on dedicated GPUs through an
OpenAI-compatible chat-completions endpoint, and publishes the exact
quantisation, hardware and measured latency behind every model.

Address models by their catalogue ID, e.g. ``deepseek/deepseek-v4-flash-0731``
or ``qwen/qwen3.6-35b-a3b``. Appending ``:free`` selects a rate-limited free
variant where one is published.

DeepSeek V4 Flash is served at its full 1,048,576-token window rather than a
reduced slice, which is the main reason to reach for this provider on
long-context work.
"""

import json
import logging
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)


def _is_chat_model(entry: dict) -> bool:
    """Return True when *entry* is a model the chat_completions transport can use.

    An OpenAI-shaped ``/v1/models`` list describes an account's whole catalogue,
    not one endpoint's worth of it, and the model object carries no capability
    field — so an entry alone cannot say which route answers it. Infersia
    publishes an ``architecture`` block for exactly this, and the test is
    ``text`` on BOTH sides: a chat model takes text in and emits text out.

    Checking the output side alone is not sufficient. Speech-to-text is
    ``audio->text``, so its output modality *is* ``["text"]`` and it passes a
    one-sided test while answering ``/v1/audio/transcriptions`` rather than
    ``/v1/chat/completions``.

    This is an allow-list, so a modality we have not accounted for is excluded
    by default: a future non-text model drops out of the picker until this
    profile is taught about it, instead of appearing and failing on first use.

    An entry with no ``architecture`` block is kept. Absence is no information
    rather than evidence of a non-chat model, and ``fallback_models`` is empty
    here, so failing closed on it would leave an empty picker with nothing to
    explain it.
    """
    architecture = entry.get("architecture")
    if not isinstance(architecture, dict):
        return True

    def _has_text(key: str) -> bool:
        modalities = architecture.get(key)
        if not isinstance(modalities, list):
            return True
        return any(str(m).strip().lower() == "text" for m in modalities)

    return _has_text("input_modalities") and _has_text("output_modalities")


class InfersiaProfile(ProviderProfile):
    """Infersia — live catalogue, narrowed to the chat-capable models."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch the live catalogue and keep only its chat models.

        Same request as the base implementation; it is repeated here rather
        than delegated because the base returns bare ID strings, and the
        capability signal being filtered on lives in the fields those strings
        are read out of.

        ``fallback_models`` is empty, so whatever this returns *is* the picker.
        """
        effective_base = base_url or self.base_url
        url = (self.models_url or "").strip()
        if not url:
            if not effective_base:
                return None
            url = effective_base.rstrip("/") + "/models"

        from hermes_cli.urllib_security import open_credentialed_url

        req = urllib.request.Request(url)
        if api_key:
            req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        for k, v in self.default_headers.items():
            req.add_header(k, v)

        try:
            with open_credentialed_url(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            items = data if isinstance(data, list) else data.get("data", [])
            return [
                m["id"]
                for m in items
                if isinstance(m, dict) and "id" in m and _is_chat_model(m)
            ]
        except Exception as exc:
            logger.debug("fetch_models(infersia): %s", exc)
            return None


infersia = InfersiaProfile(
    name="infersia",
    aliases=("infersia-ai",),
    display_name="Infersia",
    description="Infersia — open-weight models with published quantisation",
    signup_url="https://infersia.com/dashboard/keys",
    env_vars=("INFERSIA_API_KEY", "INFERSIA_BASE_URL"),
    base_url="https://api.infersia.com/v1",
    auth_type="api_key",
    # Images are accepted inside tool-result messages. Re-verified against
    # qwen/qwen3.6-35b-a3b on 2026-08-06, after Step 3.7 Flash (the model this
    # was originally checked against) was retired: a two-pixel test image sent
    # as a base64 data URI came back correctly described.
    supports_vision=True,
    # Prefix caching is automatic and unkeyed here. The endpoint tolerates a
    # ``prompt_cache_key`` field rather than honouring it, and this flag is
    # documented as opt-in for endpoints that explicitly accept it, so
    # claiming it would advertise behaviour that does not exist.
    supports_prompt_cache_key=False,
    # Auxiliary model for cheap side tasks. Qwen3 8B is the smallest thing in
    # the catalogue and the only hardcoded model id here; everything else is
    # discovered live.
    default_aux_model="qwen/qwen3-8b",
    # Deliberately empty, following the DeepInfra profile's reasoning: the
    # live catalogue at ``{base_url}/models`` is the source of truth, and it
    # returns OpenAI-shaped entries with pricing and context length. When the
    # fetch fails the picker shows nothing, which is better than routing
    # someone to a model id that has since been retired — this catalogue is
    # still changing week to week.
    fallback_models=(),
)

register_provider(infersia)
