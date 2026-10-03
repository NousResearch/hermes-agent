"""NVIDIA NIM provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _classify_api_error(
    error: Exception, *, status_code: int | None, error_code: str,
    message: str, body: Any, model: str,
) -> dict[str, Any] | None:
    """NVIDIA exposes global model IDs that may not be provisioned for this account."""
    normalized = str(message or "").lower()
    if status_code == 410:
        return {"reason": "model_not_found", "retryable": False, "should_fallback": True}
    if status_code == 404 and "function " in normalized and "not found for account" in normalized:
        return {
            "reason": "model_entitlement",
            "retryable": False,
            "should_rotate_credential": True,
            "should_fallback": True,
        }
    return None


class NvidiaProviderProfile(ProviderProfile):
    """NVIDIA NIM accepts a stricter ToolMessage schema than most OpenAI-compatible APIs."""

    @staticmethod
    def _needs_strip(msg: Any) -> bool:
        return isinstance(msg, dict) and msg.get("role") == "tool" and ("name" in msg or "tool_name" in msg)

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Copy-on-write: only tool messages that lose a field are copied
        (no deep copy of large tool outputs); untouched input returned as-is."""
        if not any(self._needs_strip(msg) for msg in messages):
            return messages
        return [
            {k: v for k, v in msg.items() if k not in ("name", "tool_name")} if self._needs_strip(msg) else msg
            for msg in messages
        ]


nvidia = NvidiaProviderProfile(
    name="nvidia", aliases=("nvidia-nim", "nim", "build-nvidia", "nemotron"), env_vars=("NVIDIA_API_KEY",), display_name="NVIDIA NIM",
    description="NVIDIA NIM — accelerated inference", signup_url="https://build.nvidia.com/",
    fallback_models=("nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3-super-120b-a12b"),
    base_url="https://integrate.api.nvidia.com/v1", default_max_tokens=16384,
    classify_api_error=_classify_api_error,
)

register_provider(nvidia)
