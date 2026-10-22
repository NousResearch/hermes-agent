"""DarkBloom provider profile.

DarkBloom is a decentralized private-inference network: OpenAI-compatible
requests are routed through verified Apple Silicon Macs with hardware-backed
encryption, so neither the coordinator nor the serving node sees plaintext.
Public alpha, no platform fee.
"""

from hermes_cli import __version__ as _HERMES_VERSION
from providers import register_provider
from providers.base import ProviderProfile

darkbloom = ProviderProfile(
    name="darkbloom", aliases=("dark", "db"), display_name="DarkBloom",
    description="DarkBloom — private inference on verified Apple Silicon Macs",
    signup_url="https://console.darkbloom.dev/api-console",
    env_vars=("DARKBLOOM_API_KEY", "DARKBLOOM_BASE_URL"),
    base_url="https://api.darkbloom.dev/v1", auth_type="api_key",
    # Attribution so DarkBloom can identify Hermes Agent traffic.
    default_headers={"User-Agent": f"HermesAgent/{_HERMES_VERSION}"},
    default_aux_model="Qwen3.5-9B",
    # Curated fallback (mirrors GET /v1/models) shown when the live fetch fails.
    fallback_models=(
        "gemma-4-26b", "Qwen3.5-9B", "gpt-oss-20b", "qwen3.5-35b-a3b",
        "EigenLabs/Qwen3.8-27B-4bit-mtp", "ternary-bonsai-2-27b",
        "nvidia-nemotron-3.5-lightning", "qwen3.6-35b-a3b-vl-mtp-mxfp8",
    ),
)

register_provider(darkbloom)
