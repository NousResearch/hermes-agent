"""Rapid (MLX) provider profile.

Rapid is an OpenAI/Anthropic-compatible local inference server for
Apple Silicon, built on top of mlx-lm and mlx-vlm. It serves models
at ``http://127.0.0.1:8000/v1`` by default; the user starts it with
``rapid-mlx serve <alias>`` from the Rapid-MLX CLI
(https://github.com/raullenchai/Rapid-MLX).

Mirrors the LM Studio profile shape since the user-facing setup is
the same (local endpoint, optional API key, configurable base URL).
"""

from providers import register_provider
from providers.base import ProviderProfile


# NOTE: ``base_url`` is intentionally empty (mirrors the ``custom`` plugin).
# The actual runtime endpoint is owned by
# ``hermes_cli/auth.py::PROVIDER_REGISTRY["rapid-mlx"].inference_base_url``
# and ``hermes_cli/providers.py::HERMES_OVERLAYS["rapid-mlx"].base_url_override``,
# both set to ``http://127.0.0.1:8000/v1``. The plugin must NOT also
# advertise that URL via ``base_url`` because ``agent/model_metadata.py``
# auto-extends ``_URL_TO_PROVIDER`` from each ProviderProfile's hostname —
# a hardcoded ``127.0.0.1`` would then claim every loopback endpoint
# (including unrelated ``custom`` / vLLM / llama.cpp setups on different
# ports) as ``rapid-mlx``, breaking the ``custom`` profile's local
# context-length probing.
rapid_mlx = ProviderProfile(
    name="rapid-mlx",
    aliases=("rapid", "rapidmlx", "rapid_mlx"),
    display_name="Rapid (MLX)",
    description="Rapid — OpenAI/Anthropic-compatible MLX inference for Apple Silicon",
    signup_url="https://github.com/raullenchai/Rapid-MLX",
    env_vars=("RAPID_MLX_API_KEY", "RAPID_MLX_BASE_URL"),
    base_url="",
    # ``models_url`` advertises the default catalog endpoint so the
    # ``hermes model`` picker can probe live for the alias the user is
    # serving, without going through ``base_url`` (which we keep empty
    # for the URL-inference reason documented above). Default
    # ``fetch_models`` already prefers ``models_url`` when set, so no
    # subclass is needed.
    models_url="http://127.0.0.1:8000/v1/models",
    auth_type="api_key",
    # The default endpoint is user-configurable (``--port`` on
    # ``rapid-mlx serve``, or ``RAPID_MLX_BASE_URL``), so a static
    # doctor probe would produce false positives whenever the user
    # has chosen a non-default port or hasn't started the server.
    # Doctor still surfaces auth-key status via the ProviderConfig
    # path in hermes_cli/auth.py.
    supports_health_check=False,
)

register_provider(rapid_mlx)
