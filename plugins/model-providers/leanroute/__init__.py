from providers import ProviderProfile, register_provider

LEANROUTE_BASE_URL = "https://api.leanroute.dev/v1"

profile = ProviderProfile(
    id="leanroute",
    name="Leanroute",
    api_mode="chat_completions",
    base_url=LEANROUTE_BASE_URL,
    env_vars=["LEANROUTE_API_KEY"],
    default_model="deepseek/deepseek-chat",
    default_aux_model="deepseek/deepseek-chat",
    supports_streaming=True,
    supports_tools=True,
    fallback_models=[
        "deepseek/deepseek-chat",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-4o-mini",
        "google/gemini-2.5-flash",
    ],
)

register_provider(profile)
