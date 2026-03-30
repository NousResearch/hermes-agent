from run_agent import AIAgent
import asyncio
from hermes_cli.runtime_provider import resolve_runtime_provider

cfg = resolve_runtime_provider(requested="google")
agent = AIAgent(
    model="google/gemini-3.1-pro-preview",
    api_mode=cfg["api_mode"],
    api_key=cfg["api_key"],
    base_url=cfg["base_url"],
    provider=cfg["provider"],
    max_iterations=1,
    quiet_mode=True
)

res = agent.chat("What is 2+2? Answer only with the number.")
print(f"Result: {res}")
