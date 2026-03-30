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
    max_iterations=3,
    quiet_mode=False
)

res = agent.chat("Use the terminal tool to run 'echo \"Hello World from Native Gemini SDK\"'. Return only the output of the command.")
print(f"Final Result: {res}")
