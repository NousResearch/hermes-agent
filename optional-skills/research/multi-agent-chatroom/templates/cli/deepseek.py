"""Start the coding/research agent (configurable model)."""
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.deepseek_researcher import DeepSeekResearcher
from server.config import load_config, get_coding_config, get_project_config

async def main():
    config = load_config()
    coding_cfg = get_coding_config(config)
    project_cfg = get_project_config(config)
    workdir = project_cfg.get("workdir", ".")

    print(f"🤖 Coding Agent: {coding_cfg['name']}")
    print(f"   Provider: {coding_cfg['provider']}")
    print(f"   Model:    {coding_cfg['model']}")
    print(f"   Workdir:  {Path(workdir).resolve()}")

    agent = DeepSeekResearcher(
        name=coding_cfg["name"],
        provider=coding_cfg["provider"],
        model=coding_cfg["model"],
        workdir=Path(workdir).resolve(),
        temperature=coding_cfg.get("temperature", 0.3),
        max_tokens=coding_cfg.get("max_tokens", 16384),
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
