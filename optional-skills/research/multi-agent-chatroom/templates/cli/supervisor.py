"""Start supervisor agent."""
import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.supervisor import Supervisor
from server.config import load_config, get_supervisor_config, get_workflow_config

async def main():
    config = load_config()
    sup_cfg = get_supervisor_config(config)
    wf_cfg = get_workflow_config(config)

    print(f"👔 Supervisor: {sup_cfg['name']}")
    print(f"   Provider: {sup_cfg['provider']}")
    print(f"   Model:    {sup_cfg['model']}")

    agent = Supervisor(
        name=sup_cfg["name"],
        max_iterations=wf_cfg.get("max_iterations", 50),
        review_timeout=wf_cfg.get("review_timeout_seconds", 180),
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
