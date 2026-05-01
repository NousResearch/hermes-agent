import sys
sys.path.append("/home/ubuntu/workspaces/oss/hermes-agent")
from plugins.memory.honcho.client import get_honcho_client, HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager

cfg = HonchoClientConfig.from_global_config()
print(f"base_url={cfg.base_url}, api_key={cfg.api_key[:10]}...")
client = get_honcho_client(cfg)
manager = HonchoSessionManager(honcho=client, config=cfg)
try:
    session = manager.get_or_create("test_session")
    print("Success:", session)
except Exception as e:
    import traceback
    traceback.print_exc()
