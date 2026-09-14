"""Personal X uses its owning runtime's configured model, not another profile."""
from pathlib import Path
import sys
from hermes_cli import config as runtime_config

# Flat content-engine audit utilities also have a 'tools' directory. Resolve the
# native Hermes toolkit before provider clients import tools.threat_patterns.
core_root = str(Path(runtime_config.__file__).resolve().parents[1])
if core_root not in sys.path:
    sys.path.insert(0, core_root)
else:
    sys.path.remove(core_root)
    sys.path.insert(0, core_root)
import tools as _native_tools
load_config = runtime_config.load_config
from llm_generate import _llm_configs, _call_llm_chain


def call_x_model(system, user, *, timeout=90, max_tokens=3000):
    # Configuration and credentials stay profile-scoped. No hardcoded model,
    # provider switch, config write, agent subprocess or static-template fallback.
    configs = _llm_configs(config=load_config())
    return _call_llm_chain(system,user,timeout=timeout,max_tokens=max_tokens,configs=configs)
