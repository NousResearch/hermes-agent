import subprocess
import sys


def test_acp_agent_executor_does_not_block_interpreter_exit():
    script = """
import sys
import time
import types
sys.path.insert(0, {repo_root!r})
acp = types.ModuleType('acp')
acp.run_agent = lambda *args, **kwargs: None
def _acp_getattr(name):
    value = type(name, (), {{}})
    setattr(acp, name, value)
    return value
acp.__getattr__ = _acp_getattr
schema = types.ModuleType('acp.schema')
def _schema_getattr(name):
    value = type(name, (), {{}})
    setattr(schema, name, value)
    return value
schema.__getattr__ = _schema_getattr
sys.modules['acp'] = acp
sys.modules['acp.schema'] = schema
from acp_adapter import server
server._executor.submit(time.sleep, 120)
time.sleep(0.3)
server._executor.shutdown(wait=False, cancel_futures=True)
print('main-done', flush=True)
""".format(repo_root=str(_repo_root()))

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert proc.returncode == 0
    assert "main-done" in proc.stdout


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2]
