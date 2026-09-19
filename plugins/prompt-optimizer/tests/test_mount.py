"""插件发现/挂载回归测试 —— 模拟 web_server 的 _discover_dashboard_plugins
与 _mount_plugin_api_routes 对 prompt-optimizer 的加载路径。

运行(仓库根 = <repo>/plugins/prompt-optimizer):
    python tests/test_mount.py
"""
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PLUGIN_DIR = os.path.dirname(TESTS_DIR)          # <repo>/plugins/prompt-optimizer
REPO_ROOT = os.path.dirname(os.path.dirname(PLUGIN_DIR))  # hermes-agent 仓库根
sys.path.insert(0, REPO_ROOT)

# 1. 发现:扫描 dashboard 插件
from hermes_cli.web_server import _get_dashboard_plugins

plugins = _get_dashboard_plugins(force_rescan=True)
po = next((p for p in plugins if p['name'] == 'prompt-optimizer'), None)
print('discovered:', po is not None)
if po:
    print('  source:', po['source'], '| api:', po['_api_file'], '| has_api:', po['has_api'])

# 2. enabled 门:user 插件必须在 plugins.enabled;bundled 只需不在 disabled
from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set

enabled = _get_enabled_set()
disabled = _get_disabled_set()
if po and po['source'] == 'user':
    ok = 'prompt-optimizer' in enabled and 'prompt-optimizer' not in disabled
    print('enabled gate (user):', ok)
else:
    ok = 'prompt-optimizer' not in disabled
    print('enabled gate (bundled, not disabled):', ok)

# 3. 挂载:模拟 _mount_plugin_api_routes 的导入(不启动服务器,只 import 模块)
import importlib.util
from pathlib import Path

dashboard_dir = Path(PLUGIN_DIR) / 'dashboard'
api_path = dashboard_dir / 'plugin_api.py'
module_name = 'hermes_dashboard_plugin_prompt-optimizer'
spec = importlib.util.spec_from_file_location(module_name, api_path)
mod = importlib.util.module_from_spec(spec)
sys.modules[module_name] = mod
spec.loader.exec_module(mod)
router = getattr(mod, 'router', None)
print('router imported:', router is not None)
routes = [r.path for r in router.routes]
print('routes:', routes)

assert po is not None, 'plugin not discovered'
assert po['_api_file'] == 'plugin_api.py'
assert ok, 'enabled gate failed'
assert router is not None and routes == ['/optimize'], f'router mismatch: {routes}'
print('\nPASS')
