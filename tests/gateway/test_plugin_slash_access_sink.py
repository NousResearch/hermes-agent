"""Dependency-free regression at the real plugin dispatch sink; inputs are synthetic."""
import ast
import asyncio
from contextlib import nullcontext
from enum import Enum
import importlib.util
import logging
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[2]

def native_function(path, name, namespace):
    tree = ast.parse((BASE / path).read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(BASE / path), 'exec'), namespace)
    return namespace[name]

class PluginAccessSinkTests(unittest.TestCase):
    def test_both_spellings_obey_scope_policy_before_handler(self):
        for spelling in ('memory-history', 'memory_history'):
            for user, extra, allowed in (
                ('reader', {'group_allow_admin_from': ['owner']}, False),
                ('owner', {'group_allow_admin_from': ['owner']}, True),
                ('reader', {'group_allow_admin_from': ['owner'],
                            'group_user_allowed_commands': ['memory-history']}, True),
                ('reader', {}, True),
            ):
                with self.subTest(spelling=spelling, user=user, extra=extra):
                    self._case(spelling, user, extra, allowed)

    def _case(self, spelling, user, extra, allowed):
        spec = importlib.util.spec_from_file_location('gateway.slash_access',
                                                     BASE / 'gateway/slash_access.py')
        policy = importlib.util.module_from_spec(spec)
        gateway = types.ModuleType('gateway'); gateway.__path__ = []
        hermes = types.ModuleType('hermes_cli'); hermes.__path__ = []
        plugins = types.ModuleType('hermes_cli.plugins')
        calls = []
        plugins.get_plugin_command_handler = lambda name: (
            lambda args: calls.append(args) or 'SYNTHETIC history'
        ) if name == 'memory-history' else None
        modules = {'gateway': gateway, 'gateway.slash_access': policy,
                   'hermes_cli': hermes, 'hermes_cli.plugins': plugins}
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(policy)
            env = {'asyncio': asyncio, 'logger': logging.getLogger('SYNTHETIC'),
                   't': lambda *a, **k: 'DENIED',
                   'build_session_context': lambda *a: types.SimpleNamespace()}
            check = native_function('gateway/run_busy.py', '_check_slash_access', env)
            dispatch = native_function('gateway/run_inbound.py',
                                       '_hm_dispatch_quick_and_plugin_commands', env)
            class Runner:
                _draining = False
                config = types.SimpleNamespace(multiplex_profiles=False, platforms={
                    'telegram': types.SimpleNamespace(extra=extra)})
                _check_slash_access = check
                def _hm_quick_commands(self): return {}
                def _session_key_for_source(self, source): return 'SYNTHETIC-session'
                def _session_env_scope(self, context): return nullcontext()
                async def _run_in_executor_with_context(self, fn, *args): return fn(*args)
            class Platform(str, Enum):
                TELEGRAM = 'telegram'
            source = types.SimpleNamespace(platform=Platform.TELEGRAM, chat_type='group',
                                           user_id=user)
            event = types.SimpleNamespace(get_command_args=lambda: 'SYNTHETIC args')
            handled, result, command = asyncio.run(dispatch(Runner(), event, source, spelling))
            self.assertTrue(handled)
            self.assertEqual(command, spelling)
            self.assertEqual(result, 'SYNTHETIC history' if allowed else 'DENIED')
            self.assertEqual(calls, ['SYNTHETIC args'] if allowed else [])

if __name__ == '__main__':
    unittest.main()
