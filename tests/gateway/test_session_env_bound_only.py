"""Public session-context API contract; stdlib only, no gateway or owner data."""
import contextvars
import ast
import __future__
from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gateway.session_context import get_session_env, set_session_vars, clear_session_vars


class BoundOnlySessionEnv(unittest.TestCase):
    def test_actual_telegram_source_and_gateway_setter_bind_plain_dm(self):
        # Execute unchanged source classes/setter without importing the full gateway
        # and its optional dependencies. No private ContextVars or latch are touched.
        root = Path(__file__).resolve().parents[2]
        nodes = []
        for rel, name in [('gateway/config.py', 'Platform'), ('gateway/session.py', 'SessionSource')]:
            tree = ast.parse((root / rel).read_text(encoding='utf-8'))
            nodes.append(next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name))
        runner = next(node for node in ast.parse((root / 'gateway/run.py').read_text(encoding='utf-8')).body
                      if isinstance(node, ast.ClassDef) and node.name == 'GatewayRunner')
        nodes.append(next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == '_set_session_env'))
        ns = dict(globals())
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<native-source-contract>', 'exec',
                     flags=__future__.annotations.compiler_flag), ns)
        source = ns['SessionSource'](platform=ns['Platform'].TELEGRAM, chat_id='SYNTHETIC-chat', chat_type='dm')
        def check():
            tokens = ns['_set_session_env'](SimpleNamespace(adapters={}),
                                            SimpleNamespace(source=source, session_key='SYNTHETIC-session'))
            self.assertIs(type(source.chat_type), str)
            self.assertEqual(get_session_env('HERMES_SESSION_PLATFORM', allow_env_fallback=False), 'telegram')
            self.assertEqual(get_session_env('HERMES_SESSION_CHAT_TYPE', allow_env_fallback=False), 'dm')
            clear_session_vars(tokens)
        contextvars.Context().run(check)

    def test_unbound_ignores_stale_environment_only_when_requested(self):
        def check():
            self.assertEqual(get_session_env('HERMES_SESSION_PLATFORM'), 'telegram')
            self.assertEqual(get_session_env('HERMES_SESSION_PLATFORM', allow_env_fallback=True), 'telegram')
            self.assertEqual(get_session_env('HERMES_SESSION_PLATFORM', allow_env_fallback=False), '')
            self.assertEqual(get_session_env('HERMES_SESSION_CHAT_TYPE', 'unknown', allow_env_fallback=False), 'unknown')
        with patch.dict(os.environ, {'HERMES_SESSION_PLATFORM': 'telegram', 'HERMES_SESSION_CHAT_TYPE': 'dm'}):
            contextvars.Context().run(check)

    def test_bound_value_and_explicit_empty_override_environment(self):
        def check():
            tokens = set_session_vars(platform='telegram', chat_type='group')
            self.assertEqual(get_session_env('HERMES_SESSION_CHAT_TYPE', allow_env_fallback=False), 'group')
            self.assertEqual(get_session_env('HERMES_SESSION_CHAT_TYPE'), 'group')
            clear_session_vars(tokens)
            self.assertEqual(get_session_env('HERMES_SESSION_PLATFORM', 'unknown', allow_env_fallback=False), '')
            self.assertEqual(get_session_env('HERMES_SESSION_PLATFORM', 'unknown'), '')
        with patch.dict(os.environ, {'HERMES_SESSION_PLATFORM': 'telegram', 'HERMES_SESSION_CHAT_TYPE': 'dm'}):
            contextvars.Context().run(check)

    def test_unknown_name_and_missing_environment_preserve_default_contract(self):
        def check():
            self.assertEqual(get_session_env('SYNTHETIC_CONTEXT_NAME', 'missing'), 'synthetic value')
            self.assertEqual(get_session_env('SYNTHETIC_CONTEXT_NAME', 'missing', allow_env_fallback=False), 'missing')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(get_session_env('HERMES_SESSION_PLATFORM', 'missing'), 'missing')
                self.assertEqual(get_session_env('HERMES_SESSION_PLATFORM', 'missing', allow_env_fallback=False), 'missing')
        with patch.dict(os.environ, {'SYNTHETIC_CONTEXT_NAME': 'synthetic value'}):
            contextvars.Context().run(check)


if __name__ == '__main__':
    unittest.main()
