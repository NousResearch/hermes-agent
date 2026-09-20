"""An external provider update loads as a new generation; a session holding the old provider keeps its module."""
import importlib
import os
from pathlib import Path
import shutil
import sys

import pytest

from hermes_cli.plugin_installation import plugin_installation_lock
from plugins import memory


PROVIDER = '''from agent.memory_provider import MemoryProvider
from .values import VALUE
STATE = []
class Provider(MemoryProvider):
    name = 'generation_probe'
    def is_available(self): return VALUE != 'unready'
    def initialize(self, *args, **kwargs): pass
    def get_tool_schemas(self): return [{'name': 'generation_tool', 'description': VALUE}]
    def delayed(self):
        from .late.helper import VALUE as late
        import importlib
        return late, importlib.import_module('.late.helper', __package__).VALUE
    def record(self, value): STATE.append(value); return list(STATE)

def register(ctx):
    ctx.register_memory_provider(Provider())
    ctx.register_hook('pre_llm_call', lambda **kwargs: {'context': VALUE})
'''


def package(path, value):
    (path / 'late').mkdir(parents=True)
    (path / 'late' / 'helper.py').write_text(f'VALUE = {value!r}\n')
    (path / 'values.py').write_text(f'VALUE = {value!r}\n')
    (path / '__init__.py').write_text(PROVIDER)
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'A'))
    monkeypatch.delenv('HERMES_ENABLE_PROJECT_PLUGINS', raising=False)
    monkeypatch.chdir(tmp_path)
    before = set(sys.modules)
    try:
        yield tmp_path / 'A'
    finally:
        for name in set(sys.modules) - before:
            if name.startswith(memory._USER_NAMESPACE):
                sys.modules.pop(name, None)


def load(name='generation_probe'):
    return memory.load_memory_provider(name, register_skills=False)


def test_old_sessions_keep_their_generation_and_new_loads_see_the_update(tmp_path, home):
    target = package(home / 'plugins' / 'generation_probe', 'before')
    old = load()
    os.utime(target / 'values.py')  # a new mtime with the same bytes is the same generation
    assert type(load()) is type(old)
    old_module = sys.modules[type(old).__module__]
    assert old.record('retained') == ['retained']
    staged = package(tmp_path / 'staged', 'unready')
    with plugin_installation_lock(home):
        target.rename(tmp_path / 'retired')
        staged.rename(target)
    shutil.rmtree(tmp_path / 'retired')
    fresh = load()
    assert not fresh.is_available() and fresh.delayed() == ('unready', 'unready') and fresh.record('fresh') == ['fresh']
    assert old.delayed() == ('before', 'before')  # delayed relative imports stay in the old generation
    assert old.record('still old') == ['retained', 'still old'] and old.is_available()
    assert importlib.import_module(type(old).__module__) is old_module
    package(home / 'plugins' / 'holographic', 'unready')  # an external homonym never replaces a bundled provider
    assert type(load('holographic')).__module__.startswith('plugins.memory.holographic')


def test_host_registrations_stay_with_the_generation_a_session_holds(tmp_path, home):
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    target = package(home / 'plugins' / 'generation_probe', 'before')
    try:
        old = load()
        assert manager.invoke_hook('pre_llm_call', session_id='') == [{'context': 'before'}]
        shutil.rmtree(target)
        package(target, 'after')
        fresh = load()  # a new session gets the update
        assert fresh.get_tool_schemas()[0]['description'] == 'after' and old.get_tool_schemas()[0]['description'] == 'before'
        assert manager.invoke_hook('pre_llm_call', session_id='') == [{'context': 'before'}]
    finally:
        manager.unload()


def test_warm_up_and_runtime_share_one_generation_module(home):
    package(home / 'plugins' / 'generation_probe', 'before')
    assert memory.import_memory_provider_module('generation_probe') is True

    def packages():
        return [n for n in sys.modules if n.startswith(f'{memory._USER_NAMESPACE}.generation_probe') and n.count('.') == 1]

    warmed = packages()
    assert type(load()).__module__ == warmed[0] and packages() == warmed
