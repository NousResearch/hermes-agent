"""``no core override`` judges a name by the scope that binds it: a function's local
``key = <core value>`` does not make every ``key`` in the module core, and calling ``add`` on a
name the module only ever binds to a set / dict / list is not calling a same-named plugin patch
helper. Every real rebind shape next to those (aliased bases, static helpers, ``global`` imports,
annotations, a builtin name rebound to a factory, a container name rebound by any other binding)
stays caught."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.plugin_validate_core_override import core_override_findings

_RECORDER = (
    "import run_agent\n"
    "class Recorder:\n"
    "    def add(self, target, fn):\n"
    "        setattr(target, 'hook', fn)\n"
)

CASES = {
    # -- false positives: a plugin's own names and containers -------------------------------------
    "function_local_taint_stays_local": ((
        "def _host_key():\n"
        "    from agent.context_compressor import MODEL_ONLY_KEY\n"
        "    return MODEL_ONLY_KEY\n"
        "def stamp(rows):\n"
        "    key = _host_key()\n"
        "    return [r.get(key) for r in rows]\n"
        "class Scheduler:\n"
        "    def schedule(self, key):\n"
        "        self.key = key\n"
        "_SCHED = Scheduler()\n"
        "def run(key):\n"
        "    _SCHED.schedule(key)\n"
        "    key.x = 1\n"
    ), {}, []),
    "container_add_is_not_plugin_add": (_RECORDER + (
        "def register(ctx):\n"
        "    seen = set()\n"
        "    seen.add(run_agent)\n"
        "    {run_agent}.add(run_agent)\n"
    ), {}, []),
    "closure_param_shadows_parent_taint": ((
        "import importlib\n"
        "def outer(register):\n"
        "    key = importlib.import_module('run_agent')\n"
        "    def callback(key):\n"
        "        key.x = 1\n"
        "    register(callback)\n"
    ), {}, []),
    "nested_global_skips_parent_taint": ((
        "class Plugin:\n"
        "    x = 0\n"
        "core = Plugin()\n"
        "def register(ctx):\n"
        "    def factory():\n"
        "        import run_agent as core\n"
        "        def callback():\n"
        "            global core\n"
        "            core.x = 1\n"
        "        return callback, core\n"
        "    ctx.cb = factory()\n"
    ), {}, []),
    # -- real rebinds: every shape main catches stays caught --------------------------------------
    "closure_sees_parent_taint": ((
        "import importlib\n"
        "def outer():\n"
        "    core = importlib.import_module('run_agent')\n"
        "    def apply():\n"
        "        core.AIAgent.run_conversation = print\n"
        "    apply()\n"
    ), {}, ["core.AIAgent.run_conversation (__init__.py:5)"]),
    "subclass_through_aliased_base": (_RECORDER + (
        "Base = Recorder\n"
        "class Sub(Base):\n"
        "    pass\n"
        "def register(ctx):\n"
        "    rec = Sub()\n"
        "    rec.add(run_agent, print)\n"
        "    ctx.add(run_agent, print)\n"
    ), {}, ["rec.add(run_agent, ...) (__init__.py:10)", "ctx.add(run_agent, ...) (__init__.py:11)"]),
    "walrus_rebinds_a_container_name_to_a_helper": (_RECORDER + (
        "def factory():\n"
        "    return Recorder()\n"
        "def register(ctx):\n"
        "    rec = {}\n"
        "    if (rec := factory()):\n"
        "        rec.add(run_agent, print)\n"
    ), {}, ["rec.add(run_agent, ...) (__init__.py:10)"]),
    "nonlocal_rebinds_a_container_name": (_RECORDER + (
        "def register(ctx):\n"
        "    rec = set()\n"
        "    def swap():\n"
        "        nonlocal rec\n"
        "        rec = Recorder()\n"
        "    swap()\n"
        "    rec.add(run_agent, print)\n"
    ), {}, ["rec.add(run_agent, ...) (__init__.py:11)"]),
    "comprehension_target_shadows_a_container_name": (_RECORDER + (
        "def register(ctx):\n"
        "    rec = set()\n"
        "    [rec.add(run_agent, print) for rec in [Recorder()]]\n"
    ), {}, ["rec.add(run_agent, ...) (__init__.py:7)"]),
    "staticmethod_helper": ((
        "import run_agent\n"
        "class Recorder:\n"
        "    @staticmethod\n"
        "    def bind(target):\n"
        "        setattr(target, 'hook', print)\n"
        "def register(ctx):\n"
        "    Recorder().bind(run_agent)\n"
    ), {}, ["bind(run_agent, ...) (__init__.py:7)"]),
    "global_assigned_in_one_function": ((
        "def init():\n"
        "    global _AGENT\n"
        "    import importlib\n"
        "    _AGENT = importlib.import_module('run_agent')\n"
        "def patch():\n"
        "    _AGENT.AIAgent.run_conversation = print\n"
    ), {}, ["_AGENT.AIAgent.run_conversation (__init__.py:6)"]),
    "global_bound_by_import": ((
        "def init():\n"
        "    global core\n"
        "    import run_agent as core\n"
        "def patch():\n"
        "    core.AIAgent.run_conversation = print\n"
    ), {}, ["core.AIAgent.run_conversation (__init__.py:5)"]),
    "self_slot_container_in_another_class": ((
        "import run_agent\n"
        "class Patches:\n"
        "    def bind(self, target):\n"
        "        setattr(target, 'hook', print)\n"
        "class A:\n"
        "    def __init__(self):\n"
        "        self._patches = {}\n"
        "class B:\n"
        "    def go(self):\n"
        "        self._patches.bind(run_agent)\n"
    ), {}, ["self._patches.bind(run_agent, ...) (__init__.py:10)"]),
    "builtin_name_rebound_to_a_factory": ((
        "import run_agent\n"
        "from .factory import make_recorder as set\n"
        "def register(ctx):\n"
        "    rec = set()\n"
        "    rec.bind(run_agent)\n"
    ), {"factory.py": (
        "class Recorder:\n"
        "    def bind(self, target):\n"
        "        setattr(target, 'hook', print)\n"
        "def make_recorder():\n"
        "    return Recorder()\n"
    )}, ["rec.bind(run_agent, ...) (__init__.py:5)"]),
    "star_import_shadows_a_builtin_constructor": ((
        "import run_agent\n"
        "from .factory import *\n"
        "def register(ctx):\n"
        "    frozenset().bind(run_agent)\n"
    ), {"factory.py": (
        "class Recorder:\n"
        "    def bind(self, target):\n"
        "        setattr(target, 'hook', print)\n"
        "def frozenset():\n"
        "    return Recorder()\n"
    )}, ["bind(run_agent, ...) (__init__.py:4)"]),
    "annotation_expression": ((
        "import run_agent\n"
        "def hook(x: setattr(run_agent, 'hook', print)):\n"
        "    pass\n"
    ), {}, ["setattr(run_agent, ...) (__init__.py:2)"]),
}


@pytest.mark.parametrize(("init_py", "extra", "expected"), CASES.values(), ids=CASES.keys())
def test_core_override_scope(tmp_path: Path, init_py: str, extra: dict, expected: list):
    plugin = tmp_path / "p"
    plugin.mkdir()
    (plugin / "__init__.py").write_text(init_py, encoding="utf-8")
    for name, src in extra.items():
        (plugin / name).write_text(src, encoding="utf-8")

    assert core_override_findings(plugin) == expected
