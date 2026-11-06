"""HX015: process-wide mutable state is refused in every module the hosted entry point imports.

A planted module-level dict reachable from the hosted roots blocks the ratchet; the same dict in
a module the roots never import does not; state that existed before is baselined; the two
declarations (``ScopedCache``, ``tenant_neutral`` with a reason) pass.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess

import pytest

from scripts.code_health import config
from scripts.code_health.hosted_graph import reachable
from scripts.code_health.cli import run
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from tests.scripts.test_code_health import REPO, _commit, _repo, _verdict


def _hits(src: str) -> list[int]:
    return sorted(set(CHECKERS["HX015"](canonical_tree(ast.parse(src)), Ctx())))


@pytest.mark.parametrize("src, flagged", [
    ("_CACHE = {}\n", True),
    ("_SEEN: set = set()\n", True),
    ("import collections\n_BY = collections.defaultdict(list)\n", True),
    ("_ITEMS = []\n", True),
    ("_TABLE = {'a': 1}\ndef put(k):\n    _TABLE[k] = 2\n", True),
    ("_NAMES = ['a']\ndef add(n):\n    _NAMES.append(n)\n", True),
    ("_MEMO = {'x': 1}\ndef reset():\n    global _MEMO\n    _MEMO = {}\n", True),
    ("try:\n    _X = {}\nexcept ImportError:\n    pass\n", True),
    ("import functools\n@functools.lru_cache(maxsize=8)\ndef f(x):\n    return x\n", True),
    ("from functools import cache\n@cache\ndef f(x):\n    return x\n", True),
    ("import functools\nclass C:\n    @functools.lru_cache\n    def m(self):\n        return 1\n", True),
    ("class C:\n    _live = {}\n    def add(self, k):\n        self._live[k] = 1\n", True),
    ("class C:\n    _live = []\n    @classmethod\n    def add(cls, k):\n        cls._live.append(k)\n", True),
    ("class C:\n    current = None\n    @classmethod\n    def set(cls, v):\n        cls.current = v\n", True),
    # constants, per-instance state, and the two declarations
    ("_TABLE = {'a': 1, 'b': 2}\n", False),
    ("_NAMES = ('a', 'b')\n", False),
    ("__all__ = []\n", False),
    ("class C:\n    _live = {}\n    def __init__(self):\n        self._live = {}\n    def add(self, k):\n"
     "        self._live[k] = 1\n", False),
    ("def f():\n    seen = {}\n    seen[1] = 2\n", False),
    ("from hermes_scope_cache import ScopedCache\n_CACHE = ScopedCache('c')\n", False),
    ("from hermes_scope_cache import tenant_neutral\n"
     "_CATALOG = tenant_neutral({}, reason='model catalog, same for every tenant')\n", False),
    ("import functools\nfrom hermes_scope_cache import tenant_neutral\n"
     "@tenant_neutral(reason='pure parse of a constant')\n@functools.cache\ndef f(x):\n    return x\n", False),
    # a declaration needs a literal, non-empty reason
    ("from hermes_scope_cache import tenant_neutral\n_C = tenant_neutral({}, reason='')\n", True),
    ("from hermes_scope_cache import tenant_neutral\n_C = tenant_neutral({}, reason=WHY)\n", True),
])
def test_detects_process_wide_state(src, flagged):
    assert bool(_hits(src)) is flagged


# Shared state a function writes without a recognised container initializer.
_RUNTIME_WRITES = {
    "thread_local": "import threading\nX = threading.local()\ndef put(v):\n    X.value = v\n",
    "thread_local_unused": "import threading\nX = threading.local()\n",
    "singleton": "class State:\n    pass\nX = State()\ndef put(v):\n    X.value = v\n",
    "global_rebind": "X = None\ndef put(v):\n    global X\n    X = v\n",
    "global_counter": "N = 0\ndef bump():\n    global N\n    N += 1\n",
    "global_created_at_runtime": "def put(v):\n    global X\n    X = v\n",
    "module_setattr": "import sys\nX = None\ndef put(v):\n    setattr(sys.modules[__name__], 'X', v)\n",
    "module_globals_item": "X = None\ndef put(v):\n    globals()['X'] = v\n",
    "populated_defaultdict": "from collections import defaultdict\nX = defaultdict(list, {'seed': []})\n"
                             "def put(v):\n    X['seed'].append(v)\n",
    "nested_constant": "X = {'seed': []}\ndef put(v):\n    X['seed'].append(v)\n",
    "nested_attribute": "X = {'seed': []}\ndef put(v):\n    X['seed'][0].value = v\n",
    "class_other_self": "class A:\n    values = {}\n    def put(this, v):\n        this.values['v'] = v\n",
    "class_type_self": "class A:\n    current = None\n    def put(self, v):\n        type(self).current = v\n",
    "class_dunder_class": "class A:\n    values = []\n    def put(self, v):\n        self.__class__.values.append(v)\n",
    "class_conditional_copy": "class A:\n    values = {}\n    def put(self, v, copy=False):\n        if copy:\n"
                              "            self.values = {}\n        self.values['v'] = v\n",
    "class_conditional_init_copy": "class A:\n    values = {}\n    def __init__(self, copy=False):\n"
                                   "        if copy:\n            self.values = {}\n"
                                   "    def put(self, v):\n        self.values['v'] = v\n",
    "class_copy_outside_init": "class A:\n    values = {}\n    def reset(self):\n        self.values = {}\n"
                               "    def put(self, v):\n        self.values['v'] = v\n",
    "nested_class": "class A:\n    class B:\n        values = []\n        @classmethod\n"
                    "        def put(cls, v):\n            cls.values.append(v)\n",
    "fake_declaration": "def tenant_neutral(v, **kw):\n    return v\nX = tenant_neutral({}, reason='not actually neutral')\n",
    "loop": "for i in range(1):\n    X = {}\n",
    "while_body": "while True:\n    X = []\n    break\n",
    "match_body": "match 1:\n    case 1:\n        X = set()\n",
    # an instance copy counts only when it is a fresh value assigned before any use, on every path
    "class_copy_is_an_alias": "class C:\n    values = {}\n    def __init__(self):\n        self.values = C.values\n"
                              "    def put(self, v):\n        self.values['v'] = v\n",
    "class_copy_is_self_alias": "class C:\n    values = {}\n    def __init__(self):\n        self.values = self.values\n"
                                "    def put(self, v):\n        self.values['v'] = v\n",
    "class_write_before_copy": "class C:\n    values = {}\n    def __init__(self):\n        self.values['v'] = 1\n"
                               "        self.values = {}\n",
    "class_read_before_copy": "class C:\n    values = []\n    def __init__(self):\n        self.values.append(1)\n"
                              "        self.values = []\n",
    "class_early_return_before_copy": "class C:\n    values = {}\n    def __init__(self, skip=False):\n"
                                      "        if skip:\n            return\n        self.values = {}\n"
                                      "    def put(self, v):\n        self.values['v'] = v\n",
    "class_conditional_init": "class C:\n    values = {}\n    if FLAG:\n        def __init__(self):\n"
                              "            self.values = {}\n    def put(self, v):\n        self.values['v'] = v\n",
    # a walrus through ``global`` rebinds the module name
    "global_walrus": "X = None\ndef put(v):\n    global X\n    return (X := v)\n",
    # a method defined under a class-body ``if``/``try`` is a method
    "class_method_under_if": "class C:\n    values = {}\n    if True:\n        def put(self, v):\n"
                             "            self.values['v'] = v\n",
    "class_method_under_try": "class C:\n    values = []\n    try:\n        def put(self, v):\n"
                              "            self.values.append(v)\n    except Exception:\n        pass\n",
    # the defining module cannot see a write through a name imported from it: it is reported here
    "imported_object_item": "from state import X\ndef put(v):\n    X['v'] = v\n",
    "imported_object_method": "from state import X\ndef put(v):\n    X.append(v)\n",
    "imported_object_attribute": "from state import X as Y\ndef put(v):\n    Y.value = v\n",
    # an own copy stops counting once anything can put the shared value back or remove the copy
    "class_copy_then_realiased": "class C:\n    values = {}\n    def __init__(self):\n        self.values = {}\n"
                                 "        self.values = C.values\n    def put(self, v):\n        self.values['v'] = v\n",
    "class_copy_then_deleted": "class C:\n    values = {}\n    def __init__(self):\n        self.values = {}\n"
                               "        del self.values\n    def put(self, v):\n        self.values['v'] = v\n",
    "class_copy_realiased_elsewhere": "class C:\n    values = {}\n    def __init__(self):\n        self.values = {}\n"
                                      "    def share(self):\n        self.values = C.values\n"
                                      "    def put(self, v):\n        self.values['v'] = v\n",
    "class_copy_deleted_elsewhere": "class C:\n    values = {}\n    def __init__(self):\n        self.values = {}\n"
                                    "    def drop(self):\n        delattr(self, 'values')\n"
                                    "    def put(self, v):\n        self.values['v'] = v\n",
    "class_copy_setattr_alias": "class C:\n    values = {}\n    def __init__(self):\n        self.values = {}\n"
                                "    def share(self, name):\n        setattr(self, name, C.values)\n"
                                "    def put(self, v):\n        self.values['v'] = v\n",
    "class_copy_dict_removed": "class C:\n    values = {}\n    def __init__(self):\n        self.values = {}\n"
                               "    def drop(self):\n        self.__dict__.pop('values')\n"
                               "    def put(self, v):\n        self.values['v'] = v\n",
    "class_copy_unpacked_alias": "class C:\n    values = {}\n    def __init__(self):\n        self.values = {}\n"
                                 "    def share(self):\n        self.values, n = C.values, 1\n"
                                 "    def put(self, v):\n        self.values['v'] = v\n",
    # an imported object is one through whichever binding reaches it
    "relative_imported_object": "from . import X\ndef put(v):\n    X['v'] = v\n",
    "relative_imported_member": "from .state import X\ndef put(v):\n    X['v'] = v\n",
    "imported_object_unrelated_module_alias": "from state import X\ndef unrelated():\n    import state.X as unused\n"
                                              "def put(v):\n    X['v'] = v\n",
    "imported_object_local_alias": "def put(v):\n    from state import X\n    X['v'] = v\n",
    "imported_object_closure": "def outer():\n    from state import X as Y\n    def put(v):\n"
                               "        Y.append(v)\n    return put\n",
}


@pytest.mark.parametrize("name", sorted(_RUNTIME_WRITES))
def test_detects_state_whatever_its_initializer(name):
    assert _hits(_RUNTIME_WRITES[name]), name


@pytest.mark.parametrize("src, line", [
    ("from . import X\ndef put(v):\n    X['v'] = v\n", 1),
    ("from state import X\ndef unrelated():\n    import state.X as unused\ndef put(v):\n    X['v'] = v\n", 1),
    ("import os\ndef put(v):\n    from .state import X\n    X['v'] = v\n", 3),
])
def test_imported_object_write_is_reported_on_its_import(src, line):
    assert _hits(src) == [line]


@pytest.mark.parametrize("src", [
    # a module bound by ``import m.n as a`` is a module: its attribute is judged in that module
    "def put(v):\n    import state.X as X\n    X.value = v\n",
    "import state.X as X\ndef put(v):\n    X['v'] = v\n",
    # an own copy kept fresh by every later assignment stays the instance's
    "class C:\n    values = {}\n    def __init__(self):\n        self.values = {}\n"
    "    def reset(self):\n        self.values = {}\n    def put(self, v):\n        self.values['v'] = v\n",
    "class C:\n    values = {}\n    def __init__(self):\n        self.values = {}\n"
    "    def grow(self, more):\n        self.values |= more\n",
])
def test_module_aliases_and_kept_copies_pass(src):
    assert _hits(src) == []


# A mutation is the module's only through its lexical binding.
_TABLE = "TABLE = {'a': 1}\n"


@pytest.mark.parametrize("body, flagged", [
    ("def f(TABLE):\n    TABLE.clear()\n", False),                              # parameter
    ("def f():\n    TABLE = []\n    TABLE.append(1)\n", False),                # local, no global
    ("def f():\n    def g(TABLE):\n        TABLE['x'] = 1\n    return g\n", False),  # nested param
    ("def f():\n    def g():\n        TABLE = {}\n        TABLE['x'] = 1\n    return g\n", False),
    ("def f(rows):\n    return [TABLE.update(r) for TABLE in rows]\n", False),  # comprehension var
    ("class C:\n    TABLE = {}\n    TABLE['x'] = 1\n", False),                 # class-body name
    ("def f():\n    for TABLE in ({},):\n        TABLE['x'] = 1\n", False),    # loop target
    ("def f():\n    with open('p') as TABLE:\n        TABLE.write('x')\n", False),
    ("def f():\n    global TABLE\n    TABLE['x'] = 1\n", True),                # global: the module's
    ("def f():\n    TABLE['x'] = 1\n", True),                                    # free name: the module's
    ("def f():\n    def g():\n        TABLE['x'] = 1\n    return g\n", True),
    ("class C:\n    def m(self):\n        TABLE['x'] = 1\n", True),             # class scope skipped
    ("def f(TABLE=TABLE):\n    TABLE['x'] = 1\n", False),                       # default is a param
])
def test_resolves_each_mutation_to_its_binding(body, flagged):
    assert bool(_hits(_TABLE + body)) is flagged


@pytest.mark.parametrize("src", [
    # a local function of the same name
    "def tenant_neutral(v, **kw):\n    return v\n_X = tenant_neutral({}, reason='r')\n",
    "class ScopedCache(dict):\n    pass\n_X = ScopedCache()\n",
    # the same name imported from another module
    "from elsewhere import tenant_neutral\n_X = tenant_neutral({}, reason='r')\n",
    "from elsewhere import ScopedCache\n_X = ScopedCache('c')\n",
    "import elsewhere\n_X = elsewhere.tenant_neutral({}, reason='r')\n",
    # the real import, shadowed where the declaration is made
    "from hermes_scope_cache import tenant_neutral\ntenant_neutral = lambda v, **kw: v\n"
    "_X = tenant_neutral({}, reason='r')\n",
    "from hermes_scope_cache import tenant_neutral\nclass C:\n    tenant_neutral = staticmethod(lambda v, **kw: v)\n"
    "    values = tenant_neutral({}, reason='r')\n    def add(self, k):\n        self.values[k] = 1\n",
    "import functools\ndef tenant_neutral(**kw):\n    return lambda f: f\n"
    "@tenant_neutral(reason='r')\n@functools.cache\ndef f(x):\n    return x\n",
])
def test_declarations_are_recognised_by_api_identity(src):
    assert _hits(src)


@pytest.mark.parametrize("src", [
    "import hermes_scope_cache\n_X = hermes_scope_cache.tenant_neutral({}, reason='same for all')\n",
    "from hermes_scope_cache import ScopedCache as Per\n_X = Per('c')\n",
    "import hermes_scope_cache as hsc\n_X = hsc.ScopedCache('c')\ndef f():\n    _X.slot()['k'] = 1\n",
    # a module-level table filled while the module imports is the same for every tenant
    "_T = {'a': 1}\nfor k in ('b',):\n    _T[k] = 2\n",
    # an imported module's state is judged in that module
    "import os\ndef f(v):\n    os.environ['X'] = v\n",
    # an instance copy that is a fresh value, made first: per instance
    "class C:\n    values = {}\n    def __init__(self):\n        super().__init__()\n        self.values = dict(C.values)\n"
    "    def put(self, v):\n        self.values['v'] = v\n",
    "class C:\n    values = []\n    def __init__(self):\n        self.values = [x for x in C.values]\n"
    "    def put(self, v):\n        self.values.append(v)\n",
    # reading a name imported from another module, or a walrus local to the function
    "from state import X\ndef get(k):\n    return X.get(k)\n",
    "def put(v):\n    return (X := v)\n",
])
def test_declared_and_import_time_state_passes(src):
    assert not _hits(src)


def _graph_repo(tmp_path, files):
    root = tmp_path / "g"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=60)
    return root


def test_graph_follows_import_time_edges_only(tmp_path):
    root = _graph_repo(tmp_path, {
        "entry.py": "import pkg.sub\nfrom pkg import leaf\nfrom typing import TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n    import typed_only\ntry:\n    import guarded\nexcept ImportError:\n"
                    "    pass\ndef later():\n    import lazy_only\n",
        "pkg/__init__.py": "", "pkg/sub.py": "from . import rel\n", "pkg/rel.py": "",
        "pkg/leaf.py": "", "typed_only.py": "", "guarded.py": "", "lazy_only.py": "",
        "island.py": "",
    })
    got = reachable(root, None, ("entry",))
    assert got == {"entry.py", "pkg/__init__.py", "pkg/sub.py", "pkg/rel.py", "pkg/leaf.py", "guarded.py"}


# Each import-time edge, as (files, the module only that edge reaches). Every case also holds a
# function-body import and a TYPE_CHECKING import that must stay out of the graph.
_NOT_FOLLOWED = "def later():\n    import in_function\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import typed\n"
_EDGES = {
    "root_package_init": ({"pkg/__init__.py": "import shared\n", "pkg/entry.py": _NOT_FOLLOWED}, "shared.py"),
    "class_body": ({"pkg/entry.py": "class C:\n    import shared\n" + _NOT_FOLLOWED}, "shared.py"),
    "nested_class": ({"pkg/entry.py": "class C:\n    class D:\n        import shared\n" + _NOT_FOLLOWED},
                     "shared.py"),
    "for_loop": ({"pkg/entry.py": "for _ in range(1):\n    import shared\n" + _NOT_FOLLOWED}, "shared.py"),
    "while_loop": ({"pkg/entry.py": "while True:\n    import shared\n    break\n" + _NOT_FOLLOWED}, "shared.py"),
    "loop_else": ({"pkg/entry.py": "for _ in ():\n    pass\nelse:\n    import shared\n" + _NOT_FOLLOWED},
                  "shared.py"),
    "match_case": ({"pkg/entry.py": "match 1:\n    case 1:\n        import shared\n" + _NOT_FOLLOWED}, "shared.py"),
    "with_block": ({"pkg/entry.py": "import contextlib\nwith contextlib.suppress(ImportError):\n"
                                    "    import shared\n" + _NOT_FOLLOWED}, "shared.py"),
    "try_else": ({"pkg/entry.py": "try:\n    pass\nexcept ImportError:\n    pass\nelse:\n    import shared\n"
                                  + _NOT_FOLLOWED}, "shared.py"),
    "if_else": ({"pkg/entry.py": "import sys\nif sys.platform == 'x':\n    pass\nelse:\n    import shared\n"
                                 + _NOT_FOLLOWED}, "shared.py"),
    "type_checking_else": ({"pkg/entry.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import typed\n"
                                            "else:\n    import shared\n"}, "shared.py"),
    "lazy_forwarder": ({"pkg/entry.py": "from agent.lazy_forward import forward as _forward\nclass A:\n"
                                        "    m = _forward('shared', 'm')\n" + _NOT_FOLLOWED,
                        "agent/__init__.py": "", "agent/lazy_forward.py": ""}, "shared.py"),
    "builtin_tool_discovery": ({"pkg/entry.py": "import model_tools\n" + _NOT_FOLLOWED,
                                "model_tools.py": "from tools.registry import discover_builtin_tools\n"
                                                  "discover_builtin_tools()\n",
                                "tools/__init__.py": "", "tools/registry.py": "registry = None\n",
                                "tools/shared.py": "from tools.registry import registry\nregistry.register('t')\n",
                                "tools/library.py": "def f():\n    registry.register('t')\n",
                                "tools/pkgtool/__init__.py": "", "tools/pkgtool/tool.py":
                                    "from tools.registry import registry\nfor n in 'ab':\n    registry.register(n)\n",
                                "tools/orphan/tool.py": "from tools.registry import registry\nregistry.register('t')\n"},
                               "tools/shared.py"),
}


def _edge_files(name):
    files, target = _EDGES[name]
    base = {"pkg/__init__.py": "", "pkg/entry.py": "", "shared.py": "", "in_function.py": "", "typed.py": ""}
    return {**base, **files}, target


@pytest.mark.parametrize("name", sorted(_EDGES))
def test_graph_follows_every_import_time_edge(tmp_path, name):
    files, target = _edge_files(name)
    got = reachable(_graph_repo(tmp_path, files), None, ("pkg.entry",))
    assert target in got
    assert not {"in_function.py", "typed.py"} & got


def test_graph_models_builtin_tool_discovery(tmp_path):
    files, _ = _edge_files("builtin_tool_discovery")
    got = reachable(_graph_repo(tmp_path, files), None, ("pkg.entry",))
    assert {"tools/shared.py", "tools/pkgtool/tool.py"} <= got
    assert not {"tools/library.py", "tools/orphan/tool.py"} & got  # no module-level register; no package
    # and only through model_tools: a graph that never imports it discovers nothing
    assert "tools/shared.py" not in reachable(tmp_path / "g", None, ("tools.registry",))


def test_dynamic_discovery_includes_package_init(tmp_path):
    """Importing ``tools.pkg.tool`` runs ``tools/pkg/__init__.py`` first, and what it imports."""
    root = _graph_repo(tmp_path, {
        "model_tools.py": "", "tools/__init__.py": "", "tools/pkg/__init__.py": "import hidden\nX = {}\n",
        "tools/pkg/tool.py": "registry.register()\n", "hidden.py": "X = {}\n"})
    got = reachable(root, None, ("model_tools",))
    assert {"tools/__init__.py", "tools/pkg/__init__.py", "tools/pkg/tool.py", "hidden.py"} <= got, sorted(got)


def test_tool_discovery_mirror_agrees_with_the_registry():
    """The graph mirrors the registry's discovery predicate (the registry module is not importable
    from the lint without the tool stack); both must select the same modules on the real tree."""
    from scripts.code_health import gitio
    from scripts.code_health.hosted_graph_tools import discovered_tool_modules
    from tools import registry

    tools_dir = (REPO / "tools").resolve()
    skipped = {"__init__.py", "registry.py", "mcp_tool.py"}
    expected = sorted(
        p.relative_to(REPO.resolve()).as_posix() for p in registry._tool_module_candidates(tools_dir)
        if p.name not in skipped and (p.parent == tools_dir or (p.parent / "__init__.py").exists())
        and registry._module_registers_tools(p))
    files = {p for p in gitio.tracked_files(REPO, None) if p.endswith(".py")}
    texts = {p: (REPO / p).read_text(encoding="utf-8-sig") for p in files if p.startswith("tools/")}
    assert expected and discovered_tool_modules(files, texts) == expected


@pytest.mark.parametrize("name", sorted(_EDGES))
def test_state_behind_each_edge_blocks(tmp_path, monkeypatch, capsys, name):
    monkeypatch.setattr(config, "HOSTED_ENTRY", ("pkg.entry",))
    files, target = _edge_files(name)
    repo, _ = _repo(tmp_path)
    base = _commit(repo, files)
    code, out = _verdict(repo, base, {target: files[target] + "\n_PLANTED = {}\n"}, capsys)
    assert code != 0 and "HX015" in out, out


def test_every_hosted_root_names_a_module():
    """A renamed root would silently shrink the judged graph."""
    missing = [r for r in config.HOSTED_ENTRY if importlib.util.find_spec(r) is None]
    assert not missing
    assert {f"{r.replace('.', '/')}.py" for r in ("hermes_scope", "tools.registry")} <= reachable(REPO, None)


_ENTRY = "import pkg.state\n"
_OLD_STATE = "_OLD = {}\n"


@pytest.fixture
def hosted_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "HOSTED_ENTRY", ("pkg.entry",))
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"pkg/__init__.py": "", "pkg/entry.py": _ENTRY, "pkg/state.py": _OLD_STATE,
                          "pkg/island.py": _OLD_STATE})
    return repo, base


@pytest.mark.parametrize("head, blocks", [
    ({"pkg/state.py": _OLD_STATE + "_PLANTED = {}\n"}, True),       # reachable: blocks
    ({"pkg/island.py": _OLD_STATE + "_PLANTED = {}\n"}, False),     # nothing hosted imports it
    ({"pkg/state.py": _OLD_STATE + "\n\ndef f():\n    return 1\n"}, False),  # old state baselined
    # a new import pulls an old module onto the hosted path: its state was never hosted, so it
    # is new there, whether or not the module is edited in the same change
    ({"pkg/entry.py": _ENTRY + "import pkg.island\n"}, True),
    ({"pkg/entry.py": _ENTRY + "import pkg.island\n",
      "pkg/island.py": _OLD_STATE + "\n\ndef f():\n    return 1\n"}, True),
    # dropping an import and re-adding nothing: no module is newly reached
    ({"pkg/entry.py": ""}, False),
    ({"pkg/state.py": _OLD_STATE + "from hermes_scope_cache import ScopedCache\n"
                                   "_NEW = ScopedCache('new')\n"}, False),
])
def test_ratchet(hosted_repo, capsys, head, blocks):
    repo, base = hosted_repo
    code, out = _verdict(repo, base, head, capsys)
    assert (code != 0) is blocks, out
    if blocks:
        assert "HX015" in out


def test_new_reachability_cannot_import_unchecked_state(tmp_path, monkeypatch, capsys):
    """State added off the hosted path in one change cannot be brought onto it by another change
    that only adds an import: the newly reached module is judged though it is unchanged."""
    monkeypatch.setattr(config, "HOSTED_ENTRY", ("pkg.entry",))
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"pkg/__init__.py": "", "pkg/entry.py": "", "pkg/state.py": "X = {}\n"})
    code, out = _verdict(repo, base, {"pkg/entry.py": "import pkg.state\n"}, capsys)
    assert code != 0 and "pkg/state.py" in out and "HX015" in out, out
    assert "1 newly on the hosted path" in out


def test_newly_reached_module_is_judged_only_for_hosted_state(tmp_path, monkeypatch, capsys):
    """An unchanged module pulled onto the hosted path brings its other debt (a swallowed
    exception, an over-long function) with it as before: only its shared state is new."""
    monkeypatch.setattr(config, "HOSTED_ENTRY", ("pkg.entry",))
    repo, _ = _repo(tmp_path)
    legacy = "def f():\n    try:\n        return 1\n    except Exception:\n        pass\n"
    base = _commit(repo, {"pkg/__init__.py": "", "pkg/entry.py": "", "pkg/legacy.py": legacy})
    code, out = _verdict(repo, base, {"pkg/entry.py": "import pkg.legacy\n"}, capsys)
    assert code == 0, out


def test_base_credit_covers_only_state_already_hosted(tmp_path, monkeypatch, capsys):
    """A module edited while it is pulled onto the hosted path gets no credit for the state it
    had off the path; a module already on the path keeps its baseline."""
    monkeypatch.setattr(config, "HOSTED_ENTRY", ("pkg.entry",))
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"pkg/__init__.py": "", "pkg/entry.py": "import pkg.on\n",
                          "pkg/on.py": "X = {}\n", "pkg/off.py": "X = {}\n"})
    edit = "X = {}\n\n\ndef g():\n    return 2\n"
    assert _verdict(repo, base, {"pkg/on.py": edit}, capsys)[0] == 0
    code, out = _verdict(repo, base, {"pkg/entry.py": "import pkg.on\nimport pkg.off\n", "pkg/off.py": edit}, capsys)
    assert code != 0 and "pkg/off.py" in out, out


def test_new_root_has_no_base_credit(tmp_path, monkeypatch, capsys):
    """The base graph walks the BASE revision's roots: adding an existing module with state as a
    hosted root brings that state onto the hosted path, so none of it is baseline."""
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"entry.py": "", "state.py": "X = {}\n",
                          "scripts/code_health/config.py": 'HOSTED_ENTRY = ("entry",)\n'})
    head = _commit(repo, {"scripts/code_health/config.py": 'HOSTED_ENTRY = ("entry", "state")\n',
                          "entry.py": "VALUE = 1\n"})
    monkeypatch.setattr(config, "HOSTED_ENTRY", ("entry", "state"))
    result = run(repo, base, head)
    output = capsys.readouterr().out
    assert result != 0 and "state.py" in output and "HX015" in output, output


def test_base_without_declared_roots_uses_the_head_roots(tmp_path, monkeypatch, capsys):
    """A base from before the hosted graph existed declares no roots: the check's introduction
    baselines the state already on the hosted path instead of failing on all of it."""
    repo, _ = _repo(tmp_path)
    base = _commit(repo, {"entry.py": "import state\n", "state.py": "X = {}\n"})
    head = _commit(repo, {"entry.py": "import state\nVALUE = 1\n"})
    monkeypatch.setattr(config, "HOSTED_ENTRY", ("entry",))
    assert run(repo, base, head) == 0, capsys.readouterr().out


@pytest.mark.parametrize(("text", "expected"), [
    ('HOSTED_ENTRY = ("a", "b")\n', ("a", "b")),
    ('HOSTED_ENTRY: tuple[str, ...] = ["a"]\nHOSTED_ENTRY = ("c",)\n', ("c",)),
    ("X = 1\n", None),
    ("HOSTED_ENTRY = tuple(ROOTS)\n", None),
    ("HOSTED_ENTRY = (\n", None),
])
def test_parse_hosted_entry(text, expected):
    from scripts.code_health.cli import parse_hosted_entry
    assert parse_hosted_entry(text) == expected
