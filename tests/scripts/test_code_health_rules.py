"""Rule and structure precision of the code-health ratchet (scripts/code_health): each case is a
reviewer's repro paired with the control that must keep its verdict."""

from __future__ import annotations

import ast

import pytest

from scripts.code_health.config import RULES_BY_ID
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from scripts.code_health.py_structure import body_hash, nesting_depth
from tests.scripts.test_code_health import _commit, _repo, _verdict


def _hits(rule: str, src: str) -> list[int]:
    """Lines ``rule`` reports for ``src``, through the same canonical tree the measurer uses."""
    return sorted(set(CHECKERS[rule](canonical_tree(ast.parse(src)), Ctx())))


def _judge(tmp_path, capsys, base_files: dict[str, str], head_files: dict[str, str | None]):
    repo, base = _repo(tmp_path)
    if base_files:
        base = _commit(repo, base_files)
    return _verdict(repo, base, head_files, capsys)


# --- F12: an import alias is shadowed only in the scope that rebinds it ---

_ALIAS = "from subprocess import run as execute\n\n\n{}\n\n\ndef launch(cmd):\n{}    return execute(cmd)\n"


@pytest.mark.parametrize("elsewhere, local, flagged", [
    ("def identity(execute):\n    return execute", "", True),  # unrelated parameter
    ("def other():\n    execute = 42\n    return execute", "", True),  # unrelated local
    ("def other():\n    def execute():\n        return 1\n    return execute", "", True),  # nested def
    ("def identity(x):\n    return x", "", True),
    # genuine shadowing where the call is made still stops resolution
    ("def identity(x):\n    return x", "    execute = print\n", False),
    ("execute = print", "", False),  # the module itself rebinds it: ambiguous
])
def test_import_alias_resolves_per_scope(elsewhere, local, flagged):
    src = _ALIAS.format(elsewhere, local)
    assert bool(_hits("HX006", src)) is flagged, src


def test_import_alias_parameter_of_consuming_function_shadows():
    src = "from subprocess import run as execute\n\n\ndef launch(cmd, execute):\n    return execute(cmd)\n"
    assert _hits("HX006", src) == []


def test_import_alias_with_unrelated_parameter_blocks_through_the_verdict(tmp_path, capsys):
    head = {"pkg/c.py": _ALIAS.format("def identity(execute):\n    return execute", "")}
    code, out = _judge(tmp_path, capsys, {}, head)
    assert code == 1 and "HX006" in out, out


# --- F19: every independent eager capture is its own occurrence ---

_DICT = 'import os\n\nCACHE = {{\n{}    "old": os.getenv("PATH"),\n}}\n'
_LIST = "import os\n\nPATHS = [\n{}    os.getenv(\"PATH\"),\n]\n"
_DEFAULT = "import os\n\n\ndef f(\n{}    a=os.getenv(\"PATH\"),\n):\n    return a\n"


@pytest.mark.parametrize("shape, added, blocks", [
    (_DICT, '    "new": os.getenv("HOME"),\n', True),
    (_LIST, '    os.getenv("HOME"),\n', True),
    (_DEFAULT, '    b=os.getenv("HOME"),\n', True),
    (_DICT, '    "new": lambda: os.getenv("HOME"),\n', False),  # deferred read: the fix
    (_DICT, '    "new": "literal",\n', False),  # unchanged debt stays grandfathered
])
def test_added_capture_inside_an_existing_expression_is_new(tmp_path, capsys, shape, added, blocks):
    code, out = _judge(tmp_path, capsys, {"pkg/c.py": shape.format("")},
                       {"pkg/c.py": shape.format(added)})
    assert code == (1 if blocks else 0), out
    assert ("HX005" in out) is blocks, out


def test_added_capture_in_its_own_assignment_is_new(tmp_path, capsys):
    base = _DICT.format("")
    code, out = _judge(tmp_path, capsys, {"pkg/c.py": base},
                       {"pkg/c.py": base + 'NEW = os.getenv("HOME")\n'})
    assert code == 1 and "HX005" in out, out


def test_capture_lines_reports_each_independent_capture_once():
    two = 'import os\n\nCACHE = {\n    "a": os.getenv("A"),\n    "b": os.getenv("B"),\n}\n'
    assert _hits("HX005", two) == [4, 5]
    # a capture nested inside another is the same occurrence
    nested = 'import os\n\nHOME = os.path.expanduser(\n    os.getenv("A", "~")\n)\n'
    assert _hits("HX005", nested) == [3]


# --- HX006: process handles are tracked per lexical scope ---

_SYNC_ASYNC = (
    "import asyncio\nimport subprocess\n\n\n"
    "def sync(cmd):\n    proc = subprocess.Popen(cmd)\n    return proc.communicate(timeout=1)\n\n\n"
    "async def asynchronous(cmd):\n    {name} = await asyncio.create_subprocess_exec(*cmd)\n"
    "    return await asyncio.wait_for({name}.communicate(), timeout=1)\n"
)


@pytest.mark.parametrize("name", ["proc", "child"])
def test_process_handles_in_different_functions_do_not_collide(name):
    assert _hits("HX006", _SYNC_ASYNC.format(name=name)) == []


def test_adding_a_bounded_async_helper_keeps_the_sync_one_clean(tmp_path, capsys):
    sync_only = _SYNC_ASYNC.format(name="proc").split("\n\n\nasync def")[0] + "\n"
    code, out = _judge(tmp_path, capsys, {"pkg/p.py": sync_only},
                       {"pkg/p.py": _SYNC_ASYNC.format(name="proc")})
    assert code == 0, out


def test_process_handle_follows_reassignment():
    src = ("import asyncio\nimport subprocess\n\n\nasync def f(c):\n"
           "    proc = subprocess.Popen(c)\n    proc.wait(timeout=1)\n"
           "    proc = await asyncio.create_subprocess_exec(*c)\n    await proc.wait()\n"
           "    proc = None\n    return proc\n")
    assert _hits("HX006", src) == [9]  # only the asyncio wait, which nothing bounds
    unbounded = "import subprocess\n\n\ndef f(c):\n    proc = subprocess.Popen(c)\n    return proc.wait()\n"
    assert _hits("HX006", unbounded) == [6]
    with_block = "import subprocess\n\n\ndef f(c):\n    with subprocess.Popen(c) as proc:\n        return proc.wait()\n"
    assert _hits("HX006", with_block) == [6]
    # a parameter that merely shares the spelling is not the other function's process
    param = unbounded + "\n\ndef stop(proc):\n    return proc.wait()\n"
    assert _hits("HX006", param) == [6]


def test_process_handle_on_an_attribute_spans_the_class_methods():
    src = ("import subprocess\n\n\nclass Runner:\n    def start(self, c):\n"
           "        self._proc = subprocess.Popen(c)\n\n    def stop(self):\n"
           "        return self._proc.wait()\n")
    assert _hits("HX006", src) == [9]
    closure = ("import subprocess\n\n\ndef f(c):\n    proc = subprocess.Popen(c)\n\n"
               "    def reap():\n        return proc.wait()\n    return reap\n")
    assert _hits("HX006", closure) == [8]


# --- HX006: a wait right after kill() is bounded, however its value is used ---

_KILL = "import asyncio\nimport subprocess\n\n\ndef reap(cmd, other):\n    proc = subprocess.Popen(cmd)\n{}"


@pytest.mark.parametrize("tail, flagged", [
    ("    proc.kill()\n    proc.wait()\n", False),
    ("    proc.kill()\n    rc = proc.wait()\n    return rc\n", False),
    ("    proc.kill()\n    return proc.wait()\n", False),
    ("    proc.kill()\n    rc: int = proc.wait()\n    return rc\n", False),
    ("    proc.kill()\n    return proc.wait(timeout=5)\n", False),
    ("    return proc.wait()\n", True),  # no preceding kill
    ("    other.kill()\n    return proc.wait()\n", True),  # a different receiver was killed
    ("    proc.kill()\n    return proc.communicate()\n", True),  # reads until grandchildren exit
])
def test_wait_after_kill(tail, flagged):
    assert bool(_hits("HX006", _KILL.format(tail))) is flagged


def test_async_wait_after_kill_stays_flagged():
    src = ("import asyncio\n\n\nasync def f(c):\n    proc = await asyncio.create_subprocess_exec(*c)\n"
           "    proc.kill()\n    return await proc.wait()\n")
    assert _hits("HX006", src) == [7]


def test_assignment_refactored_into_return_after_kill_passes(tmp_path, capsys):
    base = _KILL.format("    proc.kill()\n    rc = proc.wait()\n    return rc\n")
    head = _KILL.format("    proc.kill()\n    return proc.wait()\n")
    code, out = _judge(tmp_path, capsys, {"pkg/k.py": base}, {"pkg/k.py": head})
    assert code == 0, out


# --- HX009: annotated results and projected loop variables ---

_GATHER = "import asyncio\n\n\nasync def run(awaitables, payloads):\n    results{ann} = await asyncio.gather(\n        *awaitables, return_exceptions=True\n    )\n{loop}"


@pytest.mark.parametrize("ann, loop, flagged", [
    (": list", "    for result in results:\n        if isinstance(result, Exception):\n            raise result\n", True),
    ("", "    for result in results:\n        if isinstance(result, Exception):\n            raise result\n", True),
    ("", "    for payload, result in zip(payloads, results):\n        if isinstance(result, Exception):\n            raise result\n", True),
    ("", "    for payload, result in zip(payloads, results):\n        if isinstance(payload, Exception):\n            raise payload\n", False),
    ("", "    for payload, result in zip(payloads, results, strict=True):\n        if isinstance(payload, Exception):\n            raise payload\n", False),
    ("", "    for i, r in enumerate(results):\n        if isinstance(r, Exception):\n            raise r\n", True),
    ("", "    for i, r in enumerate(results):\n        if isinstance(i, Exception):\n            raise i\n", False),
    ("", "    return [r for i, r in enumerate(results) if isinstance(i, Exception)]\n", False),
    (": list", "    return [r for i, r in enumerate(results) if isinstance(r, Exception)]\n", True),
])
def test_gather_results_projection(ann, loop, flagged):
    assert bool(_hits("HX009", _GATHER.format(ann=ann, loop=loop))) is flagged


# --- HX005: annotations are evaluated at def time unless postponed ---

_FUTURE = "from __future__ import annotations\n\n"


@pytest.mark.parametrize("src, flagged", [
    ("import os\n\n\ndef f(x: os.getenv('A')):\n    return x\n", True),
    ("import os\n\n\ndef f(x) -> os.getenv('A'):\n    return x\n", True),
    ("import os\n\n\nclass C:\n    def f(self, *a: os.getenv('A')):\n        return a\n", True),
    (_FUTURE + "import os\n\n\ndef f(x: os.getenv('A')):\n    return x\n", False),
    (_FUTURE + "import os\n\n\ndef f(x) -> os.getenv('A'):\n    return x\n", False),
    (_FUTURE + "import os\n\nx: os.getenv('PATH')\n", False),
    (_FUTURE + "import os\n\n\nclass C:\n    x: os.getenv('PATH')\n", False),
    (_FUTURE + "import os\n\nx: str = os.getenv('PATH')\n", True),  # the value still runs
    ("import os\n\nx: os.getenv('PATH')\n", True),
    ("import os\n\n\ndef f():\n    def g(x: os.getenv('A')):\n        return x\n    return g\n", False),
])
def test_annotation_evaluation_time(src, flagged):
    assert bool(_hits("HX005", src)) is flagged, src


# --- F15: only the function's own bindings stop the recursive-rename normalization ---

def _recursive(name: str, inner: str = "legacy") -> str:
    return (f"def {name}(n):\n def identity({inner}):\n  return {inner}\n" + " pass\n" * 300
            + f" return {name}(n-1) if n else identity(0)\n")


def _hash(src: str) -> str:
    return body_hash(ast.parse(src).body[0])


@pytest.mark.parametrize("inner", ["legacy", "value"])
def test_recursive_rename_ignores_nested_bindings(inner):
    assert _hash(_recursive("legacy", inner)) == _hash(_recursive("renamed", inner))


def test_recursive_rename_with_nested_parameter_keeps_its_cap(tmp_path, capsys):
    code, out = _judge(tmp_path, capsys, {"pkg/r.py": _recursive("legacy")},
                       {"pkg/r.py": _recursive("renamed")})
    assert code == 0, out


def test_recursive_rename_controls():
    # a rename that leaves the old self-call behind is different code
    stale = _recursive("renamed").replace("return renamed(", "return legacy(")
    assert _hash(_recursive("legacy")) != _hash(stale)
    # a body that rebinds the name in its own scope refers to that local, not to itself
    own = "def {0}(n):\n    {0} = n\n    return {0}\n"
    assert _hash(own.format("legacy")) != _hash(own.format("renamed"))
    # a closure's reference to the enclosing function is still a self-reference
    closure = "def {0}(n):\n    def helper():\n        return {0}(n - 1)\n    return helper()\n"
    assert _hash(closure.format("legacy")) == _hash(closure.format("renamed"))


# --- nesting: `else:` + indented `if` is nesting; `elif` is not ---

# `>` keeps the fixtures off HX011 (an if/elif ladder on one name).
def _else_if(levels: int, inert: bool = False) -> str:
    out, pad = ["def f(x):"], "    "
    for k in range(levels):
        if inert and k:
            out.append(pad + "pass")
        out += [f"{pad}if x > {k}:", f"{pad}    return {k}", f"{pad}else:"]
        pad += "    "
    out.append(pad + "return -1")
    return "\n".join(out) + "\n"


def _elif(levels: int) -> str:
    arms = "".join(f"    {'if' if k == 0 else 'elif'} x > {k}:\n        return {k}\n" for k in range(levels))
    return "def f(x):\n" + arms + "    else:\n        return -1\n"


def _depth(src: str) -> int:
    return nesting_depth(ast.parse(src).body[0].body)


def test_nesting_counts_physically_nested_else_if():
    assert _depth(_else_if(7)) == _depth(_else_if(7, inert=True)) == 7
    assert _depth(_elif(7)) == 1
    assert _depth(_else_if(6)) == 6


def test_seven_physical_else_if_levels_block(tmp_path, capsys):
    code, out = _judge(tmp_path / "nested", capsys, {}, {"pkg/n.py": _else_if(7)})
    assert code == 1 and "NESTING" in out, out
    code, out = _judge(tmp_path / "flat", capsys, {}, {"pkg/n.py": _elif(7)})
    assert code == 0, out


# --- HX001 / HX003 / HX012 precision ---

@pytest.mark.parametrize("expr, flagged", [
    ('Path.home() / ".hermes"', True),
    ('Path.home() / ".hermes/profiles"', True),
    ('Path.home() / ".hermes" / "x"', True),
    ('Path.home() / ".hermes-profile-exports"', False),
    ('Path.home() / ".hermes_backup"', False),
    ('os.path.expanduser("~/.hermes")', True),
    ('os.path.expanduser("~/.hermes/config.yaml")', True),
    ('os.path.expanduser("~/.hermes-profile-exports")', False),
])
def test_hardcoded_home_matches_the_exact_component(expr, flagged):
    src = f"import os\nfrom pathlib import Path\n\n\ndef f():\n    return {expr}\n"
    assert bool(_hits("HX001", src)) is flagged


@pytest.mark.parametrize("src, flagged", [
    ('"""Unlike `ps aux`, this reads /proc."""\n', False),
    ('def f():\n    """Never `pgrep -f hermes`: argv substrings lie."""\n    return 1\n', False),
    ('import subprocess\n\n\ndef f():\n    return subprocess.run("ps aux", shell=True, timeout=5)\n', True),
    ('CMD = "pgrep -f hermes"\n', True),
    ('def f(n):\n    return f"pgrep -f {n}"\n', True),
])
def test_argv_identity_ignores_docstrings(src, flagged):
    assert bool(_hits("HX003", src)) is flagged


_THREAD = "import contextvars\nimport threading\n\n\ndef f(fn):\n{}"


@pytest.mark.parametrize("body, flagged", [
    ("    return threading.Thread(target=contextvars.copy_context().run, args=(fn,))\n", False),
    ("    ctx = contextvars.copy_context()\n    return threading.Thread(target=ctx.run, args=(fn,))\n", False),
    ("    ctx = contextvars.copy_context()\n    return threading.Thread(None, ctx.run, args=(fn,))\n", False),
    ("    return threading.Thread(target=fn)\n", True),
    ("    ctx = make()\n    return threading.Thread(target=ctx.run, args=(fn,))\n", True),
    ("    ctx = contextvars.copy_context()\n    ctx = make()\n    return threading.Thread(target=ctx.run)\n", True),
    ("    return threading.Thread(target=ctx.run, args=(fn,))\n", True),  # ctx from elsewhere
])
def test_raw_thread_accepts_a_copied_context(body, flagged):
    assert bool(_hits("HX012", _THREAD.format(body))) is flagged


def test_raw_thread_accepts_from_import_copy_context():
    src = ("from contextvars import copy_context\nimport threading\n\n\ndef f(fn):\n"
           "    return threading.Thread(target=copy_context().run, args=(fn,))\n")
    assert _hits("HX012", src) == []


def test_raw_thread_context_from_another_function_is_not_accepted():
    src = ("import contextvars\nimport threading\n\n\ndef g():\n    ctx = contextvars.copy_context()\n"
           "    return ctx\n\n\ndef f(fn, ctx):\n    return threading.Thread(target=ctx.run)\n")
    assert _hits("HX012", src) == [11]


# --- fix texts say what the engine actually accepts ---

def test_fix_texts():
    hx010 = RULES_BY_ID["HX010"].fix
    assert "is not None" in hx010 and "is_truthy_value" in hx010
    assert hx010.index("is not None") < hx010.index("is_truthy_value")
    assert "allow" in RULES_BY_ID["HX001"].fix
    ble = RULES_BY_ID["BLE001"].fix
    assert "noqa" in ble and "health: allow BLE001" in ble and "exc_info=True" in ble


_HANDLER = "import logging\n\nlogger = logging.getLogger(__name__)\n\n\ndef boundary():\n    try:\n        pass\n{}"


@pytest.mark.parametrize("handler, blocks", [
    ("    except Exception:\n        logger.exception('boundary failed')\n", False),
    ("    except Exception:\n        logger.warning('boundary failed', exc_info=True)\n", False),
    ("    except Exception as exc:\n        raise RuntimeError('boundary failed') from exc\n", False),
    ("    except Exception as exc:\n        logger.warning('boundary failed: %s', exc)\n", True),
    ("    except Exception:  # noqa: BLE001\n        logger.warning('boundary failed')\n", True),
    ("    except Exception:  # health: allow BLE001 -- top-level boundary\n        logger.warning('x')\n", False),
])
def test_ble001_fix_text_matches_the_engine(tmp_path, capsys, handler, blocks):
    code, out = _judge(tmp_path, capsys, {}, {"pkg/h.py": _HANDLER.format(handler)})
    assert code == (1 if blocks else 0), out
