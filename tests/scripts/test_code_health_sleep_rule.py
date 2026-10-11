"""HX013: a new wall-clock sleep in a test blocks; existing ones are grandfathered (moved code keeps
its debt) and the shared ``tests.waiting`` helper is the compliant way to wait."""

from __future__ import annotations

import ast
import asyncio
import threading

import pytest

from scripts.code_health.config import RULES_BY_ID, rule_applies
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from tests.scripts.test_code_health import _commit, _repo, _verdict
from tests.waiting import async_wait_until, wait_until


def _hits(src: str) -> list[int]:
    return sorted(set(CHECKERS["HX013"](canonical_tree(ast.parse(src)), Ctx())))


@pytest.mark.parametrize("src, flagged", [
    ("import time\ntime.sleep(0.5)\n", True),
    ("import time as t\nt.sleep(0.01)\n", True),
    ("from time import sleep\nsleep(1)\n", True),
    ("from time import sleep as nap\nnap(secs=2)\n", True),
    ("import asyncio\nasync def f():\n    await asyncio.sleep(0.05)\n", True),
    ("from asyncio import sleep\nasync def f():\n    await sleep(3600)\n", True),
    ("import asyncio\ntask = asyncio.create_task(asyncio.sleep(60))\n", True),
    ("import time\ndef f(interval):\n    time.sleep(interval)\n", True),  # a sleep whatever the value
    # a loop that only waits out the clock, or polls with no deadline, is still a fixed delay
    ("import time\nend = time.monotonic() + 2\nwhile time.monotonic() < end:\n    time.sleep(0.1)\n",
     True),
    ("import time\nwhile not done():\n    time.sleep(0.01)\n", True),
    ("import time\nend = time.monotonic() + 2\nwhile True:\n    time.sleep(1)\n"
     "    if time.monotonic() > end:\n        break\n", True),
    # ... but the interval of a poll under a deadline is the compliant shape
    ("import time\nend = time.monotonic() + 5\nwhile not done() and time.monotonic() < end:\n"
     "    time.sleep(0.05)\n", False),
    ("import time\ndef w(p):\n    end = time.monotonic() + 5\n    while time.monotonic() < end:\n"
     "        if p():\n            return True\n        time.sleep(0.5)\n", False),
    ("import asyncio\nasync def w(p):\n    loop = asyncio.get_running_loop()\n    end = loop.time() + 5\n"
     "    while not p():\n        assert loop.time() < end\n        await asyncio.sleep(0.01)\n", False),
    # a def inside the poll loop is not its interval
    ("import time\nend = time.monotonic() + 5\nwhile not done() and time.monotonic() < end:\n"
     "    def cb():\n        time.sleep(1)\n", True),
    ("import threading\nthreading.Event().wait(0.2)\n", True),  # an Event nobody can set
    # yields and non-sleeps
    ("import time\ntime.sleep(0)\n", False),
    ("import asyncio\nasync def f():\n    await asyncio.sleep(0.0)\n", False),
    ("import threading\nev = threading.Event()\nev.wait(1)\n", False),  # the fix shape
    ("import threading\nthreading.Event().wait()\n", False),
    ("def sleep(x):\n    return x\nsleep(5)\n", False),  # a local helper, not the API
])
def test_wall_clock_sleep_shapes(src, flagged):
    assert bool(_hits(src)) is flagged, src


@pytest.mark.parametrize("path, applies", [
    ("tests/gateway/test_x.py", True),
    ("tests/e2e/core/test_y.py", True),
    ("tests/conftest.py", True),
    ("plugins/foo/tests/test_z.py", True),
    ("gateway/run.py", False),  # production sleeps are not this rule's business
    ("scripts/release.py", False),
])
def test_scope_is_tests_only(path, applies):
    assert rule_applies(RULES_BY_ID["HX013"], path) is applies


_OLD = ("import time\n\n\ndef test_old():\n    time.sleep(0.2)\n    assert True\n\n\n"
        "def test_other():\n    time.sleep(0.5)\n    assert 1\n")
_NEW = "import time\n\n\ndef test_new():\n    time.sleep(0.5)\n    assert 2\n"


def _judge(tmp_path, capsys, head: dict[str, str | None]):
    repo, base = _repo(tmp_path)
    base = _commit(repo, {"tests/test_a.py": _OLD})
    return _verdict(repo, base, head, capsys)


@pytest.mark.parametrize("head, blocks", [
    # red: a new sleep, in an old file or a new one
    ({"tests/test_a.py": _OLD + "\n\n" + _NEW.split("\n\n\n", 1)[1]}, True),
    ({"tests/test_b.py": _NEW}, True),
    # red: deleting one old sleep does not pay for a new one elsewhere
    ({"tests/test_a.py": _OLD.replace("    time.sleep(0.5)\n", ""), "tests/test_b.py": _NEW}, True),
    # red: copying an old test keeps the original's debt where it is; the copy is new
    ({"tests/test_b.py": _OLD.replace("test_old", "test_copy").split("\n\n\ndef test_other")[0]
      + "\n"}, True),
    # green: the old sleeps untouched while the file changes around them
    ({"tests/test_a.py": _OLD + "\n\ndef test_more():\n    assert 3\n"}, False),
    # green: an old test moved to another file keeps its debt
    ({"tests/test_a.py": "import time\n\n\n" + _OLD.split("\n\n\n", 2)[2],
      "tests/test_b.py": _OLD.split("\n\n\ndef test_other")[0] + "\n"}, False),
    # green: a renamed file keeps its debt
    ({"tests/test_a.py": None, "tests/test_renamed.py": _OLD}, False),
    # green: a waiver with a reason, a yield, and sleeps outside tests/
    ({"tests/test_b.py": _NEW.replace("(0.5)", "(0.5)  # health: allow HX013 -- the delay is the"
                                      " behaviour under test")}, False),
    ({"tests/test_b.py": _NEW.replace("(0.5)", "(0)")}, False),
    ({"pkg/backoff.py": _NEW.replace("test_new", "backoff")}, False),
])
def test_verdict(tmp_path, capsys, head, blocks):
    code, out = _judge(tmp_path, capsys, head)
    assert code == (1 if blocks else 0), out
    if blocks:
        assert "HX013" in out and "tests.waiting.wait_until" in out, out


def test_bare_waiver_does_not_count(tmp_path, capsys):
    code, out = _judge(tmp_path, capsys,
                       {"tests/test_b.py": _NEW.replace("(0.5)", "(0.5)  # health: allow HX013")})
    assert code == 1 and "needs a reason" in out, out


# --- the helper the fix text names ---

def test_wait_until_returns_the_first_truthy_value():
    done = threading.Event()
    threading.Timer(0.05, done.set).start()
    assert wait_until(lambda: done.is_set() and "ok", timeout=10) == "ok"


def test_wait_until_names_what_never_happened():
    with pytest.raises(AssertionError, match="timed out after 0.05s waiting for the flag"):
        wait_until(lambda: False, timeout=0.05, what="the flag")


def test_async_wait_until():
    async def scenario():
        box: list[int] = []
        asyncio.get_running_loop().call_later(0.05, box.append, 1)
        assert await async_wait_until(lambda: box, timeout=10) == [1]
        with pytest.raises(AssertionError, match="waiting for nothing"):
            await async_wait_until(lambda: None, timeout=0.05, what="nothing")

    asyncio.run(scenario())
