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
    ("import asyncio\nasync def settle(r):\n    for _ in range(200):\n        await asyncio.sleep(0.01)\n"
     "        if not r.busy:\n            return\n", False),  # a round-bounded poll
    ("import time\nfor _ in range(5):\n    time.sleep(0.1)\n    tick()\n", True),  # five fixed delays
    ("import time\nfor x in items:\n    time.sleep(0.1)\n    if x.done:\n        break\n", True),
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


_DEADLINE = "import time\nend = time.monotonic() + 5\n"


@pytest.mark.parametrize("src, flagged", [
    # red: a test made only of clock reads and constants waits out the clock
    (_DEADLINE + "other = time.monotonic() + 9\nwhile time.monotonic() < end and time.monotonic() < other:\n"
     "    time.sleep(0.1)\n", True),
    (_DEADLINE + "while time.monotonic() < end or False:\n    time.sleep(0.1)\n", True),
    (_DEADLINE + "while not time.monotonic() >= end:\n    time.sleep(0.1)\n", True),
    (_DEADLINE + "now = time.monotonic()\nwhile now < end:\n    time.sleep(0.1)\n"
     "    now = time.monotonic()\n", True),  # locals assigned only from the clock
    (_DEADLINE + "while True:\n    if time.monotonic() > end and True:\n        break\n"
     "    time.sleep(0.1)\n", True),
    # red: `or` ends the loop only when every operand is false, so the clock still decides
    (_DEADLINE + "while not done() or time.monotonic() < end:\n    time.sleep(0.1)\n", True),
    # green: any non-clock operand is a condition
    (_DEADLINE + "while time.monotonic() < end and not done():\n    time.sleep(0.05)\n", False),
    (_DEADLINE + "now = time.monotonic()\nwhile now < end and not done():\n    time.sleep(0.05)\n"
     "    now = time.monotonic()\n", False),
    (_DEADLINE + "while time.monotonic() < end:\n    if done() or time.monotonic() > end:\n"
     "        break\n    time.sleep(0.05)\n", False),
    (_DEADLINE + "while time.monotonic() < end:\n    if not r.busy:\n        break\n"
     "    else:\n        time.sleep(0.05)\n", False),
    (_DEADLINE + "while True:\n    left = alive()\n    if not left or time.monotonic() >= end:\n"
     "        return left\n    time.sleep(0.1)\n", False),
    # red: a timestamp kept in a dict or on an object is still the clock (fails closed)
    ("import time\nt = {'at': time.time()}\nwhile time.time() - t['at'] < 0.5:\n    time.sleep(0.005)\n", True),
    (_DEADLINE + "while time.monotonic() < self.end and time.monotonic() < end:\n    time.sleep(0.1)\n", True),
    ("import asyncio\nasync def drain():\n    end = asyncio.get_event_loop().time() + 0.15\n"
     "    while asyncio.get_event_loop().time() < end:\n        await asyncio.sleep(0.01)\n", True),
    # green: an age read from observed state is a condition
    (_DEADLINE + "while True:\n    age = time.time() - path.stat().st_mtime\n    if age > 5:\n"
     "        return\n    assert time.monotonic() < end\n    time.sleep(0.02)\n", False),
    # green: only time arithmetic makes a clock local; a value merely built from one is state
    ("import time\nnonce = time.time_ns()\npid = spawn(f'x{nonce}')\nend = time.monotonic() + 5\n"
     "while time.monotonic() < end:\n    if not alive(pid):\n        break\n    time.sleep(0.05)\n", False),
    # red: a `try` that only sleeps cannot fail, so its `break` is unconditional
    ("import time\nfor _ in range(10):\n    try:\n        time.sleep(0.5)\n        break\n"
     "    except OSError:\n        pass\n", True),
    # green: a `try` body's `return` is taken only when the probe did not raise
    (_DEADLINE + "while time.monotonic() < end:\n    try:\n        return probe()\n"
     "    except OSError:\n        time.sleep(0.1)\n", False),
    # red: the same chain of clock locals with no state in it
    ("import time\nstart = time.monotonic()\nend = start + 5\nnow = start\nwhile now < end:\n"
     "    time.sleep(0.1)\n    now = time.monotonic()\n", True),
])
def test_condition_must_read_more_than_the_clock(src, flagged):
    assert bool(_hits(src)) is flagged, src


@pytest.mark.parametrize("src, flagged", [
    # red: a `break` in a nested loop ends that loop, not the clock-only wait around it
    (_DEADLINE + "while time.monotonic() < end:\n    for x in items:\n        if x.done:\n            break\n"
     "    time.sleep(0.1)\n", True),
    (_DEADLINE + "while time.monotonic() < end:\n    while busy():\n        if ready():\n            break\n"
     "    time.sleep(0.1)\n", True),
    ("import time\nfor _ in range(5):\n    for x in items:\n        if x.done:\n            break\n"
     "    time.sleep(0.1)\n", True),
    # red: a nested def's `return` is not the loop's exit
    (_DEADLINE + "while time.monotonic() < end:\n    def check():\n        if done():\n            return True\n"
     "    time.sleep(0.1)\n", True),
    # red: a loop's `else:` runs once after the loop; it is no poll interval
    ("import time\nfor _ in range(3):\n    if ready():\n        break\nelse:\n    time.sleep(1)\n", True),
    (_DEADLINE + "while not done() and time.monotonic() < end:\n    pass\nelse:\n    time.sleep(1)\n", True),
    # ... nor are a `while` test or a `for` iterable
    (_DEADLINE + "while not done() and time.monotonic() < end and time.sleep(1) is None:\n    pass\n", True),
    # red: an outer round loop whose only exit is the inner wait's `break` (tests/tools/test_approval.py)
    ("import time\nfor i in range(n):\n    if i == 0:\n        for _ in range(400):\n            if ready():\n"
     "                break\n            time.sleep(0.005)\n    else:\n        time.sleep(0.05)\n", True),
    # green: the poll's own exits, wherever they sit in its body
    ("import time\nfor _ in range(50):\n    try:\n        if ready():\n            break\n    except OSError:\n"
     "        pass\n    time.sleep(0.05)\n", False),
    ("import time\nfor _ in range(50):\n    with lock:\n        if ready():\n            return_value = 1\n"
     "            break\n    time.sleep(0.05)\n", False),
    ("import time\ndef w():\n    for _ in range(50):\n        for x in items:\n            if x.done:\n"
     "                return x\n        time.sleep(0.05)\n", False),  # a `return` ends every loop
    ("import time\nfor _ in range(50):\n    for x in items:\n        pass\n    else:\n        if ready():\n"
     "            break\n    time.sleep(0.05)\n", False),  # an inner `else:` break ends the outer loop
    # green: a poll interval inside the loop body, beside a nested loop
    (_DEADLINE + "while not done() and time.monotonic() < end:\n    for x in items:\n        x.poke()\n"
     "    time.sleep(0.05)\n", False),
])
def test_exit_belongs_to_the_loop_and_the_interval_to_its_body(src, flagged):
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
