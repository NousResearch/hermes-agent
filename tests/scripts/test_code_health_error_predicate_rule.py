"""HX014: a new private API-error predicate on the aux/compressor/turn paths blocks; old ones are grandfathered."""

import ast

import pytest

from scripts.code_health.config import RULES_BY_ID, rule_applies
from scripts.code_health.py_rules import CHECKERS, Ctx, canonical_tree
from tests.scripts.test_code_health import _commit, _repo, _verdict

_NEW, _AUX = "def _is_quota(exc):\n    return 'quota' in str(exc).lower()\n", "agent/auxiliary_client.py"


@pytest.mark.parametrize("src, flagged", [
    ("def f(exc):\n    return exc.response.status_code == 400\n", True),
    ("def f(err):\n    return getattr(err, 'status_code', None) in (502, 503)\n", True),
    ("def f(ex):\n    msg = str(ex)\n    low = msg.casefold()\n    return 'x' in low\n", True),
    ("import re\ncheck = lambda last_error: re.search('40[13]', str(last_error))\n", True),
    ("def f(failure: Exception | None):\n    return str(failure).startswith('429')\n", True),
    ("def f(err: ClassifiedError):\n    return err.status_code == 429\n", False),
    ("def f(exc):\n    return str(exc) in SEEN\n", False),  # the needle, not the haystack
    ("def outer(exc):\n    def inner(resp):\n        return resp.status_code\n    return inner\n", False),
])
def test_shapes(src, flagged):
    assert bool(list(CHECKERS["HX014"](canonical_tree(ast.parse(src)), Ctx()))) is flagged


@pytest.mark.parametrize("path, applies", [(_AUX, True), ("agent/turn_x.py", True), ("agent/run_agent.py", False),
                                           ("agent/context_compressor.py", True), ("tests/agent/turn_x.py", False)])
def test_scope(path, applies):
    assert rule_applies(RULES_BY_ID["HX014"], path) is applies


@pytest.mark.parametrize("head, blocks", [({"agent/turn_retry.py": _NEW}, True),
                                          ({_AUX: "X = 1\n", "agent/auxiliary_quota.py": _NEW}, False)])  # moved
def test_ratchet(tmp_path, capsys, head, blocks):
    repo, _ = _repo(tmp_path)
    code, out = _verdict(repo, _commit(repo, {_AUX: _NEW}), head, capsys)
    assert code == int(blocks) and ("classify_api_error" in out) is blocks, out
