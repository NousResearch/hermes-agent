"""Focused tests for ``scripts/check_i18n_usage.py``.

Every case here feeds the checker a synthetic source string and a synthetic
catalog, so the suite exercises the real resolution logic without reading a
single repository file. The full-repository sweep is the script's own job and
runs in the lint workflow.

The scope cases pin the resolution bug the checker was written to avoid:
accumulating imports file-wide and then visiting calls in a separate pass
classifies calls that are nowhere near the import, and mistakes a local ``t``
for the catalog lookup.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_i18n_usage.py"


def _load_checker():
    """Import the checker script as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("check_i18n_usage", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_i18n_usage"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


def _keys(checker, src: str) -> list[str]:
    return [call.key for call in checker.scan_source(src, "sample.py")]


# ---------------------------------------------------------------------------
# Import forms the checker must recognize.
# ---------------------------------------------------------------------------


def test_plain_from_import(checker):
    assert _keys(checker, "from agent.i18n import t\nt('approval.choose')\n") == [
        "approval.choose"
    ]


def test_from_import_with_alias(checker):
    src = "from agent.i18n import t as _t\n_t('approval.choose')\n"
    assert _keys(checker, src) == ["approval.choose"]


def test_module_import_then_attribute_call(checker):
    src = "from agent import i18n\ni18n.t('approval.choose')\n"
    assert _keys(checker, src) == ["approval.choose"]


def test_aliased_module_import(checker):
    src = "import agent.i18n as messages\nmessages.t('approval.choose')\n"
    assert _keys(checker, src) == ["approval.choose"]


def test_unaliased_dotted_import(checker):
    """``import agent.i18n`` binds the top package, and the call goes through it.

    This form has no alias to key off, so a scanner that only tracks aliases
    reports zero calls for the file and silently checks nothing in it.
    """
    src = "import agent.i18n\nagent.i18n.t('approval.choose')\n"
    assert _keys(checker, src) == ["approval.choose"]


def test_plain_package_import_alone_is_not_enough(checker):
    """``import agent`` does not make the submodule reachable, so claim nothing."""
    src = "import agent\nagent.i18n.t('approval.choose')\n"
    assert _keys(checker, src) == []


def test_unrelated_import_of_the_same_name_is_ignored(checker):
    src = "from templating.helpers import t\nt('approval.choose')\n"
    assert _keys(checker, src) == []


# ---------------------------------------------------------------------------
# Lexical scope: a binding only classifies calls that can actually see it.
# ---------------------------------------------------------------------------


def test_function_local_import_does_not_reach_the_rest_of_the_file(checker):
    src = (
        "def render():\n"
        "    from agent.i18n import t\n"
        "    return t('inside.function')\n"
        "\n"
        "t('module.level.call')\n"
    )
    assert _keys(checker, src) == ["inside.function"]


def test_import_in_one_function_does_not_reach_another(checker):
    src = (
        "def first():\n"
        "    from agent.i18n import t\n"
        "    return t('inside.first')\n"
        "\n"
        "def second(t):\n"
        "    return t('inside.second')\n"
    )
    assert _keys(checker, src) == ["inside.first"]


def test_parameter_shadowing_the_import(checker):
    src = (
        "from agent.i18n import t\n"
        "\n"
        "def render(t):\n"
        "    return t('shadowed.by.parameter')\n"
        "\n"
        "t('real.call')\n"
    )
    assert _keys(checker, src) == ["real.call"]


def test_local_assignment_shadowing_the_import(checker):
    src = (
        "from agent.i18n import t\n"
        "\n"
        "def render(table):\n"
        "    t = table.lookup\n"
        "    return t('shadowed.by.assignment')\n"
    )
    assert _keys(checker, src) == []


def test_loop_target_shadowing_the_import(checker):
    src = (
        "from agent.i18n import t\n"
        "\n"
        "def render(rows):\n"
        "    for t in rows:\n"
        "        print(t('shadowed.by.loop'))\n"
    )
    assert _keys(checker, src) == []


def test_module_import_is_visible_inside_nested_functions(checker):
    src = (
        "from agent.i18n import t\n"
        "\n"
        "def outer():\n"
        "    def inner():\n"
        "        return t('nested.call')\n"
        "    return inner\n"
    )
    assert _keys(checker, src) == ["nested.call"]


def test_class_attribute_does_not_shadow_inside_methods(checker):
    """Class bodies are invisible to their own methods, exactly as Python resolves."""
    src = (
        "from agent.i18n import t\n"
        "\n"
        "class Panel:\n"
        "    t = staticmethod(str.strip)\n"
        "\n"
        "    def label(self):\n"
        "        return t('method.call')\n"
    )
    assert _keys(checker, src) == ["method.call"]


def test_class_body_call_uses_the_class_binding(checker):
    src = (
        "from agent.i18n import t\n"
        "\n"
        "class Panel:\n"
        "    t = staticmethod(str.strip)\n"
        "    label = t('class.body.call')\n"
    )
    assert _keys(checker, src) == []


def test_comprehension_target_shadows_the_import(checker):
    src = (
        "from agent.i18n import t\n"
        "\n"
        "def render(rows):\n"
        "    return [t('shadowed.in.comprehension') for t in rows]\n"
    )
    assert _keys(checker, src) == []


def test_conflicting_bindings_in_one_scope_claim_nothing(checker):
    src = (
        "def render(flag, table):\n"
        "    if flag:\n"
        "        from agent.i18n import t\n"
        "    else:\n"
        "        t = table.lookup\n"
        "    return t('ambiguous.binding')\n"
    )
    assert _keys(checker, src) == []


# ---------------------------------------------------------------------------
# What a call site yields.
# ---------------------------------------------------------------------------


def test_runtime_keys_are_skipped(checker):
    src = (
        "from agent.i18n import t\n"
        "\n"
        "def render(name, prefix):\n"
        "    t(name)\n"
        "    t(f'{prefix}.suffix')\n"
        "    return t('literal.key')\n"
    )
    assert _keys(checker, src) == ["literal.key"]


def test_call_records_line_kwargs_and_star_kwargs(checker):
    src = (
        "from agent.i18n import t\n"
        "t('a.plain')\n"
        "t('b.with.kwargs', count=3, lang='de')\n"
        "t('c.forwarded', **extra)\n"
    )
    calls = {call.key: call for call in checker.scan_source(src, "sample.py")}
    assert calls["a.plain"].lineno == 2
    assert calls["a.plain"].kwargs == frozenset()
    assert calls["b.with.kwargs"].kwargs == frozenset({"count", "lang"})
    assert calls["b.with.kwargs"].star_kwargs is False
    assert calls["c.forwarded"].star_kwargs is True


# ---------------------------------------------------------------------------
# The two rules the checker enforces.
# ---------------------------------------------------------------------------


def test_missing_key_is_reported_with_file_and_line(checker):
    src = "from agent.i18n import t\nt('known.key')\nt('typo.key')\n"
    calls = checker.scan_source(src, "gateway/slash.py")
    assert checker.missing_keys(calls, {"known.key": "hello"}) == [
        "gateway/slash.py:3: t('typo.key')"
    ]


def test_missing_format_kwarg_is_reported(checker):
    src = "from agent.i18n import t\nt('gateway.draining')\n"
    calls = checker.scan_source(src, "gateway/slash.py")
    problems = checker.missing_placeholders(calls, {"gateway.draining": "{count} left"})
    assert problems == [
        "gateway/slash.py:2: t('gateway.draining') does not supply ['count']"
    ]


def test_supplied_and_extra_kwargs_are_accepted(checker):
    src = "from agent.i18n import t\nt('gateway.draining', count=3, unused=1)\n"
    calls = checker.scan_source(src, "gateway/slash.py")
    assert checker.missing_placeholders(calls, {"gateway.draining": "{count} left"}) == []


def test_forwarded_kwargs_are_not_flagged(checker):
    src = "from agent.i18n import t\nt('gateway.draining', **payload)\n"
    calls = checker.scan_source(src, "gateway/slash.py")
    assert checker.missing_placeholders(calls, {"gateway.draining": "{count} left"}) == []


def test_lang_argument_does_not_satisfy_a_lang_placeholder(checker):
    """``lang=`` selects the catalog; it is not passed on to ``str.format``."""
    src = "from agent.i18n import t\nt('banner.locale', lang='de')\n"
    calls = checker.scan_source(src, "gateway/slash.py")
    assert checker.missing_placeholders(calls, {"banner.locale": "now in {lang}"}) == [
        "gateway/slash.py:2: t('banner.locale') does not supply ['lang']"
    ]


def test_non_string_catalog_values_are_not_scanned_for_placeholders(checker):
    src = "from agent.i18n import t\nt('limits.retries')\n"
    calls = checker.scan_source(src, "gateway/slash.py")
    assert checker.missing_placeholders(calls, {"limits.retries": 3}) == []


def test_flatten_builds_dotted_keys(checker):
    flat = checker.flatten({"gateway": {"draining": "{count} left"}, "top": "x"})
    assert flat == {"gateway.draining": "{count} left", "top": "x"}
