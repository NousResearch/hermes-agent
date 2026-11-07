"""The conversation backend's ``_store`` modules carry main's store code with only the receivers changed.

Every copied definition has main's text after main's imports of copied names are pointed at their copies, the spec's
substitutions are applied and ``receiver_form`` has put a class member at module level with the store holder as its
receiver (see ``_copies``); waiver comments, which main's lines do not carry, are dropped before the comparison. Each
waived line of the store's copies, and of the store holder's, the schema's and coordination's, is checked against
main's function through the table in ``core/_moved.py``, and every waiver in a ``_store`` module sits in a copy the
table lists. Each module defines exactly the spec's definitions, and no copied member has the name of a store holder
attribute, so a ``self.<name>`` of main's is either a copied member or the holder's own. Main's text is read from the
pinned main checkout.
"""

import ast
import functools
from pathlib import Path

import pytest

from persistence.sqlite import _schema
from persistence.sqlite._holder import StoreHolder
from tests.persistence_sqlite._legacy_store import legacy_store, oracle_tree
from tests.persistence_sqlite.conversation._copies import (
    DELEGATES, catalog, definition, expected_text, load_specs, segment,
)
from tests.persistence_sqlite.core._moved import (
    COPIES, WAIVER_IDS, WAIVERS, assert_copy, strip_moved_allows, waived_functions,
)

__all__ = ["legacy_store"]

STORE_DIR = Path(__file__).resolve().parents[3] / "persistence" / "sqlite" / "conversation" / "_store"
SPECS = load_specs()
CASES = [(target, i, name, False) for target, copies in SPECS.items() for i, spec in enumerate(copies)
         for name in spec.names] + [
    (target, i, name, True) for target, copies in SPECS.items() for i, spec in enumerate(copies) for name in spec.members
]
_STORE_WAIVERS = [(case, case_id) for case, case_id in zip(WAIVERS, WAIVER_IDS)
                  if not case[0].backend.startswith("persistence/sqlite/admin/")]


def _source(target: str) -> str:
    return strip_moved_allows((STORE_DIR / f"{target}.py").read_text(encoding="utf-8"))


def _bound_names(node: ast.stmt) -> set[str]:
    """The name a spec lists a definition by: a def's or class's name, an assignment's first target name."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return {next(n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name))} if targets else {"<stmt>"}


def _guarded_import(node: ast.stmt) -> bool:
    """A module-level ``try:`` of imports only, with ``except ImportError`` fallbacks (the legacy optional imports)."""
    return (isinstance(node, ast.Try) and all(isinstance(n, (ast.Import, ast.ImportFrom)) for n in node.body)
            and all(h.type is not None and ast.unparse(h.type) == "ImportError" for h in node.handlers))


@functools.cache
def _catalog():
    """The copies' catalog over the checkout's legacy store; called inside a test holding ``legacy_store``."""
    return catalog(SPECS, oracle_tree())


@pytest.mark.parametrize(("target", "copy", "name", "member"), CASES, ids=[f"{t}.{n}" for t, _, n, _ in CASES])
def test_copy_matches_legacy(legacy_store, target, copy, name, member):
    spec = SPECS[target][copy]
    source = _source(target)
    node = definition(ast.parse(source), name)
    assert segment(source, node) == expected_text(spec, name, oracle_tree(), _catalog(), member=member)


@pytest.mark.parametrize(("copy", "index"), [case for case, _ in _STORE_WAIVERS], ids=[i for _, i in _STORE_WAIVERS])
def test_each_waived_line_is_a_line_of_mains_function(copy, index):
    assert_copy(copy, index, oracle_tree())


@pytest.mark.parametrize("target", sorted(SPECS))
def test_every_waiver_sits_in_a_listed_copy(target):
    path = f"persistence/sqlite/conversation/_store/{target}.py"
    assert waived_functions(path) == {copy.function for copy in COPIES if copy.backend == path}


@pytest.mark.parametrize("target", sorted(SPECS))
def test_module_defines_only_its_copies(target):
    tree = ast.parse(_source(target))
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) or _guarded_import(node) or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        ):
            continue
        defined |= _bound_names(node)
    assert defined == {name for spec in SPECS[target] for name in (*spec.names, *spec.members)}


def test_every_store_module_has_a_spec():
    modules = {path.stem for path in STORE_DIR.glob("*.py") if path.name != "__init__.py"}
    assert modules == set(SPECS)


def test_copied_members_leave_the_holder_its_own():
    # The core's copied members are among the functions and assignments of ``_schema``: all of them are checked.
    tree = ast.parse(Path(_schema.__file__).read_text(encoding="utf-8"))
    core = set().union(*(_bound_names(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))))
    members = {name for copies in SPECS.values() for spec in copies for name in spec.members}
    shadowing = sorted(name for name in members | set(DELEGATES) | core if hasattr(StoreHolder, name))
    assert shadowing == []
