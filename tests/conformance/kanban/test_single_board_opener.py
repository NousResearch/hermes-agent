"""``connect`` is the only function that opens a kanban board database.

The shared board's hosted refusal sits at the one opener of its database
(``hermes_cli.kanban_db_connect.connect`` — read-write or read-only, explicit
path, environment pin or resolved board; see ``test_raw_reader_refusal.py``),
and in the lowest arms ``connect`` delegates to (``_sqlite_connect`` and
``_open_readonly``, each refusing a hosted process by itself — see
``test_board_byte_boundaries.py``).
A raw ``sqlite3.connect`` on a board file anywhere else would bypass that
refusal silently, which is exactly how the subscription counter, the claim
checks, the worker-result read, the export snapshot and the repair probe each
used to reach the shared board. This test parses every module under
``hermes_cli/``, ``gateway/``, ``tools/`` and ``agent/`` and fails when a
``sqlite3.connect`` (or the ``sqlite3.Connection`` constructor, or an alias of
either) appears in a file that references kanban — by module name, by a
``kanban.db``/board-path constant, or by reaching the board's path helpers —
unless the open sits in the choke-point module's own connect arms, or is one of
the named exceptions below. Every exception opens a database that is NOT a kanban
board; if you need a new one, name it here and say what database it opens.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCAN_ROOTS = ("hermes_cli", "gateway", "tools", "agent")

#: The board's path helpers: a file that resolves a board path can point a raw
#: open at it even without "kanban" in its own name.
_BOARD_PATH_HELPERS = {"kanban_home", "kanban_db_path", "boards_root", "workspaces_root"}

#: The choke point itself. ``connect`` opens the database read-write through
#: ``_sqlite_connect`` and read-only through ``_open_readonly``; those arms are
#: the lowest openers of a board database, and each refuses a hosted process by
#: itself (whoever calls it, inside ``connect`` or imported directly) — no other
#: function in this module may open raw.
_CHOKE_POINT_OPENERS = {"_sqlite_connect", "_open_readonly"}

#: Named exceptions: (module path, function name) -> what database it opens.
#: Each entry opens a SQLite database that is not a kanban board; the comment
#: must say which one, or the exception goes away.
_NAMED_NON_BOARD_EXCEPTIONS = {
    ("hermes_cli/sqlite_safe_read.py", "connect_tracked"):
        "the generic tracked-connect helper: opens whatever SQLite database the caller "
        "names (state.db, kanban.db, any probe target); kanban_db_connect is one of its callers",
    ("hermes_cli/backup.py", "_query_ro_sqlite"):
        "operator backup/diagnostics probing any managed SQLite file (state.db, kanban.db, "
        "response_store.db, ...) read-only from the filesystem layout; offline operator tooling, "
        "never a tenant surface",
    ("hermes_cli/kanban_owner_recovery.py", "_admission_row"):
        "the owner's session-state admission ledger (state.db) named by an owner receipt, "
        "never a kanban board",
    ("hermes_cli/kanban_transfer.py", "_open_staged_export_copy"):
        "the export's staged snapshot copy under a temp dir (a private file the export just "
        "created); the live source board opens through the guarded connect in _snapshot_db, and "
        "the helper itself refuses a hosted process before opening anything",
}


def _kanban_referenced(tree: ast.AST, filename: str) -> bool:
    """Whether this module can reach a kanban board: by its own name, a
    ``kanban.db`` / board-path string constant, a board path helper, or an
    import of a kanban module."""
    if "kanban" in filename:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if "kanban.db" in text or "HERMES_KANBAN_DB" in text or "kanban/boards" in text:
                return True
        if isinstance(node, ast.Attribute) and node.attr in _BOARD_PATH_HELPERS:
            return True
        if isinstance(node, ast.Name) and node.id in _BOARD_PATH_HELPERS:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                modules = ([node.module] if node.module else []) + [alias.name for alias in node.names]
            if any(module and "kanban" in module for module in modules):
                return True
    return False


def _raw_opens(tree: ast.AST):
    """Every node that opens a SQLite database raw: a call or a bare reference
    (``connect_fn=sqlite3.connect``) to ``sqlite3.connect``, a call of the
    ``sqlite3.Connection`` constructor, or a call through an alias of either.
    Bare ``sqlite3.Connection`` references are type annotations / base classes,
    not opens."""
    aliases = set()  # local names bound to the sqlite3 openers
    opens = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            if isinstance(node, ast.ImportFrom) and node.module == "sqlite3":
                for alias in node.names:
                    if alias.name in ("connect", "Connection"):
                        aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and isinstance(node.value, ast.Attribute)
                        and node.value.attr == "connect"
                        and isinstance(node.value.value, ast.Name) and node.value.value.id == "sqlite3"):
                    aliases.add(target.id)
        elif isinstance(node, ast.Attribute):
            # ``sqlite3.connect`` — the open, whether called or passed around.
            if (node.attr == "connect"
                    and isinstance(node.value, ast.Name) and node.value.id == "sqlite3"):
                opens.append(node)
        elif isinstance(node, ast.Call):
            func = node.func
            if ((isinstance(func, ast.Attribute) and func.attr == "Connection"
                    and isinstance(func.value, ast.Name) and func.value.id == "sqlite3")
                    or (isinstance(func, ast.Name) and func.id in aliases)):
                opens.append(node)
    return opens


def _enclosing_function(tree: ast.AST, target: ast.AST):
    """Name of the innermost function containing ``target`` (None at module level)."""
    parents = {child: node for node in ast.walk(tree)
               for child in ast.iter_child_nodes(node)}
    node = target
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def test_the_guarded_connect_is_the_only_opener_of_a_board_database():
    violations = []
    for root in SCAN_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError):
                continue
            if not _kanban_referenced(tree, path.name):
                continue
            relative = str(path.relative_to(REPO_ROOT))
            for node in _raw_opens(tree):
                function = _enclosing_function(tree, node) or "<module>"
                if relative == "hermes_cli/kanban_db_connect.py":
                    if function in _CHOKE_POINT_OPENERS:
                        continue
                else:
                    exception = _NAMED_NON_BOARD_EXCEPTIONS.get((relative, function))
                    if exception:
                        continue
                violations.append(f"{relative}:{node.lineno} in {function}()")
    assert not violations, (
        "a kanban-referencing module opens a SQLite database without going through "
        "hermes_cli.kanban_db_connect.connect — route the board open through it "
        "(readonly=True for read-only callers), or, if the database is not a kanban "
        "board, add a named exception saying which database it opens:\n  "
        + "\n  ".join(violations)
    )
