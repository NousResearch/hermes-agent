"""The local secret backend runs every writer through one error boundary, structurally.

The no-plaintext contract is structural, and so is this check. Whichever writer a credential
write runs — the env file's writers, the auth store's load, a record's merge and save, the
copy of the record being filed — the call must sit inside ``LocalSecretStore._run_writer``,
the one method that catches a writer's failure before it reaches any context manager the
write holds (an ``__exit__`` hands the exception to the generator it wraps and assigns its
``__traceback__``, and a hostile writer's attribute hook then runs while the record is still
bound in the frames being unwound) and raises the port's fixed error in the writer's place.

These tests parse the backend's own source, so the boundary cannot be routed around one path
at a time: a call to a writer outside the boundary fails here, whatever new path grows it —
the sibling modules probe the boundary's behaviour, and this one pins where the calls sit.
Reads keep today's raw calls and their own errors; only the write path is normalized.
"""

import ast
from pathlib import Path

#: The backend's source, as the install ships it.
SOURCE = Path(__file__).resolve().parents[3] / "hermes_secret_port_local.py"

#: Every writer the backend's write path runs: the env file's writers, and the auth store's
#: load, merge and save. Each is called only inside the writer boundary.
WRITERS = frozenset({"save_env_value", "remove_env_value", "_load_auth_store",
                     "_store_provider_state", "_save_auth_store"})
#: The callables the boundary may run beyond the writers: the copy of the record a write
#: files, made inside the boundary so even a record that cannot be copied fails as the port's
#: fixed write error, its text unread.
BOUNDARY_CALLABLES = WRITERS | {"dict"}
#: The read methods: reads keep today's raw calls, and propagate their own errors.
READERS = frozenset({"get", "names", "get_auth", "_resolved"})


def _store_methods() -> dict[str, ast.FunctionDef]:
    """The backend's methods, parsed from its own source — the tree the install ships, not
    the imported class: where a writer call sits is a property of that code."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LocalSecretStore":
            return {item.name: item for item in node.body if isinstance(item, ast.FunctionDef)}
    raise AssertionError(f"LocalSecretStore is missing from {SOURCE.name}")


def _callee(call: ast.Call) -> str | None:
    """The name a call's target carries: ``save_env_value``, ``auth._save_auth_store``."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _calls(node: ast.AST) -> list[ast.Call]:
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)]


def test_every_writer_call_sits_inside_the_boundary_or_a_read():
    """A writer call anywhere but the boundary — or today's reads, which keep their own
    errors — bypasses the catch that keeps a writer's failure out of the unwind and its
    stored values out of the error the caller reads."""
    methods = _store_methods()
    assert "_run_writer" in methods, "the writer boundary _run_writer is gone from the local backend"
    for name, method in methods.items():
        if name == "_run_writer" or name in READERS:
            continue
        for call in _calls(method):
            callee = _callee(call)
            assert callee not in WRITERS, (
                f"{name} calls the writer {callee!r} directly, outside the _run_writer boundary")


def test_every_write_routes_its_writer_through_the_boundary():
    """Each of the port's writes — the set, the delete, the auth record's write — hands its
    writer to the boundary by name, so the backend's whole writer set crosses it and nothing
    else does: a write path that computes its own writer, or stops routing one, is a path the
    no-plaintext probes have to find all over again."""
    methods = _store_methods()
    handed: set[str] = set()
    for name in ("set", "delete", "set_auth"):
        routed = [call for call in _calls(methods[name]) if _callee(call) == "_run_writer"]
        assert routed, f"{name} no longer runs its writer through _run_writer"
        for call in routed:
            assert len(call.args) >= 3, f"{name} calls _run_writer without its writer argument"
            writer = call.args[2]
            assert isinstance(writer, (ast.Name, ast.Attribute)), (
                f"{name} hands _run_writer a computed writer, not a named one")
            handed.add(writer.id if isinstance(writer, ast.Name) else writer.attr)
    assert WRITERS <= handed, f"writers no longer routed through the boundary: {sorted(WRITERS - handed)}"
    assert handed <= BOUNDARY_CALLABLES, (
        f"callables the boundary was not meant to run: {sorted(handed - BOUNDARY_CALLABLES)}")
