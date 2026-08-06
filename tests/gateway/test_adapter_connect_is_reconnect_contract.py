"""Regression: every platform adapter's ``connect()`` must accept the
``is_reconnect`` keyword-only argument.

The gateway reconnect watcher forwards ``is_reconnect=True`` to every
adapter on every retry (see ``GatewayRunner._call_adapter_connect`` in
``gateway/run.py``). An adapter whose ``connect()`` signature omits
``is_reconnect`` blows up on the first reconnect attempt with::

    TypeError: <Foo>Adapter.connect() got an unexpected
               keyword argument 'is_reconnect'

…and never recovers, leaving that platform silently disconnected until
the operator manually restarts the gateway. This exact bug shipped for
``QQAdapter`` and was only discovered after messages stopped flowing on
the QQ channel for hours.

To prevent this class of bug from regressing, we statically parse every
``adapter.py`` under ``gateway/platforms/`` and ``plugins/platforms/``
and assert that its ``connect()`` method accepts an ``is_reconnect``
keyword. Doing this via AST (rather than importing) avoids pulling every
platform's optional third-party SDK (aiohttp, slack_sdk, telegram,
matrix-nio, etc.) into the test environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories that hold platform adapters. Each entry is a directory
# whose immediate children are either ``adapter.py`` files or
# sub-packages that expose one.
ADAPTER_ROOTS = [
    REPO_ROOT / "gateway" / "platforms",
    REPO_ROOT / "plugins" / "platforms",
]


def _iter_adapter_files() -> list[Path]:
    """Every ``*adapter*.py`` under the two adapter roots.

    We intentionally cast a wide net (any ``adapter.py`` / ``*_adapter.py``
    inside these trees) so a new platform can't sneak in without the
    contract check firing.
    """
    files: list[Path] = []
    for root in ADAPTER_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if path.name == "adapter.py" or path.stem.endswith("_adapter"):
                files.append(path)
    return sorted(files)


def _find_adapter_classes(module: ast.Module) -> list[ast.ClassDef]:
    """Classes that look like a platform adapter.

    Heuristic: any class whose name ends in ``Adapter`` and that defines
    an ``async def connect`` method. This catches every subclass of
    ``BasePlatformAdapter`` in the tree today (QQAdapter, TelegramAdapter,
    SlackAdapter, …) without importing the base class.
    """
    hits: list[ast.ClassDef] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef):
            continue
        if not node.name.endswith("Adapter"):
            continue
        has_connect = any(
            isinstance(item, ast.AsyncFunctionDef) and item.name == "connect"
            for item in node.body
        )
        if has_connect:
            hits.append(node)
    return hits


def _connect_accepts_is_reconnect(cls: ast.ClassDef) -> bool:
    """True iff the class's own ``connect()`` accepts ``is_reconnect``.

    Accepts the kwarg via:
    - keyword-only argument named ``is_reconnect`` (the canonical form
      used by ``BasePlatformAdapter``), OR
    - ``**kwargs`` catch-all (also safe — the kwarg is absorbed).
    """
    for item in cls.body:
        if not (isinstance(item, ast.AsyncFunctionDef) and item.name == "connect"):
            continue
        args = item.args
        if any(a.arg == "is_reconnect" for a in args.kwonlyargs):
            return True
        if any(a.arg == "is_reconnect" for a in args.args):
            return True
        if args.kwarg is not None:  # **kwargs
            return True
        return False
    return False


ADAPTER_FILES = _iter_adapter_files()


def test_adapter_discovery_finds_platforms():
    """Sanity: the discovery walker actually found a meaningful set of
    adapters. If this drops to a trivial number, the glob broke and the
    contract test below is silently passing on nothing.
    """
    assert len(ADAPTER_FILES) >= 20, (
        f"Expected to discover >=20 platform adapter files under "
        f"{[str(p) for p in ADAPTER_ROOTS]}, found {len(ADAPTER_FILES)}. "
        f"The discovery glob is likely broken."
    )


@pytest.mark.parametrize(
    "adapter_file",
    ADAPTER_FILES,
    ids=lambda p: str(p.relative_to(REPO_ROOT)),
)
def test_adapter_connect_accepts_is_reconnect(adapter_file: Path):
    """Every ``*Adapter.connect()`` must accept ``is_reconnect``.

    This is the contract enforced by ``BasePlatformAdapter.connect`` and
    relied on by ``GatewayRunner._call_adapter_connect``. Violating it
    silently disables the affected platform after its first reconnect.
    """
    source = adapter_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(adapter_file))
    except SyntaxError as exc:
        pytest.fail(f"Could not parse {adapter_file}: {exc}")

    classes = _find_adapter_classes(tree)
    if not classes:
        pytest.skip(
            f"{adapter_file.relative_to(REPO_ROOT)} has no *Adapter class "
            f"with an async connect() — nothing to check."
        )

    offenders = [cls.name for cls in classes if not _connect_accepts_is_reconnect(cls)]

    assert not offenders, (
        f"{adapter_file.relative_to(REPO_ROOT)}: the following adapter "
        f"class(es) define `async def connect()` WITHOUT accepting the "
        f"`is_reconnect` kwarg: {offenders}. "
        f"Add `*, is_reconnect: bool = False` to the signature "
        f"(matching BasePlatformAdapter.connect). "
        f"The gateway reconnect watcher forwards this kwarg on every "
        f"retry — an adapter that rejects it silently disconnects after "
        f"the first outage."
    )
