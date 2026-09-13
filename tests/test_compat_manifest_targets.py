"""Plugin-compat pointers must resolve to the SAME object the name moved to, never a same-named stranger.

Regression: ``hermes_cli.kanban_db.connect`` was pointed at ``hermes_cli.projects_db.connect`` (a different
database, no ``board=`` parameter) because the generator ranked candidate homes by path proximity. The
manifest codified the mistake, so the compat lint treated it as valid.

Invariant checked here: for every ``moved-lazy`` entry whose target module also exists in the manifest of
some other facade under the same name, or whose facade stem has a sibling ``<stem>_*`` module defining the
name, the facade attribute IS the sibling's object.
"""
import importlib
import json
import pkgutil
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "compat_manifest.json"

# This file resolves every pointer on purpose; the once-per-name plugin warning is expected here.
pytestmark = [
    pytest.mark.skipif(not MANIFEST.exists(), reason="compat layer removed (scheduled revert)"),
    pytest.mark.filterwarnings("ignore::FutureWarning"),
]


def _entries():
    return [e for e in json.loads(MANIFEST.read_text())["entries"] if e["kind"] == "moved-lazy"]


def _sibling_modules(facade: str) -> list[str]:
    pkg, _, stem = facade.rpartition(".")
    try:
        parent = importlib.import_module(pkg) if pkg else None
    except Exception:
        return []
    paths = getattr(parent, "__path__", None) if parent else [str(ROOT)]
    if not paths:
        return []
    prefix = f"{pkg}." if pkg else ""
    return [prefix + m.name for m in pkgutil.iter_modules(paths) if m.name.startswith(stem + "_")]


def test_moved_lazy_pointers_resolve_to_the_split_off_siblings_object():
    """When a facade's own ``<stem>_*`` sibling binds the name, the facade attribute must be THAT object.

    A sibling may legitimately re-import the value from elsewhere (then the pointer target is the origin and
    the objects are identical); what must never happen is the pointer resolving to a same-named stranger.
    """
    bad = []
    for e in _entries():
        facade, name = e["facade"], e["name"]
        sibs = _sibling_modules(facade)
        if not sibs:
            continue
        try:
            got = getattr(importlib.import_module(facade), name)
        except Exception as exc:  # unresolvable pointer is its own failure
            bad.append((facade, name, f"unresolvable: {exc!r}"))
            continue
        for s in sibs:
            try:
                mod = importlib.import_module(s)
            except Exception:
                continue
            if name in vars(mod):
                sib_obj = vars(mod)[name]
                same = (sib_obj == got) if isinstance(got, (int, float, str, bytes, bool, type(None))) else (sib_obj is got)
                if not same:
                    bad.append((facade, name, e["target"], s))
    assert not bad, f"compat pointers resolve to a different object than the facade's own sibling binds: {bad}"


class TestXAIVideoCompatEntryPoint:
    """``run_xai_video_generation`` must take the provider's path, not the raw coroutine.

    ``XAIVideoGenProvider.generate`` calls ``_run_xai_video``, which resolves credentials and
    is the only thing that supplies the async flow's keyword-only ``api_key``/``base_url``. A
    restored entry point that calls the flow directly cannot satisfy those, so it raises out
    of the plugin instead of returning the structured error shape every other entry point
    returns (``run_xai_video_edit`` / ``run_xai_video_extend`` both delegate).
    """

    @staticmethod
    def _args(**overrides):
        args = dict(
            prompt="a cat on a skateboard", model=None, explicit_model=False, image_url=None,
            reference_image_urls=None, duration=None, aspect_ratio="16:9", resolution="720p",
        )
        args.update(overrides)
        return args

    def test_without_credentials_it_returns_the_auth_required_error(self, monkeypatch):
        import plugins.video_gen.xai as xai

        monkeypatch.setattr(xai, "_resolve_xai_credentials", lambda: ("", "https://api.x.ai/v1"))
        result = xai.run_xai_video_generation(**self._args())
        assert result["success"] is False
        assert result["error_type"] == "auth_required"
        assert result["provider"] == "xai"
        assert "XAI_API_KEY" in result["error"]

    def test_resolved_credentials_reach_the_async_flow(self, monkeypatch):
        import plugins.video_gen.xai as xai

        seen = {}

        async def _recorder(**kwargs):
            seen.update(kwargs)
            return {"success": True, "video": "/tmp/out.mp4"}

        monkeypatch.setattr(
            xai, "_resolve_xai_credentials", lambda: ("fake-key", "https://api.x.ai/v1"),
        )
        monkeypatch.setattr(xai, "_generate_xai_video_async", _recorder)
        result = xai.run_xai_video_generation(**self._args(prompt="a dog", aspect_ratio="1:1"))
        assert result == {"success": True, "video": "/tmp/out.mp4"}
        assert seen["api_key"] == "fake-key"
        assert seen["base_url"] == "https://api.x.ai/v1"
        # …and the caller's own arguments still arrive unchanged.
        assert seen["prompt"] == "a dog"
        assert seen["aspect_ratio"] == "1:1"
        assert seen["resolution"] == "720p"

def test_kanban_db_connect_opens_a_kanban_board(tmp_path, monkeypatch):
    """The historical ``kanban_db.connect(board=...)`` opens a Kanban DB, not projects.db."""
    import hermes_cli.kanban_db as kanban_db
    import hermes_cli.kanban_db_connect as kanban_db_connect

    assert kanban_db.connect is kanban_db_connect.connect
    assert kanban_db.connect_closing is kanban_db_connect.connect_closing
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "board.db"
    conn = kanban_db.connect(db, board="qa")
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "tasks" in tables, tables
    assert not (tmp_path / "projects.db").exists()
    assert isinstance(sqlite3.connect(db), sqlite3.Connection)
