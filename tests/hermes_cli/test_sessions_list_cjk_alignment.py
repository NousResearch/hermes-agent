"""``hermes sessions list`` aligns its columns by display width, not characters (#44199).

CJK characters occupy two terminal columns, so ``str`` slices and format-spec padding (which
count characters) shift every column to the right of CJK content. Every layout — including the
workspace-aware ones and ``--workspace`` filtering — must start the ID column at the same display
column on every row.
"""

from argparse import Namespace
from unittest.mock import MagicMock

import pytest
from wcwidth import wcswidth

from hermes_cli import sessions_cmd

_CJK = dict(title="整理论文和更新README #4 以及其他很长的标题文字", preview="我更新了paper/和其他文件，然后继续整理论文的其余部分")
_ASCII = dict(title="Research", preview="read docs")


def _row(sid, *, title, preview, repo=None):
    return {"id": sid, "title": title, "preview": preview, "last_active": None, "source": "cli",
            "git_repo_root": repo, "cwd": None}


def _list(capsys, rows, workspace=None):
    db = MagicMock()
    db.list_sessions_rich.return_value = rows
    sessions_cmd._cmd_list(db, Namespace(limit=20, source=None, workspace=workspace))
    return capsys.readouterr().out


def _id_columns(out, ids):
    return {wcswidth(line.split(sid)[0]) for line in out.splitlines() for sid in ids if line.endswith(sid)}


@pytest.mark.parametrize("with_titles", [True, False])
@pytest.mark.parametrize("repo", [None, "/src/项目仓库名称很长很长很长"])
def test_id_column_aligned_with_cjk_rows(capsys, with_titles, repo):
    rows = [_row("sess_cjk", **_CJK, repo=repo), _row("sess_ascii", **_ASCII, repo=repo)]
    if not with_titles:
        for r in rows:
            r["title"] = None
    out = _list(capsys, rows)
    cols = _id_columns(out, ("sess_cjk", "sess_ascii"))
    assert len(cols) == 1, out


def test_workspace_filter_listing_aligned_with_cjk_rows(capsys):
    repo = "/src/论文仓库"
    rows = [_row("sess_cjk", **_CJK, repo=repo), _row("sess_ascii", **_ASCII, repo=repo),
            _row("sess_other", **_ASCII, repo="/src/other")]
    out = _list(capsys, rows, workspace="论文仓库")
    assert "sess_other" not in out
    cols = _id_columns(out, ("sess_cjk", "sess_ascii"))
    assert len(cols) == 1, out


def test_ascii_rows_unchanged(capsys):
    out = _list(capsys, [_row("sess_a", title="Coding", preview="build feature")])
    line = next(l for l in out.splitlines() if l.endswith("sess_a"))
    assert line.startswith(f"{'Coding':<32} {'build feature':<40} ")


def test_pinned_listing_aligned_with_cjk_rows(capsys):
    db = MagicMock()
    db.list_sessions_rich.return_value = [dict(_row("sess_cjk", **_CJK), pinned=True),
                                          dict(_row("sess_ascii", **_ASCII), pinned=True)]
    sessions_cmd._cmd_pinned(db, Namespace(json=False, source=None))
    out = capsys.readouterr().out
    assert len(_id_columns(out, ("sess_cjk", "sess_ascii"))) == 1, out
