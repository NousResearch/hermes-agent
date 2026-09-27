"""Claude-parity workspace segments: contracted path + rich git state."""

from datetime import datetime, timedelta

from cli import HermesCLI
from hermes_cli import status_bar_git
from hermes_cli.status_bar_git import format_git_status, format_status_path, parse_porcelain_v2

_PORCELAIN = """# branch.oid abc123
# branch.head feature/login
# branch.ab +3 -2
1 M. N... 100644 100644 100644 aaa bbb staged.txt
1 .M N... 100644 100644 100644 ccc ddd modified.txt
1 MM N... 100644 100644 100644 eee fff both.txt
2 R. N... 100644 100644 100644 ggg hhh R100 new.txt\told.txt
u UU N... 100644 100644 100644 100644 iii jjj kkk conflict.txt
? untracked.txt
"""


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "anthropic/claude-opus-5"
    cli_obj.session_start = datetime.now() - timedelta(minutes=3)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    return cli_obj


def test_porcelain_v2_counts_every_dirty_class():
    status = parse_porcelain_v2(_PORCELAIN)

    assert status["branch"] == "feature/login"
    assert (status["ahead"], status["behind"]) == (3, 2)
    # `MM` counts on both sides; the rename record is staged-only.
    assert (status["staged"], status["modified"]) == (3, 2)
    assert (status["conflicted"], status["untracked"]) == (1, 1)
    assert parse_porcelain_v2("# branch.head (detached)")["branch"] == "detached"


def test_git_status_label_orders_counters_and_hides_zeros():
    assert format_git_status(parse_porcelain_v2(_PORCELAIN)) == " feature/login =1 +3 !2 ?1 ⇡3 ⇣2"
    assert format_git_status({"branch": "main"}) == " main"
    assert format_git_status({"branch": ""}) == ""


def test_status_path_contracts_only_leading_components(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    deep = tmp_path.joinpath("alpha", ".config", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "tail")
    deep.mkdir(parents=True)

    assert format_status_path(str(tmp_path / "projects")) == "~/projects"
    assert format_status_path("/private/tmp/hermes-tui-vim") == "/p/t/hermes-tui-vim"
    assert format_status_path(str(deep)) == "~/a/.c/c/c/c/c/c/c/c/tail"


def test_workspace_and_git_segments_lead_the_bar(monkeypatch, tmp_path):
    """Path + git state render by default and sit ahead of the model segment."""
    monkeypatch.setattr(
        status_bar_git, "current_git_status", lambda cwd=None: parse_porcelain_v2(_PORCELAIN))
    monkeypatch.setattr(status_bar_git, "format_status_path", lambda cwd=None: "~/projects/hermes")
    status_bar_git._cache.clear()

    cli_obj = _make_cli()
    cli_obj._status_bar_field_set_cache = None
    snapshot = cli_obj._get_status_bar_snapshot()

    assert snapshot["git_branch"] == "feature/login"
    assert snapshot["git_status_label"].startswith(" feature/login")
    assert snapshot["workspace_label"]

    text = "".join(
        t for seg in cli_obj._status_bar_segments(snapshot, 120, None, False, styled=False) for _, t in seg)

    assert text.index(snapshot["workspace_label"]) < text.index("claude-opus-5")
    assert "feature/login" in text
