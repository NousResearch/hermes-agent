from types import SimpleNamespace

from hermes_cli.projects_db import branch_name_for


def test_long_title_slug_is_cut_at_a_word_boundary():
    project = SimpleNamespace(slug="proj", name="Proj")
    name = branch_name_for(project, "t_1", title="fix the flaky telegram reconnection watcher loop")
    slug = name.split("/t_1-", 1)[1]
    assert len(slug) <= 40
    assert not slug.endswith("-")
    assert slug == "fix-the-flaky-telegram-reconnection"


def test_short_title_is_unchanged():
    project = SimpleNamespace(slug="proj", name="Proj")
    assert branch_name_for(project, "t_1", title="Fix It") == "proj/t_1-fix-it"
