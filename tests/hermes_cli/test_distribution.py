"""One official repository, and every update surface derives from it."""

from hermes_cli import banner, distribution, update_cmd_git


def test_every_consumer_follows_the_single_slug(monkeypatch):
    slug = distribution.repo_slug()

    assert distribution.https_url() == f"https://github.com/{slug}.git"
    assert banner._UPSTREAM_REPO_URL == distribution.https_url()
    assert banner._OFFICIAL_REPO_CANONICAL == distribution.canonical()
    assert update_cmd_git.OFFICIAL_REPO_URL in update_cmd_git.OFFICIAL_REPO_URLS
    assert all(slug in url for url in update_cmd_git.OFFICIAL_REPO_URLS)
    assert slug in distribution.compare_api_url("a" * 40, "b" * 40)
    assert distribution.branch_zip_url("main").endswith(f"{slug}/archive/refs/heads/main.zip")


def test_the_forked_from_project_is_not_an_official_remote():
    assert update_cmd_git._is_fork("https://github.com/NousResearch/hermes-agent.git") is True
    assert update_cmd_git._is_fork(distribution.https_url()) is False


def test_override_accepts_owner_name_only(monkeypatch):
    monkeypatch.setenv("HERMES_DISTRIBUTION_REPO", "acme/mirror")
    assert distribution.repo_slug() == "acme/mirror"

    for bad in ("nonsense", "a/b/c", "has space/repo", ""):
        monkeypatch.setenv("HERMES_DISTRIBUTION_REPO", bad)
        assert distribution.repo_slug() == "aievolutionpl/AGENT_CZESIEK"
