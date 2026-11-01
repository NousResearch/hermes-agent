"""Claim fixture order follows the claim stamp, not tagger-date/config tie breaks."""
import pytest

from tests.scripts.test_release_build_commit import claims, fixture_repo, git


@pytest.mark.parametrize("tag_sort", ["refname", "-version:refname"])
def test_claim_listing_ignores_tied_tagger_dates_and_user_sort(request, monkeypatch, tag_sort):
    _, upstream, _ = request.getfixturevalue("fixture_repo")
    git(upstream, 'config', 'tag.sort', tag_sort)
    git(upstream, 'config', 'user.name', 'Fixture')
    git(upstream, 'config', 'user.email', 'fixture@example.invalid')
    monkeypatch.setenv('GIT_COMMITTER_DATE', '2026-10-29T08:06:55Z')
    tags = ['commit.20261029T080655Z-v1.2.3', 'commit.20261029T080656Z-v1.2.3']
    for tag in tags:
        git(upstream, 'tag', '-a', tag, 'main', '-m', 'fixture')
    dates = git(upstream, 'for-each-ref', 'refs/tags/commit.*', '--format=%(creatordate:unix)').splitlines()
    assert len(dates) == 2 and len(set(dates)) == 1
    assert claims(upstream) == tags
