"""Release attribution must not choose silently between casefold-equivalent owners."""
import pytest

from scripts.releases import authors


@pytest.mark.parametrize("email", [
    "J.Doe@x.example", "j.doe@x.example", "J.DOE@X.EXAMPLE", "j.DOE@x.example",
])
@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_casefold_owners_are_rejected_even_for_exact_keys(email, reverse):
    pairs = [("J.Doe@x.example", "alice"), ("j.doe@x.example", "bob")]
    mapping = dict(reversed(pairs) if reverse else pairs)
    with pytest.raises(ValueError, match="ambiguous contributor identity"):
        authors.login_for_email(email, mapping)


def test_case_variants_for_one_owner_remain_deterministic():
    mapping = {"J.Doe@x.example": "alice", "j.doe@x.example": "alice"}
    assert authors.login_for_email("J.DOE@X.EXAMPLE", mapping) == "alice"
    assert authors.login_for_email("unknown@x.example", mapping) is None
    assert authors.login_for_email("J.DOE@X.EXAMPLE", {}) is None


@pytest.mark.parametrize("email", ["J.DOE@X.EXAMPLE", "123+someone@users.noreply.github.com"])
def test_release_and_audit_both_refuse_ambiguous_attribution(monkeypatch, email):
    from scripts import audit_pr_attribution as audit

    mapping = {email.lower(): "alice", email.upper(): "bob"}
    monkeypatch.setattr(authors, "AUTHOR_MAP", mapping)
    monkeypatch.setattr(audit, "LEGACY_AUTHOR_MAP", {})
    monkeypatch.setattr(audit, "_load_contributor_dir", lambda _directory: mapping)
    with pytest.raises(ValueError, match="ambiguous contributor identity"):
        authors.resolve_author("git-name", email)
    with pytest.raises(ValueError, match="ambiguous contributor identity"):
        audit.is_mapped(email)
