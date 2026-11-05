"""The store holder's ``state_meta`` prefix read returns nothing for an empty prefix."""
from persistence_assembly import open_persistence, profile_scope


def test_empty_prefix(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.state_meta.set("loop:s1", "1")
        assert store.holder().list_meta_prefix("") == []
