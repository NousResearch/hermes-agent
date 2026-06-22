"""Per-channel profile routing helpers."""

from gateway.platforms.base import resolve_channel_profile


def test_resolve_channel_profile_dict_exact_match():
    extra = {"channel_profile_bindings": {"1518643081354936400": "peniby-pm"}}
    assert resolve_channel_profile(extra, "1518643081354936400") == "peniby-pm"


def test_resolve_channel_profile_dict_parent_fallback():
    extra = {"channel_profile_bindings": {"parent": "peniby-pm"}}
    assert resolve_channel_profile(extra, "thread", "parent") == "peniby-pm"


def test_resolve_channel_profile_list_format():
    extra = {
        "channel_profile_bindings": [
            {"id": "other", "profile": "other-pm"},
            {"id": "1518643081354936400", "profile": "peniby-pm"},
        ]
    }
    assert resolve_channel_profile(extra, "1518643081354936400") == "peniby-pm"


def test_resolve_channel_profile_blank_or_missing_is_none():
    assert resolve_channel_profile({}, "1518643081354936400") is None
    assert resolve_channel_profile({"channel_profile_bindings": {"x": "  "}}, "x") is None
