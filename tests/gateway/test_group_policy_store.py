from gateway.group_policy_store import (
    GroupPolicyStore,
    clear_scope_policy,
    get_scope_policy,
    normalize_group_scope_key,
    set_scope_policy,
    split_group_scope_key,
)


def test_normalize_group_scope_key_accepts_platform_and_chat_id():
    assert normalize_group_scope_key("qq_napcat", "987654321") == "qq_napcat:987654321"
    assert normalize_group_scope_key("weixin", "group@chatroom") == "weixin:group@chatroom"


def test_normalize_group_scope_key_accepts_prebuilt_scope_key():
    assert normalize_group_scope_key("qq_napcat:987654321") == "qq_napcat:987654321"


def test_split_group_scope_key_parses_platform_and_chat_id():
    assert split_group_scope_key("qq_napcat:987654321") == ("qq_napcat", "987654321")
    assert split_group_scope_key("weixin:group@chatroom") == ("weixin", "group@chatroom")


def test_group_policy_store_round_trips_scope_key():
    policy = set_scope_policy(
        "qq_napcat:987654321",
        mode="collect_only",
        daily_report_enabled=True,
        daily_report_target="qq_napcat:dm:179033731",
        updated_by="test",
    )

    assert policy["scope_key"] == "qq_napcat:987654321"
    assert policy["platform"] == "qq_napcat"
    assert policy["chat_id"] == "987654321"
    assert policy["mode"] == "collect_only"
    assert get_scope_policy("qq_napcat:987654321")["daily_report_enabled"] is True


def test_group_policy_store_supports_instance_api():
    store = GroupPolicyStore()
    policy = store.set_policy(
        "weixin:group@chatroom",
        mode="project_mode",
        updated_by="test",
    )

    assert policy["scope_key"] == "weixin:group@chatroom"
    assert policy["platform"] == "weixin"
    assert policy["chat_id"] == "group@chatroom"
    assert store.has_policy("weixin:group@chatroom") is True


def test_clear_scope_policy_resets_to_default():
    set_scope_policy("qq_napcat:123456", mode="disabled", updated_by="test")

    cleared = clear_scope_policy("qq_napcat:123456")

    assert cleared["scope_key"] == "qq_napcat:123456"
    assert cleared["mode"] == "default"
    assert cleared["archive_enabled"] is False


def test_group_policy_store_reads_legacy_qq_bare_group_keys(tmp_path):
    path = tmp_path / "qq_group_policies.json"
    path.write_text(
        (
            '{"version":1,"updated_at":"2026-04-16T00:00:00+08:00","groups":'
            '{"987654321":{"mode":"collect_only","archive_enabled":true,"group_name":"研发群"}}}'
        ),
        encoding="utf-8",
    )
    store = GroupPolicyStore(path=path)

    policy = store.get_policy("qq_napcat:987654321")

    assert policy["scope_key"] == "qq_napcat:987654321"
    assert policy["platform"] == "qq_napcat"
    assert policy["chat_id"] == "987654321"
    assert policy["mode"] == "collect_only"
    assert policy["chat_name"] == "研发群"
