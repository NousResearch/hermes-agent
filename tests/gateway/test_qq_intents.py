from gateway.qq_intents import (
    _QQ_DEFAULT_TRIGGER_ALIASES,
    _QQ_GROUP_ID_ANYWHERE_RE,
    _QQ_GROUP_ID_EXPLICIT_PATTERNS,
    _QQ_SEND_INLINE_PATTERNS,
    _QQ_VISIBLE_NAME_ALIASES,
    _looks_like_qq_background_status_query,
    _looks_like_qq_joined_group_list_query,
    _looks_like_qq_runtime_status_query,
)


def test_shared_visible_aliases_include_core_names():
    assert "马嘎" in _QQ_DEFAULT_TRIGGER_ALIASES
    assert "马哥" in _QQ_DEFAULT_TRIGGER_ALIASES
    assert "@马嘎" in _QQ_VISIBLE_NAME_ALIASES


def test_shared_group_id_patterns_still_match_oral_targets():
    assert _QQ_GROUP_ID_EXPLICIT_PATTERNS[0].search("qq_napcat:group:192903718")
    assert _QQ_GROUP_ID_EXPLICIT_PATTERNS[1].search("group:192903718")
    assert _QQ_GROUP_ID_ANYWHERE_RE.search("停止QQ 群 192903718 的监听采集")


def test_shared_send_inline_patterns_still_match_target_and_message():
    match = _QQ_SEND_INLINE_PATTERNS[0].search("往 QQ 群 192903718 发：绿帽哥！")
    assert match is not None
    assert match.group(1) == "192903718"
    assert "绿帽哥" in match.group(2)


def test_joined_group_list_query_requires_actual_group_list_intent():
    assert _looks_like_qq_joined_group_list_query("列一下群")
    assert not _looks_like_qq_joined_group_list_query("列一下群里的部署问题")


def test_background_status_query_does_not_treat_explicit_intel_status_query_as_job_status():
    assert not _looks_like_qq_background_status_query("看看情报员钢镚现在什么状态。")


def test_runtime_status_query_does_not_treat_explicit_intel_status_query_as_session_status():
    assert not _looks_like_qq_runtime_status_query("看看情报员钢镚现在什么状态。")


def test_runtime_status_query_does_not_treat_explicit_group_runtime_query_as_session_status():
    assert not _looks_like_qq_runtime_status_query("这个群现在什么状态")
