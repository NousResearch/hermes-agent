from gateway.group_control_intents import (
    has_followup_group_reference,
    has_current_group_reference,
    looks_like_group_chat_enable_request,
    looks_like_group_listen_disable_request,
    looks_like_group_listen_enable_request,
    looks_like_group_report_disable_request,
    looks_like_group_report_enable_request,
    looks_like_group_report_now_request,
    looks_like_group_runtime_status_query,
    resolve_oral_report_delivery_target,
    strip_current_group_reference_terms,
    wants_report_delivery_to_current_chat,
    wants_report_delivery_to_dm,
)


def test_has_current_group_reference_matches_shared_current_group_terms():
    assert has_current_group_reference("这个群只监听，不要聊天")
    assert has_current_group_reference("当前群日报先关掉")
    assert not has_current_group_reference("726109087 这个项目群只监听")


def test_has_followup_group_reference_accepts_group_deictic_words():
    assert has_followup_group_reference("你现在在群里能说话吗")
    assert has_followup_group_reference("那个群日报开了吗")
    assert not has_followup_group_reference("能说话吗")


def test_listen_disable_request_supports_resume_chat_wording():
    assert looks_like_group_listen_disable_request("停止QQ 群 192903718 的监听采集,允许开始聊天")
    assert looks_like_group_listen_disable_request("这个群恢复聊天，不要监听采集了")
    assert looks_like_group_listen_disable_request("726109087群你已经被踢出了 去掉")
    assert not looks_like_group_listen_disable_request("这个群只监听，不要走大模型")


def test_listen_enable_request_supports_collect_only_wording():
    assert looks_like_group_listen_enable_request("这个群只监听，不要走大模型")
    assert looks_like_group_listen_enable_request("把 726109087 这个群切成只监听采集")
    assert not looks_like_group_listen_enable_request("停止 726109087 的监听采集")


def test_group_runtime_status_query_detects_shared_status_asks():
    assert looks_like_group_runtime_status_query("这个群现在谁在监听，日报开了吗？")
    assert looks_like_group_runtime_status_query("726109087 这个群是不是监听模式，日报开了吗？")
    assert looks_like_group_runtime_status_query("这个群现在什么状态")
    assert not looks_like_group_runtime_status_query("这个群只监听，不要走大模型")


def test_chat_enable_and_report_helpers_match_shared_oral_terms():
    assert looks_like_group_chat_enable_request("这个群恢复聊天")
    assert looks_like_group_report_enable_request("这个群开启日报")
    assert looks_like_group_report_disable_request("这个群不要做汇报")
    assert looks_like_group_report_now_request("这个群现在日报")
    assert not looks_like_group_report_now_request("这个群开启日报")


def test_report_delivery_target_helpers_and_current_group_strip_work():
    assert wants_report_delivery_to_dm("这个群切到监听采集，日报发我私聊")
    assert wants_report_delivery_to_current_chat("这个群日报直接发在这个群")
    assert strip_current_group_reference_terms("这个群恢复聊天") == "恢复聊天"


def test_resolve_oral_report_delivery_target_honors_explicit_target_then_preference():
    assert (
        resolve_oral_report_delivery_target("这个群切到监听采集，日报发我私聊", prefer_dm=False)
        == "current_user_dm"
    )
    assert (
        resolve_oral_report_delivery_target("这个群日报直接发在这个群", prefer_dm=True)
        == "current_chat"
    )
    assert resolve_oral_report_delivery_target("这个群开启日报", prefer_dm=True) == "current_user_dm"
    assert resolve_oral_report_delivery_target("这个群立即汇报", prefer_dm=False) == "current_chat"
