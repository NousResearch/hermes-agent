from gateway.config import Platform
from gateway.social_control_request_platform_specs import (
    QQ_SOCIAL_CONTROL_REQUEST_PLATFORM_SPEC,
    build_qq_social_control_request_platform_spec,
)
from gateway.qq_intents import (
    _looks_like_qq_social_policy_candidate,
    _looks_like_qq_social_request_list_query,
)
from gateway.qq_social_control_requests import (
    looks_like_qq_social_policy_query,
    match_qq_social_control_request,
    match_qq_social_request_type,
    qq_social_policy_notify_target,
)


def test_build_qq_social_control_request_platform_spec_uses_overrides():
    matcher = lambda **kwargs: ({"action": "list_requests"}, None)
    looks_like_request_list_query = lambda body: True
    looks_like_policy_candidate = lambda body: False
    looks_like_policy_query = lambda body: False
    request_type_matcher = lambda body: "friend"
    notify_target_resolver = lambda source, body: "qq_napcat:dm:179033731"

    spec = build_qq_social_control_request_platform_spec(
        request_matcher=matcher,
        looks_like_request_list_query=looks_like_request_list_query,
        looks_like_policy_candidate=looks_like_policy_candidate,
        looks_like_policy_query=looks_like_policy_query,
        request_type_matcher=request_type_matcher,
        notify_target_resolver=notify_target_resolver,
    )

    assert spec.platform is Platform.QQ_NAPCAT
    assert spec.request_matcher is matcher
    assert spec.looks_like_request_list_query is looks_like_request_list_query
    assert spec.looks_like_policy_candidate is looks_like_policy_candidate
    assert spec.looks_like_policy_query is looks_like_policy_query
    assert spec.request_type_matcher is request_type_matcher
    assert spec.notify_target_resolver is notify_target_resolver


def test_default_qq_social_control_request_platform_spec_is_qq():
    assert QQ_SOCIAL_CONTROL_REQUEST_PLATFORM_SPEC.platform is Platform.QQ_NAPCAT
    assert QQ_SOCIAL_CONTROL_REQUEST_PLATFORM_SPEC.request_matcher is match_qq_social_control_request
    assert (
        QQ_SOCIAL_CONTROL_REQUEST_PLATFORM_SPEC.looks_like_request_list_query
        is _looks_like_qq_social_request_list_query
    )
    assert (
        QQ_SOCIAL_CONTROL_REQUEST_PLATFORM_SPEC.looks_like_policy_candidate
        is _looks_like_qq_social_policy_candidate
    )
    assert QQ_SOCIAL_CONTROL_REQUEST_PLATFORM_SPEC.looks_like_policy_query is looks_like_qq_social_policy_query
    assert QQ_SOCIAL_CONTROL_REQUEST_PLATFORM_SPEC.request_type_matcher is match_qq_social_request_type
    assert QQ_SOCIAL_CONTROL_REQUEST_PLATFORM_SPEC.notify_target_resolver is qq_social_policy_notify_target
