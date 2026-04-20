from gateway.config import Platform
from gateway.intel_control_request_platform_specs import (
    QQ_INTEL_CONTROL_REQUEST_PLATFORM_SPEC,
    build_qq_intel_control_request_platform_spec,
)


def test_build_qq_intel_control_request_platform_spec_uses_overrides():
    matcher = lambda **kwargs: ({"action": "noop"}, None)
    worker_name_extractor = lambda body, known_worker_names: "钢镚"
    worker_context_checker = lambda body: True
    target_extractor = lambda source, body: "group:726109087"
    hire_objective_extractor = lambda body, *, worker_name, target_group: "刺探情报"

    spec = build_qq_intel_control_request_platform_spec(
        request_matcher=matcher,
        worker_name_extractor=worker_name_extractor,
        worker_context_checker=worker_context_checker,
        target_extractor=target_extractor,
        hire_objective_extractor=hire_objective_extractor,
    )

    assert spec.platform is Platform.QQ_NAPCAT
    assert spec.request_matcher is matcher
    assert spec.worker_name_extractor is worker_name_extractor
    assert spec.worker_context_checker is worker_context_checker
    assert spec.target_extractor is target_extractor
    assert spec.hire_objective_extractor is hire_objective_extractor


def test_default_qq_intel_control_request_platform_spec_is_qq():
    assert QQ_INTEL_CONTROL_REQUEST_PLATFORM_SPEC.platform is Platform.QQ_NAPCAT
