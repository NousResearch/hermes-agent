from gateway.config import Platform
from gateway.intel_worker_platform_specs import (
    IntelWorkerPlatformSpec,
    QQ_INTEL_WORKER_PLATFORM_SPEC,
    build_qq_intel_worker_platform_spec,
    load_known_intel_worker_names,
)


def test_build_qq_intel_worker_platform_spec_uses_override():
    list_workers = lambda: [{"worker_name": "钢镚"}]

    spec = build_qq_intel_worker_platform_spec(list_workers_fn=list_workers)

    assert spec.platform is Platform.QQ_NAPCAT
    assert spec.list_workers is list_workers


def test_load_known_intel_worker_names_normalizes_records():
    spec = IntelWorkerPlatformSpec(
        platform=Platform.QQ_NAPCAT,
        list_workers=lambda: [
            {"worker_name": "钢镚"},
            {"worker_name": "  钢镚  "},
            {"worker_name": ""},
            {},
            "invalid",
        ],
    )

    assert load_known_intel_worker_names(spec) == {"钢镚"}


def test_default_qq_intel_worker_platform_spec_is_qq():
    assert QQ_INTEL_WORKER_PLATFORM_SPEC.platform is Platform.QQ_NAPCAT
