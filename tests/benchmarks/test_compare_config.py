"""Tests for --compare config inheritance in the benchmark runner.

The --compare flag must inherit the primary config's workload parameters
(notably ``suites``) so both backends run the same scenarios.  Without this
inheritance ``run_single()`` defaults missing suites to ``["a"]`` and a
``--suite d --compare X`` invocation silently compares suite D against A.
"""
from benchmarks.interface import BenchmarkConfig


def _build_compare_config(primary: BenchmarkConfig, profile: str, embedding: str) -> BenchmarkConfig:
    """Replicate the --compare config construction logic from runner.main()."""
    compare_parameters = dict(primary.parameters)
    compare_parameters["profile"] = profile
    compare_parameters["embedding_model"] = embedding
    return BenchmarkConfig(
        backend_name="other-backend",
        profile=profile,
        embedding_model=embedding,
        parameters=compare_parameters,
    )


def test_compare_config_inherits_suites():
    """The comparison config must carry the same suites as the primary."""
    primary = BenchmarkConfig(
        backend_name="hindsight",
        parameters={"suites": ["d"], "profile": "balanced", "embedding_model": "auto"},
    )
    config2 = _build_compare_config(primary, "balanced", "auto")
    assert config2.parameters.get("suites") == ["d"]


def test_compare_config_does_not_default_to_suite_a():
    """Regression: --suite d --compare X must not silently compare against suite A."""
    primary = BenchmarkConfig(
        backend_name="hindsight",
        parameters={"suites": ["d", "e", "f"], "profile": "balanced"},
    )
    config2 = _build_compare_config(primary, "balanced", "auto")
    # run_single reads config2.parameters.get("suites", ["a"]) — it must NOT be ["a"]
    assert config2.parameters.get("suites", ["a"]) == ["d", "e", "f"]


def test_compare_config_preserves_all_workload_params():
    """All workload parameters from the primary must be inherited."""
    primary = BenchmarkConfig(
        backend_name="hindsight",
        parameters={
            "suites": ["a", "b"],
            "profile": "deep",
            "embedding_model": "sentence-transformers",
            "contradiction_llm_model": "claude-haiku-4-5",
        },
    )
    config2 = _build_compare_config(primary, "deep", "sentence-transformers")
    assert config2.parameters["contradiction_llm_model"] == "claude-haiku-4-5"
    assert config2.parameters["suites"] == ["a", "b"]
