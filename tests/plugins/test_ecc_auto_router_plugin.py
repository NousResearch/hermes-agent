from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[2] / "plugins" / "ecc-auto-router" / "__init__.py"
    spec = spec_from_file_location("ecc_auto_router_plugin", path)
    mod = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_routes_python_plus_tdd():
    mod = _load_module()
    skills = mod._route_skills("Fix this Django auth bug and add tests")
    assert skills[0] == "ecc-python:python-patterns"
    assert "ecc-core:tdd-workflow" in skills


def test_routes_security_only():
    mod = _load_module()
    skills = mod._route_skills("Perform a security audit for XSS and CSRF")
    assert skills == ["ecc-security:security-review"]


def test_pre_llm_call_returns_context_block():
    mod = _load_module()
    out = mod.on_pre_llm_call(user_message="Deploy docker pipeline")
    assert isinstance(out, dict)
    assert "context" in out
    assert "[ECC Auto-Router]" in out["context"]
