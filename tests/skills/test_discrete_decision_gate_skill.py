"""Contract checks for the optional discrete-decision-gate skill.

Reads the SKILL.md asset and exercises the shipped script offline: every
network call is mocked at ``urllib.request.urlopen``, so no key and no
network access are required.
"""
import io
import re
import urllib.error
import pytest
import yaml
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "optional-skills" / "autonomous-ai-agents" / "discrete-decision-gate"
SKILL_PATH = SKILL_DIR / "SKILL.md"
SCRIPT_PATH = SKILL_DIR / "scripts" / "decision_gate.py"
RUNNER_PATH = SKILL_DIR / "scripts" / "gated_harness_loop.py"
JEV_PATH = SKILL_DIR / "scripts" / "jev_gate.py"
REFERENCE_PATH = SKILL_DIR / "references" / "jev-api.md"

REQUIRED_SECTIONS = [
    "## When to Use",
    "## Prerequisites",
    "## How to Run",
    "## Quick Reference",
    "## Procedure",
    "## Pitfalls",
    "## Verification",
]

MACHINE_LOCAL = re.compile(r"/home/[a-z0-9_-]+/|/appdata/|/root/|[A-Z]:\\\\Users\\\\")


# --------------------------------------------------------------------- loading

def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---"), "SKILL.md must open with frontmatter"
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter must close with ---"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    assert isinstance(fm, dict), "frontmatter must be a YAML mapping"
    return fm, content[m.end() + 3 :]


def _load_module(name, path):
    """Import a skill script by path (the scripts are not package modules)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_script():
    """Import scripts/decision_gate.py by path."""
    return _load_module("decision_gate", SCRIPT_PATH)


def _load_runner():
    """Import scripts/gated_harness_loop.py by path (it imports decision_gate as a sibling)."""
    return _load_module("gated_harness_loop", RUNNER_PATH)


def _load_jev():
    """Import scripts/jev_gate.py by path."""
    return _load_module("jev_gate", JEV_PATH)


def _decisions(answers: dict, model: str = "typesafe/jev-1.13-20260917") -> "_Response":
    """A well-formed Decisions-route response body."""
    import json

    return _Response(json.dumps({
        "model": model, "answers": answers,
        "usage": {"input_tokens": 312, "output_tokens": 35, "cost": 1.5e-05},
    }).encode())


def _no_keys(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def _openrouter(monkeypatch):
    _no_keys(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


class _Response:
    """Minimal stand-in for the object urlopen yields as a context manager."""

    def __init__(self, payload: bytes, status: int = 200):
        self._payload = payload
        self.status = status

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _chat_completion(content: str) -> _Response:
    import json

    return _Response(json.dumps({"choices": [{"message": {"content": content}}]}).encode())


# ------------------------------------------------------------------ file assets

def test_assets_exist():
    assert SKILL_PATH.is_file()
    assert SCRIPT_PATH.is_file()
    assert RUNNER_PATH.is_file()
    assert JEV_PATH.is_file()
    assert REFERENCE_PATH.is_file()


def test_frontmatter_required_fields():
    fm, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "discrete-decision-gate"
    assert fm["name"] == SKILL_DIR.name
    assert fm["version"] == "0.1.0"
    assert fm["author"].startswith("Ron Malouin (rpmalouin)"), "credit the human first"
    assert fm["platforms"] == ["linux", "macos", "windows"]
    tags = (fm.get("metadata") or {}).get("hermes", {}).get("tags")
    assert tags, "metadata.hermes.tags must be present"


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars; hardline is 60"
    assert desc.endswith("."), "description must end with a period"
    assert desc.count(".") == 1, "description must be one sentence"


def test_related_skills_resolve_in_repo():
    fm, _ = _frontmatter_and_body()
    related = (fm.get("metadata") or {}).get("hermes", {}).get("related_skills") or []
    assert related, "related_skills must not be empty"
    in_repo = {
        p.parent.name
        for p in list(REPO_ROOT.glob("skills/**/SKILL.md"))
        + list(REPO_ROOT.glob("optional-skills/**/SKILL.md"))
    }
    dangling = [r for r in related if r not in in_repo]
    assert not dangling, f"related_skills not in repo: {dangling}"


def test_body_sections_present():
    _, body = _frontmatter_and_body()
    missing = [s for s in REQUIRED_SECTIONS if s not in body]
    assert not missing, f"missing sections: {missing}"


def test_no_machine_local_paths():
    _, content = _frontmatter_and_body()
    for path in (SKILL_PATH, REFERENCE_PATH, RUNNER_PATH, SCRIPT_PATH, JEV_PATH):
        text = path.read_text(encoding="utf-8")
        m = MACHINE_LOCAL.search(text)
        assert not m, f"{path.name}: machine-local path {m.group(0)!r}"


def test_script_compiles():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    compile(source, str(SCRIPT_PATH), "exec")


# --------------------------------------------------------------- the gate itself

def test_parse_choice_accepts_exact_single_label():
    gate = _load_script()
    assert gate._parse_choice("high", ["low", "high"]) == "high"
    assert gate._parse_choice("  HIGH  ", ["low", "high"]) == "high"
    assert gate._parse_choice("crg_first", gate.PRESETS["triage"]["choices"]) == "crg_first"


def test_parse_choice_rejects_ambiguous_and_unrelated_text():
    gate = _load_script()
    assert gate._parse_choice("low or high, it depends", ["low", "high"]) is None
    assert gate._parse_choice("complete and also retry", ["complete", "retry", "abort"]) is None
    assert gate._parse_choice("banana", ["low", "high"]) is None
    assert gate._parse_choice("", ["low", "high"]) is None
    assert gate._parse_choice(None, ["low", "high"]) is None


def test_preset_failure_defaults_are_staked():
    gate = _load_script()
    assert gate.PRESETS["triage"]["unavailable_default"] == "bash_direct"
    assert gate.PRESETS["blast-radius"]["unavailable_default"] == "high"
    assert gate.PRESETS["harness"]["unavailable_default"] == "abort"
    # a silent gate failure must never auto-apply a change or end a loop
    for name, preset in gate.PRESETS.items():
        assert preset["unavailable_default"] != "low", name
        assert preset["unavailable_default"] != "complete", name


def test_decide_returns_label_on_well_formed_answer(monkeypatch):
    gate = _load_script()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with mock.patch.object(
        gate.urllib.request, "urlopen", lambda req, timeout: _chat_completion("high")
    ):
        result = gate.decide(
            "touched models/auth.py with 31 dependents",
            "How risky is this change for auto-apply?",
            ["low", "high"],
            backend="openrouter",
        )
    assert result["status"] == "ok"
    assert result["choice"] == "high"
    assert result["backend"] == "openrouter"


def test_decide_is_unavailable_when_backend_errors(monkeypatch):
    """A dead backend must never yield a fabricated label."""
    gate = _load_script()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def boom(req, timeout):
        raise urllib.error.URLError("connection refused")

    with mock.patch.object(gate.urllib.request, "urlopen", boom):
        result = gate.decide(
            "state", "question", ["low", "high"],
            backend="openrouter", unavailable_default="high",
        )
    assert result["status"] == "unavailable"
    assert "choice" not in result, "a failed gate must not carry a choice"
    assert result["backend"] == "openrouter"
    assert "URLError" in result["reason"]
    assert result["fallback"] == "high"
    assert "not a model verdict" in result["policy"]


def test_decide_reports_http_error_as_unavailable(monkeypatch):
    """The measured ``typesafe/jev`` + /chat/completions 400 is an outage, not a label."""
    gate = _load_script()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def bad_model(req, timeout):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {}, io.BytesIO(b"is not a valid model ID")
        )

    with mock.patch.object(gate.urllib.request, "urlopen", bad_model):
        result = gate.decide(
            "state", "question", ["low", "high"],
            backend="openrouter", model="typesafe/jev",
        )
    assert result["status"] == "unavailable"
    assert "not a valid model ID" in result["reason"]
    assert "choice" not in result


def test_decide_rejects_off_schema_answer(monkeypatch):
    gate = _load_script()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with mock.patch.object(
        gate.urllib.request,
        "urlopen",
        lambda req, timeout: _chat_completion("I think it is probably fine"),
    ):
        result = gate.decide("state", "question", ["low", "high"], backend="openrouter")
    assert result["status"] == "unavailable"
    assert "choice" not in result
    assert "off-schema" in result["reason"]


def test_missing_credential_is_unavailable_without_a_call(monkeypatch):
    gate = _load_script()
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with mock.patch.object(
        gate.urllib.request, "urlopen", side_effect=AssertionError("must not be called")
    ):
        result = gate.decide("state", "question", ["low", "high"], backend="typesafe")
    assert result["status"] == "unavailable"
    assert result["reason"] == "TYPESAFE_API_KEY not set"


def test_emit_exit_code_three_on_unavailable(capsys):
    gate = _load_script()
    rc = gate._emit(
        {"status": "unavailable", "backend": "typesafe",
         "reason": "TYPESAFE_API_KEY not set", "fallback": "high"},
        as_json=False,
        quiet=False,
    )
    out = capsys.readouterr().out
    assert rc == 3
    assert "UNAVAILABLE" in out
    assert "deterministic default 'high'" in out


# ---------------------------------------------------------------- backend preflight

def test_doctor_reports_unconfigured_without_a_key(monkeypatch):
    gate = _load_script()
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(gate, "_env_file_key", lambda name: None)
    with mock.patch.object(
        gate.urllib.request, "urlopen", side_effect=AssertionError("must not be called")
    ):
        report = gate.doctor(backend="typesafe")
    assert report["ready"] is False
    assert report["ok"] == []
    entry = report["backends"]["typesafe"]
    assert entry["key"] == "MISSING"
    assert entry["status"] == "unconfigured"


def test_doctor_reports_ready_when_a_backend_answers(monkeypatch):
    gate = _load_script()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    with mock.patch.object(
        gate.urllib.request, "urlopen", lambda req, timeout: _chat_completion("pass")
    ):
        report = gate.doctor(backend="openrouter")
    assert report["ready"] is True
    assert report["ok"] == ["openrouter"]
    assert report["backends"]["openrouter"]["choice"] == "pass"


# ------------------------------------------------------------- gated harness loop

def _run(runner, tmp_path, *extra, monkeypatch, run_result, verdict, harness_calls=None):
    """Invoke the loop with a stubbed harness and a stubbed gate."""
    monkeypatch.setattr(
        runner, "run_harness",
        lambda *a, **k: (harness_calls.append(1) if harness_calls is not None else None)
        or run_result,
    )
    monkeypatch.setattr(runner, "decide", lambda *a, **k: verdict)
    return runner.main(
        ["--task", "make the check pass", "--cmd", "true {prompt}",
         "--workdir", str(tmp_path), "--log", str(tmp_path / "loop.json"), *extra]
    )


def test_runner_requires_task_and_cmd():
    runner = _load_runner()
    with pytest.raises(SystemExit):
        runner.main([])
    with pytest.raises(SystemExit):
        runner.main(["--task", "t", "--cmd", "echo no placeholder here"])


def test_runner_never_declares_success_when_the_gate_is_unavailable(monkeypatch, tmp_path):
    runner = _load_runner()
    rc = _run(
        runner, tmp_path, monkeypatch=monkeypatch,
        run_result={"exit_code": 0, "tail": "all good", "duration_ms": 1},
        verdict={"status": "unavailable", "backend": None, "reason": "no backend configured",
                 "fallback": "abort"},
    )
    assert rc == 3, "an exit 0 harness run must not be reported as complete without a verdict"


def test_runner_completes_on_a_complete_verdict(monkeypatch, tmp_path):
    runner = _load_runner()
    rc = _run(
        runner, tmp_path, monkeypatch=monkeypatch,
        run_result={"exit_code": 0, "tail": "12 passed", "duration_ms": 1},
        verdict={"status": "ok", "choice": "complete", "backend": "openrouter"},
    )
    assert rc == 0
    assert (tmp_path / "loop.json").is_file()


def test_runner_loops_on_retry_then_reports_exhaustion(monkeypatch, tmp_path):
    runner = _load_runner()
    calls: list[int] = []
    rc = _run(
        runner, tmp_path, "--max-loops", "3", monkeypatch=monkeypatch,
        run_result={"exit_code": 1, "tail": "FAILED tests/test_cart.py::test_total",
                    "duration_ms": 1},
        verdict={"status": "ok", "choice": "retry", "backend": "openrouter"},
        harness_calls=calls,
    )
    assert rc == 2
    assert len(calls) == 3, "max-loops must bound the retry loop"
    import json

    assert len(json.loads((tmp_path / "loop.json").read_text())["attempts"]) == 3


def test_runner_aborts_on_an_abort_verdict(monkeypatch, tmp_path):
    runner = _load_runner()
    calls: list[int] = []
    rc = _run(
        runner, tmp_path, monkeypatch=monkeypatch,
        run_result={"exit_code": 128,
                    "tail": "fatal: could not read Username for 'https://github.com'",
                    "duration_ms": 1},
        verdict={"status": "ok", "choice": "abort", "backend": "openrouter"},
        harness_calls=calls,
    )
    assert rc == 3
    assert len(calls) == 1, "abort must not run another attempt"


def test_runner_exits_four_when_the_harness_is_missing(monkeypatch, tmp_path):
    """A missing harness binary is a launch failure, not something to ask the gate about."""
    runner = _load_runner()
    asked: list[int] = []
    monkeypatch.setattr(
        runner, "run_harness",
        lambda *a, **k: {"exit_code": 127, "tail": "sh: 1: nonexistent-harness: not found",
                         "duration_ms": 1},
    )
    monkeypatch.setattr(runner, "decide", lambda *a, **k: asked.append(1) or {"status": "ok"})
    rc = runner.main(
        ["--task", "t", "--cmd", "nonexistent-harness {prompt}",
         "--workdir", str(tmp_path), "--log", str(tmp_path / "loop.json")]
    )
    assert rc == 4
    assert not asked, "the gate must not be consulted for a launch failure"


# --------------------------------------------------- jev_gate (protocol shape)

REPO_INTENT = "fix the failing cart test"
SAMPLE_DIFF = ("diff --git a/cart.py b/cart.py\n--- a/cart.py\n+++ b/cart.py\n"
               "@@ -1 +1 @@\n-    return sum(items)\n+    return sum(items or [])\n")


def test_jev_script_compiles():
    compile(JEV_PATH.read_text(encoding="utf-8"), str(JEV_PATH), "exec")


def test_jev_targets_the_alpha_decisions_route_with_a_flat_body(monkeypatch):
    """The measured 404/400 traps: right route, no decisionsRequest wrapper."""
    import json

    jev = _load_jev()
    _openrouter(monkeypatch)
    seen: dict = {}

    def spy(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode())
        return _decisions({"risk": {"type": "choice", "choice": "low", "confidence": 1.0,
                                    "probabilities": {"low": 1.0, "high": 0.0}}})

    with mock.patch.object(jev.urllib.request, "urlopen", spy):
        result = jev.evaluate_blast_radius("1 file, leaf module, tests only")

    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert "decisionsRequest" not in seen["body"]
    assert seen["body"]["model"] == "typesafe/jev-1.13"
    assert seen["body"]["questions"]["risk"]["type"] == "choice"
    assert result["label"] == "low"
    assert result["backend"] == "openrouter"


def test_jev_defaults_to_openrouter_and_ignores_a_stray_typesafe_key(monkeypatch):
    """A free/rate-limited 1P key must not silently take over the gate."""
    jev = _load_jev()
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-stray-1p")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen: dict = {}

    def spy(req, timeout=None):
        seen["url"] = req.full_url
        return _decisions({"risk": {"type": "choice", "choice": "low", "confidence": 1.0}})

    with mock.patch.object(jev.urllib.request, "urlopen", spy):
        result = jev.evaluate_blast_radius("1 file, leaf module, tests only")
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert result["backend"] == "openrouter"
    assert result["label"] == "low"


def test_jev_pinned_typesafe_uses_the_systemone_route(monkeypatch):
    jev = _load_jev()
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-1p")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    seen: dict = {}

    def spy(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        raise OSError("stop here - we only inspect the request")

    with mock.patch.object(jev.urllib.request, "urlopen", spy):
        result = jev.evaluate_blast_radius("auth models, 31 dependents", backend="typesafe")

    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["auth"] == "Bearer sk-1p"
    assert result["label"] == "high", "transport failure must fail closed"


def test_jev_pinned_backend_without_its_key_names_the_pin(monkeypatch):
    jev = _load_jev()
    _no_keys(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")  # present, but the caller pinned 1P
    with mock.patch.object(
        jev.urllib.request, "urlopen", side_effect=AssertionError("must not be called")
    ):
        result = jev.evaluate_blast_radius("auth models, 31 dependents", backend="typesafe")
    assert result["label"] == "high"
    assert result["error_kind"] == "no_key"
    assert result["reason"] == "TYPESAFE_API_KEY not set (backend pinned to typesafe)"


def test_jev_no_keys_returns_each_gates_fail_direction_without_a_call(monkeypatch):
    jev = _load_jev()
    _no_keys(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen", side_effect=AssertionError("must not be called")
    ):
        triage = jev.triage_intent("make the health check endpoint more reliable")
        blast = jev.evaluate_blast_radius("auth models, 31 dependents")
        diff = jev.verify_diff_matches_intent(REPO_INTENT, SAMPLE_DIFF)
        harness = jev.evaluate_harness_output(0, "12 passed", 1, 3)
    for result, expected in ((triage, "bash_direct"), (blast, "high"),
                             (diff, "mismatch"), (harness, "abort")):
        assert result["label"] == expected, result
        assert result["status"] == "unavailable"
        assert result["error_kind"] == "no_key"
        assert "choice" not in result and "label" in result


def test_jev_rule_tiers_never_spend_a_call(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen", side_effect=AssertionError("rules must not call out")
    ):
        assert jev.triage_intent("git status")["label"] == "bash_direct"
        assert jev.triage_intent("ls -la /var/log")["label"] == "bash_direct"
        assert jev.triage_intent("write a standalone log parser script")["label"] == "direct_dsh"
        assert jev.triage_intent("refactor auth across 14 modules")["label"] == "crg_first"
        # any-order failure language, the case that slipped through the first rule set
        assert jev.triage_intent("the cart test has been failing since tuesday")["label"] == "crg_first"


def test_jev_noul_probability_drives_match_and_mismatch(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    for probability, expected in ((0.96, "match"), (0.70, "match"),
                                  (0.69, "mismatch"), (0.02, "mismatch")):
        with mock.patch.object(
            jev.urllib.request, "urlopen",
            lambda req, timeout=None, p=probability: _decisions(
                {"matches": {"type": "noul", "noul": p}}),
        ):
            result = jev.verify_diff_matches_intent(REPO_INTENT, SAMPLE_DIFF)
        assert result["label"] == expected, (probability, result)
        assert result["probability"] == probability


def test_jev_noul_reading_probabilities_true_is_a_schema_error(monkeypatch):
    """The exact bug this gate replaces: a noul answer has no probabilities.true."""
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen",
        lambda req, timeout=None: _decisions(
            {"matches": {"type": "noul", "probabilities": {"true": 0.99}}}),
    ):
        result = jev.verify_diff_matches_intent(REPO_INTENT, SAMPLE_DIFF)
    assert result["label"] == "mismatch", "fail closed"
    assert result["status"] == "unavailable"
    assert result["error_kind"] == "schema"


def test_jev_confidence_floor_and_missing_confidence_differ(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen",
        lambda req, timeout=None: _decisions({"risk": {"type": "choice", "choice": "low",
                                                      "confidence": 0.30}}),
    ):
        floored = jev.evaluate_blast_radius("1 file, leaf module")
    assert floored["label"] == "high", "below the floor the fail direction wins"
    assert floored["status"] == "ok"
    assert floored["low_confidence"] is True

    with mock.patch.object(
        jev.urllib.request, "urlopen",
        lambda req, timeout=None: _decisions({"risk": {"type": "choice", "choice": "low"}}),
    ):
        drifted = jev.evaluate_blast_radius("1 file, leaf module")
    assert drifted["label"] == "high"
    assert drifted["status"] == "unavailable"
    assert drifted["error_kind"] == "schema", "missing confidence is schema drift, not a low score"


def test_jev_off_schema_choice_fails_closed(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen",
        lambda req, timeout=None: _decisions({"risk": {"type": "choice", "choice": "medium",
                                                       "confidence": 0.99}}),
    ):
        result = jev.evaluate_blast_radius("1 file, leaf module")
    assert result["label"] == "high"
    assert result["error_kind"] == "schema"


def test_jev_empty_inputs_fail_closed_without_a_call(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen", side_effect=AssertionError("must not be called")
    ):
        assert jev.evaluate_blast_radius("")["label"] == "high"
        assert jev.verify_diff_matches_intent(REPO_INTENT, "")["label"] == "mismatch"


def test_jev_environment_failures_abort_without_a_call(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen", side_effect=AssertionError("must not be called")
    ):
        for tail in ("fatal: could not read Username for 'https://github.com'",
                     "Error: invalid_api_key", "missing_credential: no key for provider"):
            result = jev.evaluate_harness_output(1, tail, 1, 3)
            assert result["label"] == "abort", tail
            assert result["error_kind"] == "environment", tail


def test_jev_benign_permission_error_is_not_an_environment_abort(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen",
        lambda req, timeout=None: _decisions(
            {"verdict": {"type": "choice", "choice": "retry", "confidence": 0.8}}),
    ):
        result = jev.evaluate_harness_output(
            1, "FAILED tests/test_fs.py::test_ro - PermissionError: [Errno 13] Permission denied",
            1, 3)
    assert result["label"] == "retry"


def test_jev_masked_failure_detection_is_case_insensitive():
    jev = _load_jev()
    assert jev._looks_failed("collected 9 items\n8 passed\n1 failed in 2.0s")
    assert jev._looks_failed("Traceback (most recent call last):")
    assert jev._looks_failed("npm ERR! code ELIFECYCLE")
    assert not jev._looks_failed("12 passed in 3.41s")
    assert not jev._looks_failed("")


def test_jev_artifact_check_beats_a_clean_exit(monkeypatch, tmp_path):
    jev = _load_jev()
    _openrouter(monkeypatch)
    missing = tmp_path / "report.json"
    with mock.patch.object(
        jev.urllib.request, "urlopen", side_effect=AssertionError("no verdict needed")
    ):
        absent = jev.evaluate_harness_output(0, "12 passed", 1, 3, [str(missing)])
    assert absent["label"] == "retry"
    assert absent["error_kind"] == "artifact_missing"

    missing.write_text("12 passed\n")
    with mock.patch.object(
        jev.urllib.request, "urlopen",
        lambda req, timeout=None: _decisions(
            {"verdict": {"type": "choice", "choice": "complete", "confidence": 0.97}}),
    ):
        present = jev.evaluate_harness_output(0, "12 passed", 1, 3, [str(missing)])
    assert present["label"] == "complete"


def test_jev_retry_ceiling_aborts_without_a_call(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen", side_effect=AssertionError("must not be called")
    ):
        result = jev.evaluate_harness_output(1, "FAILED test_x - TypeError", 3, 3)
    assert result["label"] == "abort"
    assert result["error_kind"] == "max_attempts"


def test_jev_refuses_complete_on_a_non_zero_exit_code(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen",
        lambda req, timeout=None: _decisions(
            {"verdict": {"type": "choice", "choice": "complete", "confidence": 0.99}}),
    ):
        result = jev.evaluate_harness_output(1, "FAILED test_x - TypeError", 1, 3)
    assert result["label"] == "abort"


def test_jev_http_and_transport_failures_are_distinguishable(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)

    def http_404(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {},
                                     io.BytesIO(b'{"error":{"message":"Not Found"}}'))

    with mock.patch.object(jev.urllib.request, "urlopen", http_404):
        result = jev.evaluate_blast_radius("1 file, leaf module")
    assert result["label"] == "high" and result["error_kind"] == "http"
    assert "404" in result["reason"]

    def transport(req, timeout=None):
        raise OSError("connection refused")

    with mock.patch.object(jev.urllib.request, "urlopen", transport):
        result = jev.evaluate_blast_radius("1 file, leaf module")
    assert result["label"] == "high" and result["error_kind"] == "transport"


def test_jev_profile_is_always_reported(monkeypatch):
    """Callers need to know which policy applied, even on a successful gate."""
    jev = _load_jev()
    _openrouter(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen",
        lambda req, timeout=None: _decisions({"risk": {"type": "choice", "choice": "low",
                                                      "confidence": 0.9,
                                                      "probabilities": {"low": 0.9}}}),
    ):
        ok = jev.evaluate_blast_radius("1 file, leaf module")
        no_key_expected = jev.FAIL_DIRECTION
    assert ok["fail_direction"] == no_key_expected["blast"] == "high"
    assert ok["backend"] == "openrouter"
    assert ok["cost"] == 1.5e-05


def test_jev_latest_aliases_resolve_to_the_pinned_id(monkeypatch):
    """OpenRouter has no typesafe/jev-latest, so the habitual spelling must map onto the pin."""
    import json

    jev = _load_jev()
    _openrouter(monkeypatch)
    for alias in ("latest", "jev-latest", "typesafe/jev-latest", "typesafe/jev"):
        seen: dict = {}

        def spy(req, timeout=None):
            seen["body"] = json.loads(req.data.decode())
            return _decisions({"risk": {"type": "choice", "choice": "low", "confidence": 1.0}})

        with mock.patch.object(jev.urllib.request, "urlopen", spy):
            result = jev.evaluate_blast_radius("1 file, leaf module", model=alias)
        assert seen["body"]["model"] == "typesafe/jev-1.13", alias
        assert result["label"] == "low", alias


def test_jev_doctor_reports_the_advertised_snapshot_and_a_live_probe(monkeypatch):
    jev = _load_jev()
    _openrouter(monkeypatch)
    endpoints_payload = {
        "data": {
            "id": "typesafe/jev-1.13",
            "name": "TypeSafe: Jev 1.13",
            "created": 1789689684,
            "architecture": {"modality": "text->decisions"},
            "endpoints": [{
                "name": "TypeSafe | typesafe/jev-1.13-20260917",
                "pricing": {"prompt": "0.000000042", "completion": "0"},
                "context_length": 32000,
                "uptime_last_1d": 100,
            }],
        }
    }

    def router(req, timeout=None):
        if req.full_url.endswith("/endpoints"):
            import json

            return _Response(json.dumps(endpoints_payload).encode())
        return _decisions({"ok": {"type": "choice", "choice": "yes", "confidence": 1.0}})

    with mock.patch.object(jev.urllib.request, "urlopen", router):
        report = jev.doctor()

    assert report["status"] == "ok"
    assert report["backend"] == "openrouter"
    assert report["advertised"]["provider_endpoint"] == "TypeSafe | typesafe/jev-1.13-20260917"
    assert report["advertised"]["prompt_price_per_token"] == "0.000000042"
    assert report["advertised"]["modality"] == "text->decisions"
    assert report["probe"]["served_model"] == "typesafe/jev-1.13-20260917"
    assert report["probe"]["choice"] == "yes"


def test_jev_doctor_without_keys_reports_no_keys(monkeypatch):
    jev = _load_jev()
    _no_keys(monkeypatch)
    with mock.patch.object(
        jev.urllib.request, "urlopen", side_effect=AssertionError("must not be called")
    ):
        report = jev.doctor()
    assert report["status"] == "no_keys"
    assert report["keys"] == {"TYPESAFE_API_KEY": False, "OPENROUTER_API_KEY": False}


# ------------------------------------------- decision_gate: the jev backend in auto order

def test_decision_gate_prefers_jev_over_the_chat_backend(monkeypatch):
    """`--backend auto` must reach Jev on the decisions route before any chat model."""
    gate = _load_script()
    _openrouter(monkeypatch)
    seen: dict = {}

    def spy(req, timeout=None):
        seen["url"] = req.full_url
        import json

        seen["body"] = json.loads(req.data.decode())
        return _decisions({"answer": {"type": "choice", "choice": "high",
                                      "probabilities": {"low": 0.1, "high": 0.9},
                                      "confidence": 0.9}})

    with mock.patch.object(gate.urllib.request, "urlopen", spy):
        result = gate.decide(
            "touched models/auth.py, 31 dependents",
            gate.PRESETS["blast-radius"]["question"],
            gate.PRESETS["blast-radius"]["choices"],
            gate.PRESETS["blast-radius"]["describe"],
        )
    assert gate.AUTO_ORDER.index("jev") < gate.AUTO_ORDER.index("openrouter")
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["body"]["questions"]["answer"]["type"] == "choice"
    assert result["backend"] == "jev"
    assert result["choice"] == "high"
    assert result["confidence"] == 0.9


def test_decision_gate_jev_backend_rejects_an_off_schema_choice(monkeypatch):
    gate = _load_script()
    _openrouter(monkeypatch)
    with mock.patch.object(
        gate.urllib.request, "urlopen",
        lambda req, timeout=None: _decisions({"answer": {"type": "choice", "choice": "medium",
                                                        "confidence": 0.99}}),
    ):
        result = gate.decide("state", "question", ["low", "high"], backend="jev")
    assert result["status"] == "unavailable"
    assert "off-schema" in result["reason"]
    assert "choice" not in result


def test_decision_gate_jev_backend_requires_the_openrouter_key(monkeypatch):
    gate = _load_script()
    _no_keys(monkeypatch)
    with mock.patch.object(
        gate.urllib.request, "urlopen", side_effect=AssertionError("must not be called")
    ):
        result = gate.decide("state", "question", ["low", "high"], backend="jev")
    assert result["status"] == "unavailable"
    assert result["reason"] == "OPENROUTER_API_KEY not set"
