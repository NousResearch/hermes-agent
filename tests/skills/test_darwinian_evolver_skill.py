"""
Smoke tests for the darwinian-evolver optional skill.

We can't actually run the evolution loop in CI (it needs network + a paid LLM),
so these tests verify:
  - SKILL.md frontmatter conforms to the hardline format
  - shipped scripts parse as valid Python
  - the scripts reference the right env var / module paths
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "research" / "darwinian-evolver"


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_name_matches_dir(frontmatter) -> None:
    assert frontmatter["name"] == "darwinian-evolver"


def test_platforms_excludes_windows(frontmatter) -> None:
    # Upstream uses func_timeout (POSIX signals) and uv subprocess pipelines; the
    # skill is gated [linux, macos]. If we ever port to Windows, update this test
    # to assert ["linux", "macos", "windows"].
    assert "windows" not in frontmatter["platforms"]
    assert set(frontmatter["platforms"]) >= {"linux", "macos"}


def test_author_credits_contributor(frontmatter) -> None:
    author = frontmatter["author"]
    assert "Bihruze" in author, f"author should credit the original contributor: {author!r}"


def test_license_mit(frontmatter) -> None:
    assert frontmatter["license"] == "MIT"


@pytest.mark.parametrize(
    "path",
    [
        "scripts/parrot_openrouter.py",
        "scripts/show_snapshot.py",
        "templates/custom_problem_template.py",
    ],
)
def test_shipped_scripts_parse(path: str) -> None:
    src = (SKILL_DIR / path).read_text()
    ast.parse(src)  # raises SyntaxError on broken Python


def test_parrot_script_uses_openrouter() -> None:
    src = (SKILL_DIR / "scripts" / "parrot_openrouter.py").read_text()
    assert "OPENROUTER_API_KEY" in src, "parrot driver should read OPENROUTER_API_KEY"
    assert "openrouter.ai/api/v1" in src, "parrot driver should target OpenRouter"
    assert "EVOLVER_MODEL" in src, "model should be overridable via EVOLVER_MODEL"


def test_parrot_script_has_error_swallowing() -> None:
    """Provider content-filter / rate-limit must not kill the run — see Pitfall 2."""
    src = (SKILL_DIR / "scripts" / "parrot_openrouter.py").read_text()
    assert "LLM_ERROR" in src, "_prompt_llm should swallow provider errors and tag them"


def test_skill_calls_out_agpl(frontmatter) -> None:
    """The upstream tool is AGPL-3.0. The skill MUST flag this so users don't
    import it into MIT-licensed code by accident."""
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "AGPL" in src, "SKILL.md must mention upstream AGPL license"


def test_skill_pitfalls_section_present() -> None:
    src = (SKILL_DIR / "SKILL.md").read_text()
    assert "## Pitfalls" in src
    # Pitfalls we discovered during the spike — keep them in sync with reality.
    assert "Initial organism must be viable" in src
    assert "generator" in src  # loop.run() pitfall


# ---------------------------------------------------------------------------
# SSTI hardening: model-proposed prompt templates must be rendered sandboxed
# ---------------------------------------------------------------------------


def _load_parrot_module():
    """Import parrot_openrouter.py with its heavy/optional deps stubbed.

    The script imports ``openai`` and the upstream (AGPL, installed at runtime
    via ``uv --project``) ``darwinian_evolver`` package — neither is available
    in the hermetic test env. We register minimal stand-ins in ``sys.modules``
    so the *real* ``ParrotOrganism`` / ``ImproveParrotMutator`` classes load and
    we can exercise the actual Jinja render path.
    """

    class _Base:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def __class_getitem__(cls, item):  # support Generic[...] subscription
            return cls

    def _mod(name):
        m = types.ModuleType(name)
        sys.modules[name] = m
        return m

    openai = _mod("openai")

    class _OpenAI:
        def __init__(self, *a, **k):
            ...

    openai.OpenAI = _OpenAI

    _mod("darwinian_evolver")
    cc = _mod("darwinian_evolver.cli_common")
    cc.build_hyperparameter_config_from_args = lambda *a, **k: None
    cc.register_hyperparameter_args = lambda *a, **k: None
    cc.parse_learning_log_view_type = lambda *a, **k: None
    _mod("darwinian_evolver.evolve_problem_loop").EvolveProblemLoop = _Base
    _mod("darwinian_evolver.learning_log").LearningLogEntry = _Base
    prob = _mod("darwinian_evolver.problem")
    for _n in (
        "EvaluationFailureCase",
        "EvaluationResult",
        "Evaluator",
        "Mutator",
        "Organism",
        "Problem",
    ):
        setattr(prob, _n, _Base)

    path = SKILL_DIR / "scripts" / "parrot_openrouter.py"
    spec = importlib.util.spec_from_file_location("parrot_openrouter_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def parrot(monkeypatch):
    mod = _load_parrot_module()
    # Never hit the network: make the LLM call echo its rendered prompt so the
    # tests can observe exactly what the template produced.
    monkeypatch.setattr(mod, "_prompt_llm", lambda prompt: f"RENDERED:{prompt}")
    return mod


# A classic SSTI escape: reach into Python object internals via attribute
# access. Non-weaponized — it only reads a count of subclasses, but on an
# unsandboxed renderer this same chain leads to arbitrary code execution.
_SSTI_PAYLOAD = "{{ ''.__class__.__mro__[1].__subclasses__()|length }}"


def test_model_template_ssti_is_blocked(parrot):
    """A model-proposed template that escapes to object internals must not render.

    On the unpatched tree ``jinja2.Template`` evaluates the attribute chain and
    returns a number (proof the sandbox escape worked). On the patched tree the
    SandboxedEnvironment raises SecurityError (a TemplateError subclass), which
    ``ParrotOrganism.run`` catches and surfaces as a render error.
    """
    out = parrot.ParrotOrganism(prompt_template=_SSTI_PAYLOAD).run("anything")

    assert out.startswith("Error rendering prompt:"), (
        f"expected the sandbox to reject the injection, got {out!r}"
    )
    assert "unsafe" in out  # SecurityError message from the sandbox
    # The escape must not have produced its numeric result.
    assert not out.startswith("RENDERED:")


def test_benign_template_still_evaluates(parrot):
    """Ordinary expressions must still render — the sink stays a live template."""
    out = parrot.ParrotOrganism(prompt_template="{{ 7*7 }}").run("anything")
    assert out == "RENDERED:49"


def test_phrase_substitution_is_verbatim(parrot):
    """``{{ phrase }}`` must pass the phrase through unchanged (no HTML escaping).

    Autoescape is intentionally off: the rendered text is a plaintext LLM prompt
    and the evolver compares the model's echo against the exact phrase, so a
    quoted phrase must survive byte-for-byte.
    """
    out = parrot.ParrotOrganism(prompt_template="Say {{ phrase }}").run('"bla bla".')
    assert out == 'RENDERED:Say "bla bla".'


def test_mutator_to_run_cycle_is_sandboxed(parrot):
    """End-to-end (the PoC fuzz target): a malicious template the *mutator*
    extracts from model output is also rendered sandboxed when the resulting
    organism runs."""
    # Drive ImproveParrotMutator.mutate with model output whose LAST fenced block
    # is the SSTI payload, exactly as the real parsing expects.
    fenced = f"Here is an improved template:\n```\n{_SSTI_PAYLOAD}\n```"
    parrot._prompt_llm = lambda prompt: fenced  # the mutate() call

    fc = parrot.ParrotEvaluationFailureCase(
        phrase="bla", response="nope", data_point_id="t0"
    )
    seed = parrot.ParrotOrganism(prompt_template="Say {{ phrase }}")
    children = parrot.ImproveParrotMutator().mutate(seed, [fc], [])
    assert children, "mutator should have produced a child organism"
    assert children[0].prompt_template == _SSTI_PAYLOAD

    # Now the child runs; restore the echo stub so we'd *see* any leaked render.
    parrot._prompt_llm = lambda prompt: f"RENDERED:{prompt}"
    out = children[0].run("anything")
    assert out.startswith("Error rendering prompt:")
    assert not out.startswith("RENDERED:")
