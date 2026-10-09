"""Execute optional research helpers offline; upstream/SDK doubles are boundary spies.

This covers importability and client routing, not the paid evolution loop or a
live vector service. Document policy belongs to test_skill_document_contracts.
"""
import pickle
from pathlib import Path
import runpy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

RESEARCH = Path(__file__).resolve().parents[2] / "optional-skills/research"


@pytest.mark.parametrize("script", ["rag_pipeline.py", "memory_manager.py"])
def test_pinecone_client_uses_key_and_rejects_missing_key(script, monkeypatch, capsys):
    sdk = ModuleType("pinecone")
    constructor = Mock()
    monkeypatch.setattr(sdk, "Pinecone", constructor, raising=False)
    monkeypatch.setitem(sys.modules, "pinecone", sdk)
    get_client = runpy.run_path(str(RESEARCH / "pinecone-research/scripts" / script))["get_pinecone_client"]
    monkeypatch.setenv("PINECONE_API_KEY", "test-only-key")
    assert get_client() is constructor.return_value
    constructor.assert_called_once_with(api_key="test-only-key")
    monkeypatch.delenv("PINECONE_API_KEY")
    with pytest.raises(SystemExit, match="1"):
        get_client()
    assert "PINECONE_API_KEY" in capsys.readouterr().err
    assert constructor.call_count == 1


@pytest.mark.parametrize("relative", [
    "scripts/parrot_openrouter.py", "templates/custom_problem_template.py",
])
def test_evolver_provider_routing(relative, monkeypatch):
    # The upstream package is optional/AGPL; only its imported types are needed
    # to load these scripts and exercise their own provider client functions.
    class UpstreamType:
        def __class_getitem__(cls, _):
            return cls

    for module, names in {
        "cli_common": "build_hyperparameter_config_from_args register_hyperparameter_args parse_learning_log_view_type",
        "evolve_problem_loop": "EvolveProblemLoop",
        "learning_log": "LearningLogEntry",
        "problem": "EvaluationFailureCase EvaluationResult Evaluator Mutator Organism Problem",
    }.items():
        stub = ModuleType(f"darwinian_evolver.{module}")
        for name in names.split():
            setattr(stub, name, UpstreamType)
        monkeypatch.setitem(sys.modules, stub.__name__, stub)
    import openai

    constructor = Mock()
    monkeypatch.setattr(openai, "OpenAI", constructor)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-key")
    monkeypatch.setenv("EVOLVER_MODEL", "test/provider-model")
    script = runpy.run_path(str(RESEARCH / "darwinian-evolver" / relative))
    create = constructor.return_value.chat.completions.create
    create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="reply"))])
    assert script["_prompt_llm"]("question") == "reply"
    constructor.assert_called_once_with(api_key="test-only-key", base_url="https://openrouter.ai/api/v1")
    assert create.call_args.kwargs["model"] == "test/provider-model"
    assert create.call_args.kwargs["messages"] == [{"role": "user", "content": "question"}]
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        script["_client"]()
    assert constructor.call_count == 1


def test_snapshot_requires_trust_before_unpickling(tmp_path, monkeypatch, capsys):
    snapshot = tmp_path / "snapshot.pkl"
    snapshot.write_bytes(pickle.dumps({"population_snapshot": pickle.dumps({"organisms": []})}))
    script = RESEARCH / "darwinian-evolver/scripts/show_snapshot.py"
    main = runpy.run_path(str(script))["main"]
    monkeypatch.setattr(sys, "argv", [str(script), str(snapshot)])
    with pytest.raises(SystemExit, match="refusing to unpickle"):
        main()
    monkeypatch.setattr(sys, "argv", [str(script), str(snapshot), "--i-trust-this-file"])
    assert main() == 0
    assert "# organisms: 0" in capsys.readouterr().out
