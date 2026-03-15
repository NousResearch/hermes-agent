from __future__ import annotations

from argparse import Namespace
from unittest.mock import patch

import hermes_cli.setup as setup_mod


def test_setup_sections_include_workspace():
    section_names = [name for name, _, _ in setup_mod.SETUP_SECTIONS]
    assert "workspace" in section_names


def test_setup_workspace_rag_installs_optional_runtime_and_updates_config(monkeypatch):
    config = {}

    yes_no_answers = iter([True, True, True, True])
    monkeypatch.setattr(setup_mod, "prompt_yes_no", lambda *args, **kwargs: next(yes_no_answers))
    monkeypatch.setattr(setup_mod, "prompt_choice", lambda *args, **kwargs: 1)  # gated
    monkeypatch.setattr(setup_mod, "_workspace_rag_dependencies_ready", lambda: False)
    monkeypatch.setattr(setup_mod, "_install_workspace_rag_dependencies", lambda: True)

    setup_mod.setup_workspace_rag(config)

    assert config["workspace"]["enabled"] is True
    assert config["knowledgebase"]["enabled"] is True
    assert config["knowledgebase"]["retrieval_mode"] == "gated"
    assert config["knowledgebase"]["embeddings"]["provider"] == "local"
    assert config["knowledgebase"]["embeddings"]["model"] == "google/embeddinggemma-300m"
    assert config["knowledgebase"]["reranker"]["enabled"] is True
    assert config["knowledgebase"]["reranker"]["provider"] == "local"


def test_run_setup_wizard_workspace_section_dispatches(monkeypatch, tmp_path):
    args = Namespace(section="workspace", non_interactive=False, reset=False)
    config = {}

    monkeypatch.setattr(setup_mod, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(setup_mod, "load_config", lambda: config)
    monkeypatch.setattr(setup_mod, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr(setup_mod, "is_interactive_stdin", lambda: True)

    called = {}

    def fake_workspace(cfg):
        called["config"] = cfg

    monkeypatch.setattr(setup_mod, "setup_workspace_rag", fake_workspace)
    monkeypatch.setattr(setup_mod, "SETUP_SECTIONS", [
        ("workspace", "Workspace Knowledgebase & Local RAG", fake_workspace),
    ])

    with patch.object(setup_mod, "save_config") as save_config:
        setup_mod.run_setup_wizard(args)

    assert called["config"] is config
    save_config.assert_called_once_with(config)
