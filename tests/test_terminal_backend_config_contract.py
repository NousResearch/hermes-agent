"""Static terminal backend config contract tests.

Batch 004 is deliberately docs + contract tests only. These tests keep the
existing terminal backend configuration contract explicit without changing
runtime terminal backend behavior.
"""

from __future__ import annotations

import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PLAN = REPO_ROOT / "docs" / "plans" / "2026-06-20-004-terminal-backend-config-contract.md"
CONFIG_DOC = REPO_ROOT / "website" / "docs" / "user-guide" / "configuration.md"
CLI_EXAMPLE = REPO_ROOT / "cli-config.yaml.example"
SANDBOX_POLICY = REPO_ROOT / "docs" / "security" / "sandbox-approval-policy.md"


def test_terminal_backend_config_plan_exists_and_documents_contract():
    assert PLAN.exists(), f"Missing batch 004 plan: {PLAN}"
    content = PLAN.read_text(encoding="utf-8")
    required_phrases = [
        "Batch 004: Terminal Backend Config Contract",
        "terminal.backend",
        "terminal.docker_image, terminal.docker_mount_cwd_to_workspace, terminal.modal_image, terminal.daytona_image",
        "no undocumented terminal backend alias keys are part of this contract",
        '  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"',
        '  docker_mount_cwd_to_workspace: false  # SECURITY: off by default.',
        '  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"',
        '  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"',
        "The default. Commands run directly on your machine with no isolation.",
        "local` has no isolation. Commands run with the operator user's host access.",
        "docker`, `singularity`, `modal`, and `daytona` run commands in a configured",
        "sandbox target.",
        "no runtime behavior changes",
        "static contract tests + plan/docs/cross-links for terminal backend configuration",
    ]
    for phrase in required_phrases:
        assert phrase in content, f"plan missing required phrase: {phrase}"


def test_configuration_doc_has_exact_terminal_backend_contract_phrases():
    content = CONFIG_DOC.read_text(encoding="utf-8")
    # exact documentation claims from configuration.md
    assert "The default. Commands run directly on your machine with no isolation." in content
    assert '  backend: local    # local | docker | ssh | modal | daytona | singularity' in content
    assert 'docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"' in content
    assert 'docker_mount_cwd_to_workspace: false  # Mount launch dir into /workspace' in content
    assert 'modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"' in content
    assert 'daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"' in content
    assert ":::warning" in content
    assert "same filesystem access as your user account" in content
    # cross-link to sandbox policy (existing, for terminal backend)
    assert "sandbox-approval-policy.md" in content
    assert "2026-06-20-004-terminal-backend-config-contract.md" in content


def test_cli_config_example_has_exact_terminal_backend_keys():
    content = CLI_EXAMPLE.read_text(encoding="utf-8")
    # exact config keys from cli-config.yaml.example
    assert '  backend: "local"' in content
    assert '  docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"' in content
    assert '  docker_mount_cwd_to_workspace: false  # SECURITY: off by default.' in content
    assert '  modal_image: "nikolaik/python-nodejs:python3.11-nodejs20"' in content
    assert '  daytona_image: "nikolaik/python-nodejs:python3.11-nodejs20"' in content
    assert "OPTION 1: Local execution (default)" in content
    assert "OPTION 3: Docker container" in content
    assert "OPTION 5: Modal cloud execution" in content
    assert "OPTION 6: Daytona cloud execution" in content


def test_sandbox_policy_has_terminal_backend_isolation_contract():
    content = SANDBOX_POLICY.read_text(encoding="utf-8")
    # exact claims from sandbox policy
    assert "local` has no isolation. Commands run with the operator user's host access." in content
    assert "docker`, `singularity`, `modal`, and `daytona` run commands in a configured" in content
    assert "sandbox target." in content
    assert "docker_mount_cwd_to_workspace: false" in content
    assert "Terminal-backend isolation" in content
    assert "2026-06-20-004-terminal-backend-config-contract.md" in content
    assert "config.yaml" in content
