from types import SimpleNamespace

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ("/hatch fox --drafts 0", "--drafts must be between 1 and 8"),
        ("/hatch fox --concurrency 5", "--concurrency must be between 1 and 4"),
        ("/hatch fox --pose-attempts 4", "--pose-attempts must be between 1 and 3"),
    ],
)
def test_hatch_rejects_out_of_range_options(command, message, capsys):
    CLICommandsMixin()._handle_hatch_command(command)

    assert message in capsys.readouterr().out


def test_hatch_help_documents_run_scoped_options(capsys):
    CLICommandsMixin()._handle_hatch_command("/hatch --help")

    output = capsys.readouterr().out
    assert "--provider NAME" in output
    assert "--model ID" in output
    assert "--pose-attempts 1-3" in output
    assert "--no-adopt" in output


def test_hatch_forwards_quoted_concept_and_options(monkeypatch, tmp_path, capsys):
    from agent.pet.generate import imagegen, orchestrate

    captured = {}
    sprite = object()
    draft = tmp_path / "draft.png"
    draft.write_bytes(b"draft")

    def resolve_provider(**kwargs):
        captured["resolve"] = kwargs
        return sprite

    def generate_base_drafts(concept, **kwargs):
        captured["concept"] = concept
        captured["draft"] = kwargs
        return [draft]

    def hatch_pet(**kwargs):
        captured["hatch"] = kwargs
        return SimpleNamespace(slug="cyber-fox", display_name="Cyber Fox")

    adopted = []
    monkeypatch.setattr(imagegen, "resolve_provider", resolve_provider)
    monkeypatch.setattr(orchestrate, "generate_base_drafts", generate_base_drafts)
    monkeypatch.setattr(orchestrate, "hatch_pet", hatch_pet)
    monkeypatch.setattr("hermes_cli.pets._set_active", adopted.append)

    CLICommandsMixin()._handle_hatch_command(
        '/hatch "cyber fox" --provider fal --model fal-edit --style pixel '
        "--seed 42 --drafts 1 --concurrency 2 --pose-attempts 3 --name Sparky --no-adopt"
    )

    assert captured["resolve"] == {
        "require_references": True,
        "prefer": "fal",
        "model": "fal-edit",
    }
    assert captured["concept"] == "cyber fox"
    assert captured["draft"]["seed"] == 42
    assert captured["draft"]["concurrency"] == 2
    assert captured["hatch"]["pose_attempts"] == 3
    assert captured["hatch"]["provider"] is sprite
    assert adopted == []
    assert "not adopted" in capsys.readouterr().out


def test_hatch_rejects_missing_reference(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _prompt: "unused")

    CLICommandsMixin()._handle_hatch_command("/hatch fox --reference /definitely/missing.png")

    assert "Reference image not found" in capsys.readouterr().out
