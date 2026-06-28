from types import SimpleNamespace


def test_cmd_moa_configure_preserves_reference_role_prompt(monkeypatch, tmp_path, capsys):
    """Interactive configure edits provider/model but must not erase role prompts."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openrouter
          model: x-ai/grok-4.3
          role_prompt: Find real-world edge cases.
      aggregator:
        provider: openrouter
        model: openai/gpt-5.5
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import moa_cmd
    from hermes_cli.config import load_config

    monkeypatch.setattr(
        moa_cmd,
        "_model_options",
        lambda: [
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": ["x-ai/grok-4.3", "openai/gpt-5.5"],
            }
        ],
    )

    def fake_prompt(title, rows, default=0):
        if title == "Add another reference model?":
            return 1
        return default

    monkeypatch.setattr(moa_cmd, "_prompt_choice", fake_prompt)

    moa_cmd.cmd_moa(SimpleNamespace(moa_command="configure", name="review"))

    cfg = load_config()
    refs = cfg["moa"]["presets"]["review"]["reference_models"]
    assert refs == [
        {
            "provider": "openrouter",
            "model": "x-ai/grok-4.3",
            "role_prompt": "Find real-world edge cases.",
        }
    ]
    assert "role: Find real-world edge cases." in capsys.readouterr().out
