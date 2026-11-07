"""Falsy saved provider values survive a wizard re-run (#123599 review P2s).

`saved or default` treated 0/False/'' as unset, and the choices guard
`current and current in choices` short-circuited a native False to index 0 —
Enter then silently rewrote legal falsy saved values to the schema defaults.
The choices path runs against real provider discovery (byterover's
``auto_extract``, the one surviving schema field with falsy-able choices);
the free-prompt path is exercised directly against ``_prompt_schema_fields``.
Only the curses picker, the deps installer and stdin are stubbed.
"""

from types import SimpleNamespace

import hermes_cli.memory_setup as memory_setup


def test_native_false_auto_extract_preselects_false(tmp_path, monkeypatch):
    """A YAML native false must preselect 'false' in the choices picker, not index 0
    ('true'): the old guard short-circuited on the falsy current and Enter flipped
    auto-extraction off→on."""
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  provider: byterover\n"
        "  byterover:\n"
        "    auto_extract: false\n"
    )

    providers = memory_setup._get_available_providers()
    idx = next(i for i, (name, _hint, _p) in enumerate(providers) if name == "byterover")
    picked = []

    def fake_select(title, items, default=0, *, cancel_returns=None):
        if title == "Memory provider setup":
            return idx
        picked.append((default, [label for label, _note in items]))
        return default  # Enter: keep the offered current value

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(memory_setup, "_curses_select", fake_select)
    monkeypatch.setattr(memory_setup, "_prompt",
                        lambda label, default=None, secret=False: default or "")
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda name: None)
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)

    memory_setup.cmd_setup(SimpleNamespace())

    assert picked, "the auto_extract choices picker never ran"
    for default, labels in picked:
        assert labels[default] == "false", (
            f"choices picker offered {labels[default]!r} as current; the saved "
            "native false must preselect 'false', not the 'true' default"
        )


def test_falsy_saved_zero_is_offered_as_current(monkeypatch):
    """A legal falsy saved value (0) must be offered as current on the free-prompt
    path: `saved or default` used to coerce it to the schema default (0.5), so Enter
    silently rewrote it."""
    fields = [{"key": "default_trust", "description": "Trust threshold", "default": "0.5"}]
    offered = {}

    def fake_prompt(label, default=None, secret=False):
        offered["default"] = default
        return default or ""

    monkeypatch.setattr(memory_setup, "_prompt", fake_prompt)

    provider_config = {"default_trust": 0}
    assert memory_setup._prompt_schema_fields("byterover", fields, provider_config, {}) is True
    assert offered["default"] == "0", (
        f"prompt offered {offered['default']!r}; the saved 0 must be offered as "
        "current, not the 0.5 schema default"
    )
