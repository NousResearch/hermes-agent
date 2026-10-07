"""Falsy saved provider values survive a wizard re-run (#123599 review P2s).

`saved or default` treated 0/False/'' as unset, and the choices guard
`current and current in choices` short-circuited a native False to index 0 —
Enter then silently rewrote legal falsy saved values (default_trust: 0,
auto_extract: false from YAML) to the schema defaults. Real provider
discovery and real config.yaml I/O against a temp HERMES_HOME, matching the
harness of test_cmd_setup_rerun_offers_provider_saved_config; only the
curses picker, the deps installer and stdin are stubbed.
"""

from types import SimpleNamespace

import hermes_cli.memory_setup as memory_setup


def _run_holographic_rerun(tmp_path, monkeypatch, plugin_block: str) -> dict:
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  provider: holographic\n"
        f"plugins:\n"
        f"  hermes-memory-store:\n"
        + plugin_block
    )

    providers = memory_setup._get_available_providers()
    idx = next(i for i, (name, _hint, _p) in enumerate(providers) if name == "holographic")

    def fake_select(title, items, default=0, *, cancel_returns=None):
        if title == "Memory provider setup":
            return idx
        return default  # Enter: keep the offered current value

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(memory_setup, "_curses_select", fake_select)
    monkeypatch.setattr(memory_setup, "_prompt",
                        lambda label, default=None, secret=False: default or "")
    monkeypatch.setattr(memory_setup, "_install_dependencies", lambda name: None)
    monkeypatch.setattr(memory_setup, "get_hermes_home", lambda: tmp_path)

    memory_setup.cmd_setup(SimpleNamespace())

    import hermes_yaml as yaml

    return yaml.safe_load((tmp_path / "config.yaml").read_text())["plugins"]["hermes-memory-store"]


def test_falsy_saved_default_trust_zero_survives_rerun(tmp_path, monkeypatch):
    """default_trust: 0 is legal (initialize() floats it) — Enter must offer 0, not 0.5."""
    after = _run_holographic_rerun(tmp_path, monkeypatch,
                                   "    db_path: x.db\n"
                                   "    auto_extract: 'true'\n"
                                   "    default_trust: 0\n")
    # str(0) round-trips as the string '0' via save_config — the invariant is
    # that the ZERO survived (initialize() floats it); '0.5' would be the old bug.
    assert str(after["default_trust"]) == "0"


def test_native_false_auto_extract_preselects_false(tmp_path, monkeypatch):
    """A YAML native false must preselect 'false' in the choices picker, not index 0
    ('true'): the old guard short-circuited on the falsy current and Enter flipped
    auto-extraction off→on."""
    after = _run_holographic_rerun(tmp_path, monkeypatch,
                                   "    db_path: x.db\n"
                                   "    auto_extract: false\n"
                                   "    default_trust: '0.5'\n")
    assert after["auto_extract"] == "false"
