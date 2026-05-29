"""Regression test: NeuTTS install detection in setup summary.

Guards against the bug where a redundant function-local
``import importlib.util`` inside ``_print_setup_summary`` /
``_setup_tts_provider`` made ``importlib`` a function-local for the whole
function. The earlier ``importlib.util.find_spec("neutts")`` reference then
raised ``UnboundLocalError``, which the surrounding ``except Exception``
silently swallowed -> NeuTTS was *always* reported as "not installed",
even when the package was importable. ``find_spec`` never actually ran.

See: fix(install) — remove redundant local `import importlib.util`.
"""
import importlib.util

from hermes_cli import setup as setup_mod


def test_print_setup_summary_detects_installed_neutts(monkeypatch, capsys):
    # Make NeuTTS look importable and record whether find_spec is reached.
    calls = []
    real_find_spec = importlib.util.find_spec

    def spy_find_spec(name, *args, **kwargs):
        calls.append(name)
        if name == "neutts":
            return object()  # pretend the package is installed
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", spy_find_spec)

    config = {"tts": {"provider": "neutts"}}
    setup_mod._print_setup_summary(config, "/tmp")

    out = capsys.readouterr().out

    # The detection probe must actually run (it never did with the bug).
    assert "neutts" in calls, (
        "importlib.util.find_spec('neutts') was never called — the "
        "UnboundLocalError was swallowed by `except Exception`."
    )
    # And an installed NeuTTS must be reported as available, not missing.
    assert "NeuTTS local" in out
    assert "NeuTTS — not installed" not in out
