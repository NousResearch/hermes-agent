"""Wiring of probe classification into the interactive custom-provider flow.

Task 7 / #3263. The pure classification/decision logic is covered in
test_probe_classification.py; here we guard the highest-risk wiring: the
non-interactive fail-closed path (never prompt, exit non-zero, do not save)
and the --skip-validation escape hatch (no probe at all).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _drive_inputs(monkeypatch, url="https://typo.invalid/v1", key="sk-x"):
    """Feed the base-URL input() and the masked key prompt."""
    monkeypatch.setattr("builtins.input", lambda *a, **k: url)
    monkeypatch.setattr("hermes_cli.secret_prompt.masked_secret_prompt", lambda *a, **k: key)


def test_non_interactive_unreachable_fails_closed(monkeypatch):
    """A non-interactive run against an unreachable endpoint exits non-zero
    and never reaches save_config."""
    import hermes_cli.main as main
    import hermes_cli.models as models
    import hermes_cli.config as config

    _drive_inputs(monkeypatch)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(models, "probe_api_models",
                        lambda *a, **k: {"models": None, "error_class": "dns",
                                         "error_detail": "DNS resolution failed",
                                         "probed_url": "https://typo.invalid/v1/models",
                                         "suggested_base_url": None, "used_fallback": False})

    saved = {"called": False}
    monkeypatch.setattr(config, "save_config", lambda *a, **k: saved.__setitem__("called", True))

    with pytest.raises(SystemExit) as exc:
        main._model_flow_custom({"model": {}}, args=SimpleNamespace(skip_validation=False))

    assert exc.value.code == 1
    assert saved["called"] is False, "must not save on a fail-closed non-interactive probe"


def test_skip_validation_does_not_probe(monkeypatch):
    """--skip-validation must not call the network probe at all."""
    import hermes_cli.main as main
    import hermes_cli.models as models

    _drive_inputs(monkeypatch, url="https://api.example.com/v1")
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))

    probed = {"called": False}
    def _spy(*a, **k):
        probed["called"] = True
        return {"models": None, "error_class": "dns", "error_detail": "", "used_fallback": False}
    monkeypatch.setattr(models, "probe_api_models", _spy)

    # Abort the flow right after the probe step by sending EOF to the next
    # prompt (api_mode selection) — we only care that the probe was skipped.
    monkeypatch.setattr(main, "_prompt_custom_api_mode_selection",
                        lambda *a, **k: (_ for _ in ()).throw(EOFError()))

    try:
        main._model_flow_custom({"model": {}}, args=SimpleNamespace(skip_validation=True))
    except EOFError:
        pass

    assert probed["called"] is False, "--skip-validation must not probe the endpoint"


class _Inputs:
    """Sequenced input() stub: pops scripted answers; an Exception class/instance
    in the sequence is raised when reached (to bail out of the flow cleanly)."""

    def __init__(self, seq):
        self._seq = list(seq)

    def __call__(self, *a, **k):
        val = self._seq.pop(0)
        if isinstance(val, BaseException) or (isinstance(val, type) and issubclass(val, BaseException)):
            raise val
        return val


def test_declined_reenter_then_decline_save_does_not_save(monkeypatch):
    """auth → decline re-enter → 'Save anyway?' default-No → declined → returns
    without saving (locks the `proceed` fall-through after a declined re-enter)."""
    import hermes_cli.main as main
    import hermes_cli.models as models
    import hermes_cli.config as config

    # base URL, then "Re-enter API key now?" → n, then "Save anyway?" → n
    monkeypatch.setattr("builtins.input", _Inputs(["https://typo.invalid/v1", "n", "n"]))
    monkeypatch.setattr("hermes_cli.secret_prompt.masked_secret_prompt", lambda *a, **k: "sk-bad")
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(models, "probe_api_models",
                        lambda *a, **k: {"models": None, "error_class": "auth",
                                         "error_detail": "HTTP 401", "probed_url": "u",
                                         "suggested_base_url": None, "used_fallback": False})

    saved = {"called": False}
    monkeypatch.setattr(config, "save_config", lambda *a, **k: saved.__setitem__("called", True))

    result = main._model_flow_custom({"model": {}}, args=SimpleNamespace(skip_validation=False))
    assert result is None
    assert saved["called"] is False, "declining the save-anyway prompt must not persist"


def test_reenter_key_reprobes_with_new_key(monkeypatch):
    """auth → re-enter a new key → loop re-probes WITH the new key and succeeds."""
    import hermes_cli.main as main
    import hermes_cli.models as models

    # base URL, then "Re-enter API key now?" → y, then bail at model selection.
    monkeypatch.setattr("builtins.input", _Inputs(["https://api.example.com/v1", "y", EOFError]))
    keys = iter(["sk-bad", "sk-good"])
    monkeypatch.setattr("hermes_cli.secret_prompt.masked_secret_prompt", lambda *a, **k: next(keys))
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(main, "_prompt_custom_api_mode_selection", lambda *a, **k: "")

    calls = []
    def _probe(api_key, base_url, *a, **k):
        calls.append(api_key)
        if len(calls) == 1:
            return {"models": None, "error_class": "auth", "error_detail": "HTTP 401",
                    "probed_url": "u", "suggested_base_url": None, "used_fallback": False}
        return {"models": ["m1"], "error_class": None, "probed_url": "u",
                "resolved_base_url": base_url, "suggested_base_url": None, "used_fallback": False}
    monkeypatch.setattr(models, "probe_api_models", _probe)

    # The flow bails at the model-name prompt (EOFError) → caught → returns.
    main._model_flow_custom({"model": {}}, args=SimpleNamespace(skip_validation=False))

    assert calls == ["sk-bad", "sk-good"], \
        "re-entering a key must re-probe with the NEW key"
