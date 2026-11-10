"""Built-in provider shadowing warning in the runtime resolver (GitHub #43026)."""

import contextlib
import logging


class _WarningCapture(logging.Handler):
    """Capture ``hermes_cli.runtime_provider`` warnings via a handler on the logger itself.

    ``caplog`` attaches to the ROOT logger and relies on propagation plus the process-wide
    ``logging.disable`` level, both of which an earlier test in the same session can leave
    changed (``hermes_cli/oneshot.py`` calls ``logging.disable(logging.CRITICAL)`` and never
    restores it). Recording at the source keeps this test about the warning, not about suite
    ordering."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _captured_provider_warnings():
    log = logging.getLogger("hermes_cli.runtime_provider")
    handler = _WarningCapture()
    prior_level, prior_disabled = log.level, log.disabled
    prior_global_disable = logging.root.manager.disable
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    log.disabled = False
    logging.disable(logging.NOTSET)
    try:
        yield handler.messages
    finally:
        logging.disable(prior_global_disable)
        log.removeHandler(handler)
        log.setLevel(prior_level)
        log.disabled = prior_disabled


def test_shadowed_builtin_provider_entry_warns_once(monkeypatch):
    """GitHub #43026: a ``providers.gemini`` routing entry is ignored because
    ``gemini`` is a canonical built-in — the ignore must be loud, not silent,
    and must fire at most once per provider per process."""
    import hermes_cli.config as cfg
    import hermes_cli.runtime_provider_custom as rpc

    shadow_config = {
        "providers": {
            "gemini": {
                "api_key": "AIzaSy-test",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            }
        }
    }
    monkeypatch.setattr(cfg, "load_config_readonly", lambda: shadow_config)
    monkeypatch.setattr(rpc, "_shadow_warned_providers", set())

    # Call through the owning module, not the ``runtime_provider`` re-export: other tests in the
    # suite re-import ``runtime_provider_custom``, which leaves the re-exported function bound to
    # a previous module incarnation whose (already-primed) warn-once cache this test cannot reset.
    with _captured_provider_warnings() as messages:
        assert rpc._get_named_custom_provider("gemini") is None
        assert rpc._get_named_custom_provider("gemini") is None

    warnings = [m for m in messages if "built-in provider" in m]
    assert len(warnings) == 1, messages
    assert "gemini" in warnings[0]


def test_builtin_provider_without_user_entry_does_not_warn(monkeypatch):
    """No providers./custom_providers entry for the built-in -> no warning."""
    import hermes_cli.config as cfg
    import hermes_cli.runtime_provider_custom as rpc

    monkeypatch.setattr(cfg, "load_config_readonly", dict)
    monkeypatch.setattr(rpc, "_shadow_warned_providers", set())

    with _captured_provider_warnings() as messages:
        assert rpc._get_named_custom_provider("gemini") is None

    assert not [m for m in messages if "built-in provider" in m]
