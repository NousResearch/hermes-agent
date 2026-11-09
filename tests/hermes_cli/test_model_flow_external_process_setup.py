"""`hermes model` for a process provider gates on the CLI's login and lists its live picker.

The generic plugin flow (``_model_flow_plugin_provider``) drives a registered ``external_process``
profile end to end: credential resolution, the optional ``setup_status`` login gate, and the
optional ``discover_models`` rows (ids + per-row notes) merged with the pinned catalog.
"""
import sys
from unittest.mock import patch

import pytest

from hermes_cli import model_setup_flows as flows
from providers import register_provider
from providers.base import ProviderProfile


class _Profile(ProviderProfile):
    def __init__(self, status, live):
        super().__init__(name="proc-provider", display_name="Proc Provider", auth_type="external_process",
                         base_url="process://proc-provider", process_command=sys.executable,
                         fallback_models=("pinned-a", "pinned-b"))
        self._status, self._live = status, live

    def setup_status(self, **_):
        return self._status

    def discover_models(self, **_):
        return self._live


def _run(profile, capsys):
    register_provider(profile)
    picked = {}

    def fake_pick(model_list, prompt, **kwargs):
        picked.update(models=model_list, notes=kwargs.get("notes"))
        return model_list[0] if model_list else None

    with patch.object(flows, "_pick_model_or_prompt", fake_pick), \
         patch.object(flows, "_finish_model", lambda *a, **k: picked.update(finished=a[0])):
        flows._model_flow_plugin_provider({}, profile.name)
    return picked, capsys.readouterr().out


def test_logged_out_without_tty_stops_with_login_instruction(capsys):
    status = {"available": True, "logged_in": False, "plan": "", "login_command": ["claude", "auth", "login"],
              "detail": "Claude Code is installed but not logged in. Run `claude auth login`, then select this provider again."}
    with patch("sys.stdin.isatty", return_value=False):
        picked, out = _run(_Profile(status, None), capsys)
    assert not picked and "not logged in" in out and "claude auth login" in out


def test_logged_in_lists_live_picker_with_notes(capsys):
    status = {"available": True, "logged_in": True, "plan": "Claude Pro", "login_command": ["claude", "auth", "login"], "detail": ""}
    live = [{"id": "claude-sonnet-5[1m]", "label": "Sonnet 5", "note": ""},
            {"id": "claude-fable-5-1[1m]", "label": "Fable 5.1", "note": "usage credits"}]
    picked, out = _run(_Profile(status, live), capsys)
    assert "Claude Pro" in out
    # Live rows are offered together with the pinned catalog so no declared id disappears.
    assert set(picked["models"]) >= {"claude-sonnet-5[1m]", "claude-fable-5-1[1m]", "pinned-a", "pinned-b"}
    assert picked["notes"] == {"claude-fable-5-1[1m]": "usage credits"}
    assert picked["finished"] == picked["models"][0]


@pytest.mark.parametrize("via_status", [True, False])
def test_missing_cli_stops_before_any_pick(capsys, via_status):
    """Missing binary: the generic credential step refuses (core check) and, when the profile also
    reports it, its own ``detail`` is what the user reads. Either way nothing is picked or saved."""
    detail = "Claude Code is not installed (no `claude` on PATH). Install it with `npm install -g @anthropic-ai/claude-code`."
    status = {"available": False, "logged_in": False, "plan": "", "login_command": None, "detail": detail}
    profile = _Profile(status if via_status else None, None)
    if not via_status:
        profile.process_command = "definitely-not-a-real-cli-binary"
    picked, out = _run(profile, capsys)
    assert not picked
    assert (detail in out) if via_status else ("Could not find" in out and "definitely-not-a-real-cli-binary" in out)
