"""`hermes model` for a process provider gates on the CLI's login and lists its live picker."""
from unittest.mock import patch

from hermes_cli import model_setup_flows as flows
from providers.base import ProviderProfile


class _Profile(ProviderProfile):
    def __init__(self, status, live):
        super().__init__(name="proc-provider", display_name="Proc Provider", auth_type="external_process",
                         base_url="process://proc-provider", fallback_models=("pinned-a", "pinned-b"))
        self._status, self._live = status, live

    def setup_status(self, **_):
        return self._status

    def discover_models(self, **_):
        return self._live


def _run(profile, capsys):
    picked = {}
    def fake_pick(model_list, prompt, **kwargs):
        picked.update(models=model_list, notes=kwargs.get("notes"))
        return model_list[0] if model_list else None
    with patch("providers.get_provider_profile", return_value=profile), \
         patch("hermes_cli.auth.resolve_external_process_provider_credentials",
               return_value={"base_url": profile.base_url}), \
         patch.object(flows, "_pick_model_or_prompt", fake_pick), \
         patch.object(flows, "_finish_model", lambda *a, **k: picked.update(finished=a[0])):
        flows._model_flow_external_process({}, profile.name)
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
    assert picked["models"] == ["claude-sonnet-5[1m]", "claude-fable-5-1[1m]"]
    assert picked["notes"] == {"claude-fable-5-1[1m]": "usage credits"}
    assert picked["finished"] == "claude-sonnet-5[1m]"


def test_missing_cli_is_a_single_clean_line(capsys):
    status = {"available": False, "logged_in": False, "plan": "", "login_command": None,
              "detail": "Claude Code is not installed (no `claude` on PATH). Install it with `npm install -g @anthropic-ai/claude-code`."}
    picked, out = _run(_Profile(status, None), capsys)
    assert not picked and out.strip() == "✗ " + status["detail"]
