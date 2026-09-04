"""Regression tests for the Honcho doctor probe (#86126).

The original code only called ``get_honcho_client(hcfg)``, which constructs
the SDK client without performing any network I/O — so a self-hosted Honcho
with a wrong port would still print "✓ Honcho connected". These tests pin
the post-fix behaviour: after constructing the client, doctor.py must issue
a real authenticated call (``_ensure_workspace``) and surface connection /
auth errors with distinct messages.
"""

import contextlib
import io
import sys
import types
from argparse import Namespace
from types import SimpleNamespace

import pytest

import hermes_cli.doctor as doctor_mod
import hermes_cli.gateway as gateway_cli  # noqa: F401  (imported for monkeypatch symmetry)


class _FakeHttpError(Exception):
    """Duck-typed Honcho SDK exception: status-only (auth rejected)."""

    def __init__(self, status: int, message: str = "rejected"):
        super().__init__(message)
        self.status = status
        self.code = ""


class _FakeConnError(Exception):
    """Duck-typed Honcho SDK exception: connection / timeout."""

    def __init__(self, code: str = "connection_error", message: str = "no route"):
        super().__init__(message)
        self.status = 0
        self.code = code


class TestHonchoDoctorProbe:
    """Doctor must actually probe the server, not only construct the SDK client."""

    def _make_hermes_home(self, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        import yaml
        (home / "config.yaml").write_text(yaml.dump({"memory": {"provider": "honcho"}}))
        # Also drop a honcho.json config file so the doctor reaches the branch
        honcho_cfg = home / "honcho.json"
        honcho_cfg.write_text(
            '{"enabled": true, "workspace_id": "ws-test", '
            '"base_url": "http://127.0.0.1:8000", "api_key": "sk-fake"}'
        )
        return home

    def _run(self, monkeypatch, tmp_path, *, get_client_side_effect=None, ensure_workspace_side_effect=None):
        """Run doctor with stubbed Honcho client and capture stdout.

        ``get_client_side_effect`` controls what ``get_honcho_client`` does:
        - ``None`` (default) → returns a fake client whose ``_ensure_workspace``
          does whatever ``ensure_workspace_side_effect`` says.
        - an Exception → raised by ``get_honcho_client`` itself (legacy path).
        """
        home = self._make_hermes_home(tmp_path)
        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", tmp_path / "project")
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        (tmp_path / "project").mkdir(exist_ok=True)

        # Stub tool availability
        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        # Stub auth checks
        try:
            from hermes_cli import auth as _auth_mod
            monkeypatch.setattr(_auth_mod, "get_nous_auth_status_local", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_xai_oauth_auth_status", lambda: {})
        except Exception:
            pass

        # Build a fake HonchoClientConfig
        fake_hcfg = SimpleNamespace(
            enabled=True,
            api_key="sk-fake",
            base_url="http://127.0.0.1:8000",
            workspace_id="ws-test",
            recall_mode="hybrid",
            write_frequency="async",
        )
        # Fake honcho.json resolver so doctor.py uses it
        monkeypatch.setattr(
            "plugins.memory.honcho.client.HonchoClientConfig.from_global_config",
            lambda: fake_hcfg,
        )
        monkeypatch.setattr(
            "plugins.memory.honcho.client.resolve_config_path",
            lambda: home / "honcho.json",
        )

        # Fake _ensure_workspace side effect
        def _ensure_workspace():
            if isinstance(ensure_workspace_side_effect, Exception):
                raise ensure_workspace_side_effect

        fake_client = SimpleNamespace(_ensure_workspace=_ensure_workspace)

        # Build get_honcho_client behavior
        def _get_honcho_client(_hcfg=None):
            if isinstance(get_client_side_effect, Exception):
                raise get_client_side_effect
            return fake_client

        def _reset_honcho_client():
            return None

        monkeypatch.setattr(
            "plugins.memory.honcho.client.get_honcho_client",
            _get_honcho_client,
        )
        monkeypatch.setattr(
            "plugins.memory.honcho.client.reset_honcho_client",
            _reset_honcho_client,
        )

        # Also patch the doctor module's symbol table so the late-bound imports
        # inside the function pick up the fakes.
        # The function uses:
        #   from plugins.memory.honcho.client import get_honcho_client, reset_honcho_client
        # so monkeypatching at the honcho.client module is sufficient.

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        return buf.getvalue()

    # --- Red proofs: each one fails on the pre-fix code ------------------------

    def test_unreachable_server_does_not_pretend_connected(self, monkeypatch, tmp_path):
        """#86126 L1 — server unreachable must surface as a FAIL, not '✓ Honcho connected'."""
        out = self._run(
            monkeypatch,
            tmp_path,
            ensure_workspace_side_effect=_FakeConnError("connection_error", "no route to host"),
        )
        assert "Honcho connected" not in out, (
            f"doctor printed 'Honcho connected' despite connection_error: {out!r}"
        )
        assert "Honcho" in out  # section is rendered
        # The fixed code classifies this as 'could not reach'
        assert "could not reach" in out or "Honcho unreachable" in out or "Honcho connection failed" in out, (
            f"missing unreachable signal: {out!r}"
        )

    def test_401_distinguishes_credential_rejection_from_connectivity(self, monkeypatch, tmp_path):
        """#86126 L3 — 401 must surface as credentials-rejected, not connectivity-failed."""
        out = self._run(
            monkeypatch,
            tmp_path,
            ensure_workspace_side_effect=_FakeHttpError(401, "no token"),
        )
        assert "Honcho connected" not in out
        assert "Honcho" in out
        assert "rejected" in out.lower(), f"401 not distinguished from connectivity: {out!r}"

    def test_403_distinguishes_credential_rejection_from_connectivity(self, monkeypatch, tmp_path):
        """#86126 L3 — 403 (forbidden) must surface as credentials-rejected."""
        out = self._run(
            monkeypatch,
            tmp_path,
            ensure_workspace_side_effect=_FakeHttpError(403, "forbidden"),
        )
        assert "Honcho connected" not in out
        assert "rejected" in out.lower(), f"403 not distinguished from connectivity: {out!r}"

    def test_happy_path_still_works(self, monkeypatch, tmp_path):
        """Real server reachable + auth ok → 'Honcho connected' must still print."""
        out = self._run(monkeypatch, tmp_path)  # default: no exception, success
        assert "Honcho connected" in out, f"happy path broken: {out!r}"
        assert "workspace=ws-test" in out

    def test_legacy_get_honcho_client_failure_still_caught(self, monkeypatch, tmp_path):
        """Legacy path: get_honcho_client() itself raises (e.g. ValueError, no key).

        The pre-fix except branch handles this; we must preserve it.
        """
        out = self._run(
            monkeypatch,
            tmp_path,
            get_client_side_effect=ValueError("Honcho API key not found"),
        )
        assert "Honcho connected" not in out
        # Should surface via the legacy 'Honcho connection failed' branch.
        assert "Honcho" in out