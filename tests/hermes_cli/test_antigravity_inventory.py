"""Model picker exposure for the local Antigravity runtime."""

from agent.transports.antigravity_cli import AntigravityCapabilities, AntigravityClient
from hermes_cli.inventory import build_model_options_payload, load_picker_context


def test_model_options_exposes_detected_antigravity_runtime(monkeypatch):
    monkeypatch.setattr(
        AntigravityClient,
        "probe",
        lambda self, timeout=2.0: AntigravityCapabilities(
            available=True,
            executable="/opt/agy",
            version=(1, 2, 7),
            stream_json=True,
            sandbox=True,
            resume=True,
            authenticated=True,
        ),
    )

    payload = build_model_options_payload(load_picker_context())
    row = next(provider for provider in payload["providers"] if provider["slug"] == "google-antigravity")

    assert row["models"] == ["auto"]
    assert row["authenticated"] is True
    assert row["runtime_status"] == {
        "installed": True,
        "version": "1.2.7",
        "authentication": "authenticated",
    }
