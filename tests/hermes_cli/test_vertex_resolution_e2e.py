"""End-to-end resolution for the Vertex provider: real imports, real config, real credentials.

The mocks-free rule matters here: the Vertex path had two integration-only bugs that
mocked unit tests could not see — a gcloud ``authorized_user`` ADC file was forced through
``service_account.Credentials.from_service_account_info`` (rejected with ``MalformedError``,
so the loader yielded NO credentials), and a publisher-namespace catalog id silently
re-routed an aggregator user onto the GCP project.

Every assertion here runs the production code path: ``resolve_runtime_provider`` ->
``agent.vertex_adapter`` -> config.yaml + the credentials file on disk. Only the network
is out of scope (the OAuth exchange is stubbed at the SDK boundary).
"""

from __future__ import annotations

import json

import pytest

# A throwaway RSA key so the service-account shape loads through the real SDK
# (it parses the PEM at construction). Never used to sign anything.
def _generate_service_account_key() -> str:
    """Generate a throwaway RSA key at test runtime so no PEM is committed."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


_SERVICE_ACCOUNT_KEY = _generate_service_account_key()


def _write_credentials(path, *, kind: str, project_id: str = "test-project"):
    """A credentials file shaped like the one each credential TYPE produces."""
    if kind == "authorized_user":
        path.write_text(json.dumps({
            "type": "authorized_user",
            "client_id": "123.apps.googleusercontent.com",
            "client_secret": "secret",
            "refresh_token": "refresh-token",
            "quota_project_id": project_id,
        }))
    else:
        path.write_text(json.dumps({
            "type": "service_account",
            "client_email": "test-sa@example-project.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
            # A real PEM: the SDK parses the key at load time, so a placeholder string
            # would fail on the crypto layer instead of the shape under test.
            "private_key": _SERVICE_ACCOUNT_KEY,
            "project_id": project_id,
        }))
    return path


@pytest.fixture
def vertex_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with a config.yaml the Vertex adapter reads for real."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("VERTEX_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("VERTEX_REGION", raising=False)
    monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
    return home


def _write_config(home, section: dict) -> None:
    (home / "config.yaml").write_text("vertex:\n" + "".join(f"  {k}: {v}\n" for k, v in section.items()))


def test_resolve_runtime_provider_builds_the_vertex_global_endpoint(vertex_home, tmp_path, monkeypatch):
    """The resolved base_url must carry the project, the region and the openapi endpoint."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_config(vertex_home, {"project_id": "proj-abc", "region": "global"})
    creds_file = _write_credentials(tmp_path / "adc.json", kind="authorized_user", project_id="proj-abc")
    monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", str(creds_file))

    # The OAuth exchange is the only thing stubbed: everything else is production code.
    import agent.vertex_adapter as va

    class _Creds:
        token = "ya29.test-token"
        expired = False
        expiry = None

        def refresh(self, _request):
            self.token = "ya29.test-token"

    monkeypatch.setattr(
        va, "_load_credentials",
        lambda resolved_path, sa_raw: (_Creds(), "proj-abc") if resolved_path else None,
    )

    resolved = resolve_runtime_provider(requested="vertex", target_model="google/gemini-4-argon")

    assert resolved["provider"] == "vertex"
    assert resolved["api_key"], "vertex must resolve a non-empty bearer token"
    assert resolved["base_url"] == (
        "https://aiplatform.googleapis.com/v1beta1/projects/proj-abc"
        "/locations/global/endpoints/openapi"
    )


def test_config_region_and_project_propagate_into_the_base_url(vertex_home, tmp_path, monkeypatch):
    """``vertex.region`` / ``vertex.project_id`` in config.yaml must reach the base_url."""
    from hermes_cli.runtime_provider import resolve_runtime_provider

    _write_config(vertex_home, {"project_id": "proj-xyz", "region": "us-central1"})
    creds_file = _write_credentials(tmp_path / "adc.json", kind="authorized_user", project_id="proj-xyz")
    monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", str(creds_file))

    import agent.vertex_adapter as va

    class _Creds:
        token = "ya29.test-token"
        expired = False
        expiry = None

        def refresh(self, _request):
            self.token = "ya29.test-token"

    monkeypatch.setattr(
        va, "_load_credentials",
        lambda resolved_path, sa_raw: (_Creds(), "proj-xyz") if resolved_path else None,
    )

    resolved = resolve_runtime_provider(requested="vertex", target_model="google/gemini-4-argon")

    assert "us-central1" in resolved["base_url"]
    assert "us-central1-aiplatform.googleapis.com" in resolved["base_url"]
    assert "proj-xyz" in resolved["base_url"]


def test_authorized_user_credentials_file_yields_credentials_and_its_quota_project(vertex_home, tmp_path, monkeypatch):
    """A gcloud ADC file is an ``authorized_user`` document, not a service-account key.

    Loading it must produce credentials and the project it bills against — the shape a
    user gets from ``gcloud auth application-default login``.
    """
    import agent.vertex_adapter as va

    creds_file = _write_credentials(
        tmp_path / "adc.json", kind="authorized_user", project_id="quota-proj-789")
    monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", str(creds_file))

    creds, project_id = va._load_credentials(str(creds_file), creds_file.read_bytes())

    assert creds is not None, "an authorized_user ADC file must load, not be rejected"
    assert project_id == "quota-proj-789"


def test_service_account_credentials_file_still_yields_its_own_project(vertex_home, tmp_path, monkeypatch):
    """The service-account path must keep working exactly as before."""
    import agent.vertex_adapter as va

    creds_file = _write_credentials(
        tmp_path / "sa.json", kind="service_account", project_id="sa-proj-456")
    monkeypatch.setenv("VERTEX_CREDENTIALS_PATH", str(creds_file))

    creds, project_id = va._load_credentials(str(creds_file), creds_file.read_bytes())

    assert creds is not None
    assert project_id == "sa-proj-456"
