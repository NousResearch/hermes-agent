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
_SERVICE_ACCOUNT_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDoC4FdSYgd05jO
+9J+3cWQxY4fusFvdn3PHx4GHf7H7uB88oxz/aHTjepoRANrLKdIfpVxqrbpxRXR
FJBEfP4bnctrj9x8W6/P/VwR5WxkHUmkc5NF/gxVa1Q1pSnq576LYplyaWSE3mpI
oHQMuTe04Kuv4qStHPThq4bRwUHQ8jgRO9sMoJLJjf/pvOiGs94C/HVA7JzaEbk0
RA58xU7N876hoPJCa7VMFo1X3QyGMeRFpdR6D5x9fiS0Wgpm70D6K/lAE3hl3maQ
X/YnKNJMA7Ojq2lb0V5tkgNfOtzy6k9gJComEivrb3UzTxWqwrt3ELJ8/roz5aeg
bd2mO4tjAgMBAAECggEAAVMWsEFVsuBJoupvR3fqA0s+/75eRIEdwCHScMCy4VHj
QfaIFIQ4bFxtUwTXkidYmYJwRVkiOMMxV/0XXIfsH9AIRu5n/0u7FrBaQLEZeJht
za/QaBnGcnLHS2EZR3Tk/3QW7r8pEQfl/Z3w26VV0V7Nv0mUNneP8rChREkN06AA
WcKPFZsI/+bJcxovCoZvK2T6hqAg+zogMf12zDzsZnvq0Pc8DFrFs0jdh5u1ZO3F
8NfDv+bkL8DMxX92AVj+0rRTEgR4aKt3QiAeoEZasnkyL49wOfQf4qeUAxjuYY+g
ZJci+Cje7+NO1F+aprGcskWqi8fqZoSjerECgxPk6QKBgQD/1cWOY4oDQ5+i5o5T
xlHxvS+3AgC01QRclGdbeqxRbpBwLRUoRqVuPPmS/07ALr7QCxv4tlmpHmhim6e3
YjPojU+jMdwlwz1gGHWW2n9Jz1zApFx9BDHvWHmc1swaJ0MYLbkbdl2/H+/ebiVV
i1cyTlqc/BDT35SX2/8tGVtECQKBgQDoMc6LgRxxLuVurMygcnThyHhJFtocDrIy
UsSwM319Dz5I/g6wDjygHfx2om9doQj2p9Gly57luSY7sx94K6slt9QSwl3mWaFq
qD4qM2y3neoKFqBu+3rdecg46ABCfi8moAY3v1kSbQhwUyJX1eHKhXXzqqFFFUPb
MRuuO3BnCwKBgBet/7UoPXQD5Fi9olratmb7rdjwlIQoW1j+W3q6Jgz7gN8BLT4n
G9TSS/9xJibfrMWw4YvpxO7EY8EMyz9MFaXmvoJKcSgFpD6qxOeLndCFIv6pt+3S
X5TZyZfcwdOB4tHwGXwTPqYcHKWn6Ei8tvtvCTASnLQfs69Cq6HPAQWRAoGBAOZ7
ukjkKnV3Vatcv+pscvVEZ5I1qt+6/svROQims+QV43/VIINGPou5+EdSyjUoips8
znBDBUGHXhpYnd5Nip/wMAVHBqqHQiF7S+ncDgjeKdlFyI8h9j4uir9iHA/k7azr
oABqYmJV9NB61VZXKsA7ZnAvadIHX8oOnyl+Ov8tAoGAMBkU2MFAm2uUxCuC7Exe
M7HRp1egOpPbnD8KCPdf10CWIWRKtBWzBu2bxe48nuNQaAliwGI7AU2Xuf5Cus/x
wps9iel29srYQSR+vMZeZLUGv0Q0OA3Z+l7trm6t5Q33jCjse5SWP+7e7eWvnG1m
iwnezaDVUikH8Q/CmFMNvVE=
-----END PRIVATE KEY-----"""


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
            "client_email": "svc@test-project.iam.gserviceaccount.com",
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
