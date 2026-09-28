import hashlib
import json

import pytest

from hermes_cli import report_catalog


@pytest.fixture(autouse=True)
def approved_root(monkeypatch, tmp_path):
    monkeypatch.setenv(report_catalog.APPROVED_ROOTS_ENV, str(tmp_path))


def _write_catalog(tmp_path):
    root = tmp_path / "report"
    root.mkdir()
    html = root / "seo-report.html"
    html.write_text("<html><body>bound-run</body></html>", encoding="utf-8")
    csv = root / "findings.csv"
    csv.write_text("id\nF-1\n", encoding="utf-8")
    detail = root / "f001-details.html"
    detail.write_text("<html><body>F001 bound detail</body></html>", encoding="utf-8")
    manifest = root / "export-manifest.json"
    manifest.write_text(json.dumps({
        "run_id": "run-1",
        "prompt_version": "1.4",
        "artifacts": [
            {"filename": "findings.csv", "sha256": hashlib.sha256(csv.read_bytes()).hexdigest()},
            {"filename": "f001-details.html", "sha256": hashlib.sha256(detail.read_bytes()).hexdigest()},
        ],
    }), encoding="utf-8")
    entry = {
        "report_id": "report-1",
        "run_id": "run-1",
        "prompt_version": "1.4",
        "root": str(root),
        "manifest": "export-manifest.json",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "primary_artifacts": [{"filename": "seo-report.html", "sha256": hashlib.sha256(html.read_bytes()).hexdigest()}],
    }
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"schema_version": 1, "bound_run_id": "run-1", "bound_prompt_version": "1.4", "allowed_roots": [str(tmp_path)], "reports": [entry]}), encoding="utf-8")
    return catalog, root


def test_catalog_resolves_only_manifest_bound_files(tmp_path):
    catalog, _root = _write_catalog(tmp_path)
    filename, content, mime, digest = report_catalog.resolve_artifact("report-1", "seo-report.html", catalog)
    assert filename == "seo-report.html"
    assert content == b"<html><body>bound-run</body></html>"
    assert mime.startswith("text/html")
    assert len(digest) == 64
    with pytest.raises(report_catalog.ReportCatalogError):
        report_catalog.resolve_artifact("report-1", "../secret", catalog)
    with pytest.raises(report_catalog.ReportCatalogError):
        report_catalog.resolve_artifact("report-1", "not-registered.csv", catalog)


def test_catalog_fails_closed_on_hash_drift(tmp_path):
    catalog, root = _write_catalog(tmp_path)
    (root / "seo-report.html").write_text("tampered", encoding="utf-8")
    with pytest.raises(report_catalog.ReportCatalogError, match="hash drift"):
        report_catalog.resolve_artifact("report-1", "seo-report.html", catalog)


def test_catalog_fails_closed_on_manifest_drift(tmp_path):
    catalog, root = _write_catalog(tmp_path)
    manifest = root / "export-manifest.json"
    manifest.write_text(manifest.read_text().replace("run-1", "changed"), encoding="utf-8")
    with pytest.raises(report_catalog.ReportCatalogError, match="manifest hash drift"):
        report_catalog.resolve_artifact("report-1", "findings.csv", catalog)


def test_catalog_rejects_root_outside_trusted_root(tmp_path):
    catalog, root = _write_catalog(tmp_path)
    payload = json.loads(catalog.read_text())
    payload["reports"][0]["root"] = str(root.parent.parent)
    catalog.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(report_catalog.ReportCatalogError, match="outside"):
        report_catalog.load_catalog(catalog)


def test_catalog_v2_uses_per_report_run_binding(tmp_path):
    catalog, _root = _write_catalog(tmp_path)
    payload = json.loads(catalog.read_text())
    payload["schema_version"] = 2
    payload.pop("bound_run_id")
    payload.pop("bound_prompt_version")
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    reports = report_catalog.load_catalog(catalog)

    assert reports["reports"][0]["run_id"] == "run-1"
    assert reports["reports"][0]["prompt_version"] == "1.4"


def test_catalog_v2_rejects_legacy_global_binding(tmp_path):
    catalog, _root = _write_catalog(tmp_path)
    payload = json.loads(catalog.read_text())
    payload["schema_version"] = 2
    catalog.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(report_catalog.ReportCatalogError, match="per-report binding"):
        report_catalog.load_catalog(catalog)


def test_seo_catalog_environment_binding_precedes_legacy_catalog(tmp_path, monkeypatch):
    seo_catalog = tmp_path / "seo-catalog.json"
    legacy_catalog = tmp_path / "legacy-catalog.json"
    monkeypatch.setenv(report_catalog.SEO_CATALOG_ENV, str(seo_catalog))
    monkeypatch.setenv(report_catalog.CATALOG_ENV, str(legacy_catalog))

    assert report_catalog.catalog_path() == seo_catalog.resolve()


def test_report_routes_require_auth_and_serve_bound_bytes(tmp_path, monkeypatch):
    catalog, _root = _write_catalog(tmp_path)
    monkeypatch.setenv(report_catalog.CATALOG_ENV, str(catalog))
    from starlette.testclient import TestClient
    from hermes_cli import web_server

    previous = getattr(web_server.app.state, "auth_required", False)
    web_server.app.state.auth_required = False
    try:
        client = TestClient(web_server.app)
        assert client.get("/api/reports").status_code == 401
        assert client.get("/reports/report-1/f001-details.html").status_code == 401
        client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        listed = client.get("/api/reports")
        assert listed.status_code == 200
        assert listed.json()["reports"][0]["run_id"] == "run-1"
        report = client.get("/api/reports/report-1/report.html")
        assert report.status_code == 200
        assert report.headers["content-type"].startswith("text/html")
        assert "bound-run" in report.text
        detail = client.get("/reports/report-1/f001-details.html")
        assert detail.status_code == 200
        assert detail.headers["content-disposition"].startswith("inline")
        assert detail.headers["x-report-sha256"] == hashlib.sha256(detail.content).hexdigest()
        download = client.get("/api/reports/report-1/findings.csv")
        assert download.status_code == 200
        assert download.headers["content-disposition"].startswith("attachment")
        assert download.headers["x-report-sha256"] == hashlib.sha256(download.content).hexdigest()
        assert client.get("/api/reports/report-1/../secret").status_code in {404, 400, 307}
    finally:
        web_server.app.state.auth_required = previous


def test_report_routes_fail_closed_when_catalog_artifact_changes(tmp_path, monkeypatch):
    catalog, root = _write_catalog(tmp_path)
    monkeypatch.setenv(report_catalog.CATALOG_ENV, str(catalog))
    from starlette.testclient import TestClient
    from hermes_cli import web_server

    client = TestClient(web_server.app, headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN})
    (root / "findings.csv").write_text("tampered", encoding="utf-8")
    assert client.get("/api/reports").status_code == 503
    assert client.get("/api/reports/report-1/findings.csv").status_code == 409
