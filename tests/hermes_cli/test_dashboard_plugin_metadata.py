"""Dashboard declarations must survive real discovery and token-bearing HTTP requests.

Backend metadata slice only: these tests do not execute JS or enforce SDK admission.
"""

import base64
import hashlib
import json

import pytest


JS = b"window.syntheticPlugin = true;\n"
CSS = b".synthetic { color: red; }\n"


def sri(content, algorithm="sha384"):
    return algorithm + "-" + base64.b64encode(hashlib.new(algorithm, content).digest()).decode("ascii")


@pytest.fixture
def plugin_host(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: home)
    monkeypatch.delenv("HERMES_ENABLE_PROJECT_PLUGINS", raising=False)
    from hermes_cli import plugins

    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: home / "bundled")
    from hermes_cli import web_server
    from starlette.testclient import TestClient

    monkeypatch.setattr(web_server, "_dashboard_plugins_cache", None)
    # Do not enter TestClient: that would start the app's lifespan services.
    monkeypatch.setattr(web_server.app.state, "bound_host", "127.0.0.1", raising=False)
    client = TestClient(web_server.app, base_url="http://127.0.0.1")
    headers = {web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN}

    def install(name, declarations):
        dashboard = home / "plugins" / name / "dashboard"
        (dashboard / "dist").mkdir(parents=True)
        manifest = {
            "name": name, "label": name, "entry": "dist/index.js",
            "css": "dist/style.css", "api": "../outside.py",
            "unrecognized_metadata": {"must_not": "leak"}, **declarations,
        }
        (dashboard / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (dashboard / "dist/index.js").write_bytes(JS)
        (dashboard / "dist/style.css").write_bytes(CSS)
        (dashboard / "private.py").write_text("# synthetic backend\n", encoding="utf-8")
        (dashboard.parent / "outside.js").write_text("// outside dashboard\n", encoding="utf-8")
        return manifest

    def configure(enabled, disabled=()):
        (home / "config.yaml").write_text(
            json.dumps({"plugins": {"enabled": list(enabled), "disabled": list(disabled)}}),
            encoding="utf-8",
        )

    yield client, headers, install, configure, web_server
    client.close()


@pytest.mark.parametrize("declarations", [
    {},
    {"integrity": sri(JS)},
    {"integrity": sri(JS, "sha256")},
    {"integrity": sri(JS, "sha512")},
    {"integrity": sri(JS) + " " + sri(JS)},
    {"integrity": " \t" + sri(JS) + "\n"},
    {"integrity": sri(JS, "sha256") + "\t\r\n\f " + sri(JS) + " " + sri(JS, "sha512")},
    *[{"integrity": sri(JS, algorithm).rstrip("=")} for algorithm in ("sha256", "sha512")],
    {"integrity": "sha384-" + base64.urlsafe_b64encode(hashlib.sha384(JS).digest()).decode("ascii")},
    {"integrity": sri(JS) + "?reserved-option?another=value"},
    {"integrity": sri(JS) + "?"},
    {"css_integrity": sri(CSS)},
    {"sdk": {"min": "1.1", "max": "1.x"}},
    {"sdk": {"min": "1.1.0", "max": "1.9.2"}, "integrity": sri(JS), "css_integrity": sri(CSS)},
    {"sdk": {"min": "0.0", "max": "2.0"}, "integrity": sri(JS), "css_integrity": sri(CSS)},
    {"sdk": {"min": "1.1", "max": "1.1.0"}},
])
def test_declared_metadata_roundtrips_without_weakening_guards(plugin_host, declarations):
    client, headers, install, configure, server = plugin_host
    original = install("synthetic", declarations)
    configure(["synthetic"])
    discovered = server._get_dashboard_plugins(force_rescan=True)
    entry = next(p for p in discovered if p["name"] == "synthetic")

    url = "/api/dashboard/plugins"
    # Enumeration is intentionally public on current upstream. Rescan is not.
    assert client.get(url).status_code == 200
    rescan = url + "/rescan"
    assert client.get(rescan).status_code == 401
    assert client.get(rescan, headers={next(iter(headers)): "invalid-test-token"}).status_code == 401
    assert client.get(rescan, headers=headers).status_code == 200
    assert client.get(url, headers={**headers, "host": "untrusted.invalid"}).status_code == 400
    response = client.get(url, headers=headers)
    assert response.status_code == 200
    public = next(p for p in response.json() if p["name"] == "synthetic")
    for field in ("sdk", "integrity", "css_integrity"):
        if field in declarations:
            assert entry[field] == original[field]
            assert public[field] == original[field]
        else:
            assert field not in entry
            assert field not in public
    assert "unrecognized_metadata" not in public
    assert not any(key.startswith("_") for key in public)
    assert public["has_api"] is False  # unsafe API path remains rejected

    assets = "/dashboard-plugins/synthetic/"
    for path, content in (("dist/index.js", JS), ("dist/style.css", CSS)):
        asset = client.get(assets + path)
        assert asset.status_code == 200
        assert asset.content == content
        assert "no-store" in asset.headers["cache-control"]
    assert client.get(assets + "private.py").status_code == 404
    assert client.get(assets + "%2e%2e/outside.js").status_code == 403
    assert client.get(assets + "dist/missing.js").status_code == 404

    for enabled, disabled in (([], []), (["synthetic"], ["synthetic"])):
        configure(enabled, disabled)
        assert not any(p["name"] == "synthetic" for p in client.get(url, headers=headers).json())
        assert client.get(assets + "dist/index.js").status_code == 404


@pytest.mark.parametrize("field,value", [
    # Replace the old uppercase-positive policy: the SRI spec permits case
    # variants, but Chromium ignores them, potentially leaving no effective hash.
    # Reject the whole list; do not repair casing or drop its strongest member.
    *[pytest.param("integrity", sri(JS, algorithm).replace(algorithm, algorithm.upper()),
                   id=f"uppercase-{algorithm}") for algorithm in ("sha256", "sha384", "sha512")],
    *[pytest.param("integrity", sri(JS, algorithm).replace(algorithm, algorithm.capitalize()),
                   id=f"mixed-case-{algorithm}") for algorithm in ("sha256", "sha384", "sha512")],
    pytest.param("integrity", sri(b"different bytes").replace("sha384-", "SHA384-"),
                 id="uppercase-mismatched-ignored-by-chromium"),
    pytest.param("integrity", sri(JS, "sha256") + " "
                 + sri(b"different bytes", "sha512").replace("sha512-", "SHA512-"),
                 id="uppercase-stronger-token-dropped"),
    pytest.param("integrity", sri(JS, "sha512").replace("sha512-", "SHA512-")
                 + " " + sri(JS, "sha256"), id="uppercase-first-in-mixed-list"),
    *[(field, value) for field in ("integrity", "css_integrity") for value in (
        None, False, 384, [], {}, "", "sha256-" + "A" * 64,
        "sha384-" + "A" * 63, "sha384-" + "A" * 65,
        "sha384-" + "!" * 64, sri(JS) + "=",
    )],
    # Whitespace and lists are valid JS SRI, not invalid declarations (R1).
    # CSS retains its separate, single standard-base64 SHA384 policy.
    *[("css_integrity", value) for value in (
        "sha384-" + "_" * 64, " " + sri(JS), sri(JS) + "\n",
        sri(JS) + " " + sri(CSS), sri(CSS, "sha256"), sri(CSS, "sha512"),
    )],
    *[("integrity", value) for value in (
        " \t\r\n\f", "sha1-" + "A" * 28,
        "sha256-" + "A" * 42 + "=", "sha256-" + "A" * 44 + "=",
        "sha512-" + "A" * 85 + "==", "sha512-" + "A" * 87 + "==",
        sri(JS, "sha256") + "=", sri(JS, "sha512")[:-1],
        sri(JS) + " sha384-bad", "sha384-bad " + sri(JS),
        sri(JS) + "," + sri(JS), sri(JS) + "\v" + sri(JS),
        sri(JS) + "\u00a0" + sri(JS), sri(JS) + "?\x00", sri(JS) + "?\u00e9",
    )],
    *[("sdk", value) for value in (
        None, False, 1, "1.1", [], {}, {"min": "1.1"}, {"max": "1.x"},
        {"min": 1.1, "max": "1.x"}, {"min": "1.1", "max": None},
        {"min": "1.x", "max": "1.x"}, {"min": "1.1", "max": "*"},
        {"min": "1.1", "max": "1.2.x"}, {"min": "01.1", "max": "1.x"},
        {"min": "1.1.0-beta", "max": "1.x"}, {"min": " 1.1", "max": "1.x"},
        {"min": "1.1", "max": "1.x\n"}, {"min": "1.1", "max": "1.x", "extra": True},
        {"min": "2.0", "max": "1.x"}, {"min": "1.2.1", "max": "1.2"},
    )],
])
def test_invalid_declarations_fail_closed_without_hiding_legacy_plugins(plugin_host, caplog, field, value):
    client, headers, install, configure, server = plugin_host
    install("invalid", {field: value})
    install("legacy", {})
    configure(["invalid", "legacy"])
    discovered = server._get_dashboard_plugins(force_rescan=True)
    assert "invalid" not in {p["name"] for p in discovered}
    assert "legacy" in {p["name"] for p in discovered}
    response = client.get("/api/dashboard/plugins", headers=headers)
    assert response.status_code == 200
    assert "invalid" not in {p["name"] for p in response.json()}
    assert "legacy" in {p["name"] for p in response.json()}
    assert client.get("/dashboard-plugins/invalid/dist/index.js").status_code == 404
    assert client.get("/dashboard-plugins/legacy/dist/index.js").status_code == 200
    assert "invalid" in caplog.text
    assert field in caplog.text
    assert "must" in caplog.text
