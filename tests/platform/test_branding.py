"""The tenant's own identity: company details, brand, and the logo.

Both files behind every branded surface were editable only by editing YAML on the server.
These cover the seam that changed that, and in particular the two things that are easy to
get wrong once a control plane can write a bundle: the tenant id must not be renameable,
and an uploaded image must not be able to become something the control plane serves as
active content from its own origin.
"""

from __future__ import annotations

import base64
import shutil
import struct
import zlib
from pathlib import Path

import pytest

from nova import branding
from nova.audit import AuditLog
from nova.control import ControlAPI
from nova.control.auth import Principal
from nova.errors import SpecError
from nova.runtime import get_runtime
from nova.spec import load_bundle

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "nova" / "examples" / "acme"
ADMIN = Principal(name="ops", role="admin")
VIEWER = Principal(name="watcher", role="viewer")


def png_bytes(width: int = 8, height: int = 8) -> bytes:
    """A real, minimal PNG. Generated rather than committed so the test carries no binary."""
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


@pytest.fixture
def live(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    shutil.copytree(EXAMPLE, root)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NOVA_HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home))
    bundle = load_bundle(root)
    runtime = get_runtime("hermes", home=home, tenant_id=bundle.tenant_id)
    audit = AuditLog.for_home(home, tenant_id=bundle.tenant_id, actor="test")
    return {"api": ControlAPI(bundle, runtime, audit=audit), "root": root, "home": home}


def _write(api, path, principal, payload):
    return api.write(f"/platform/v1{path}", principal, payload)


def _settings(api):
    return api.handle("/platform/v1/settings").body


# -- company and brand ---------------------------------------------------------


def test_company_details_are_editable_and_persist(live):
    response = _write(live["api"], "/settings/organization", ADMIN, {
        "fields": {"legal_name": "Northwind Logistics Ltd", "region": "eu-west-2"},
    })
    assert response.status == 200, response.body

    reloaded = load_bundle(live["root"])
    assert reloaded.organization.legal_name == "Northwind Logistics Ltd"
    assert reloaded.organization.region == "eu-west-2"
    # Untouched fields survive: the form sends what changed, not the whole file.
    assert reloaded.organization.timezone == "Europe/London"


def test_the_brand_is_editable_and_reaches_the_identity_route(live):
    _write(live["api"], "/settings/identity", ADMIN, {
        "fields": {
            "product_name": "Northwind Intelligence",
            "theme": {"accent": "#0F62FE", "on_accent": "#FFFFFF"},
        },
    })
    identity = live["api"].handle("/platform/v1/identity").body
    assert identity["product_name"] == "Northwind Intelligence"
    assert identity["theme"]["accent"] == "#0F62FE"
    # The surface colour was already declared and is not in the form; it must survive a
    # partial theme write rather than being replaced away.
    assert identity["theme"]["surface"] == "#f5f7f7"


def test_the_tenant_id_cannot_be_renamed(live):
    """It is stamped on every audit event already written and names the runtime home, so a
    rename through a form would orphan the history rather than move the tenant."""
    response = _write(live["api"], "/settings/organization", ADMIN,
                      {"fields": {"tenant_id": "hijacked"}})
    assert response.status == 400
    assert "cannot be changed" in response.body["error"]["message"]
    assert load_bundle(live["root"]).tenant_id == "acme"


def test_an_agents_display_name_changes_without_touching_its_id(live):
    _write(live["api"], "/settings/agent-name", ADMIN,
           {"agent_id": "operations", "display_name": "Night Ops"})
    reloaded = load_bundle(live["root"])
    assert reloaded.identity.agent_display_names["operations"] == "Night Ops"
    assert any(a.id == "operations" for a in reloaded.agents), "the id moved"

    # Cleared by sending an empty name, rather than needing a separate route.
    _write(live["api"], "/settings/agent-name", ADMIN,
           {"agent_id": "operations", "display_name": ""})
    assert "operations" not in load_bundle(live["root"]).identity.agent_display_names


def test_an_unsettable_field_is_refused_by_name(live):
    response = _write(live["api"], "/settings/identity", ADMIN, {"fields": {"agents": {}}})
    assert response.status == 400
    assert "agents" in response.body["error"]["message"]


def test_a_viewer_may_read_the_branding_but_not_change_it(live):
    assert VIEWER.may("/settings") is True
    assert VIEWER.may_write("/settings/identity") is False
    assert _write(live["api"], "/settings/identity", VIEWER,
                  {"fields": {"product_name": "x"}}).status == 403


# -- the logo ------------------------------------------------------------------


def test_a_logo_is_stored_in_the_bundle_and_served_from_this_origin(live):
    """Served rather than linked: the Control Centre runs under `img-src 'self' data:`, so
    a hosted URL would be blocked by the browser and show as a broken image."""
    response = _write(live["api"], "/settings/logo", ADMIN, {
        "kind": "logo", "content_type": "image/png",
        "data": base64.b64encode(png_bytes()).decode(),
    })
    assert response.status == 200, response.body
    assert "branding/logo.png" in response.body["files_changed"]
    assert (live["root"] / "branding" / "logo.png").is_file()
    assert load_bundle(live["root"]).identity.logo == "branding/logo.png"

    served = live["api"].handle("/platform/v1/branding/logo")
    assert served.status == 200
    assert served.content_type == "image/png"
    assert served.raw.startswith(b"\x89PNG")


def test_an_svg_logo_is_refused(live):
    """An SVG can carry script, and this image is served from the control plane's own
    origin — where it would run with the control plane's privileges."""
    response = _write(live["api"], "/settings/logo", ADMIN, {
        "kind": "logo", "content_type": "image/svg+xml",
        "data": base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"/>').decode(),
    })
    assert response.status == 400
    assert "svg" in response.body["error"]["message"].lower()
    assert not (live["root"] / "branding").exists()


def test_bytes_that_are_not_the_declared_type_are_refused(live):
    """A caller claiming image/png while sending HTML would otherwise get a file the
    control plane serves from its own origin."""
    for content_type, payload in (
        ("image/png", b"<html><script>alert(1)</script></html>"),
        ("image/jpeg", b"not a jpeg at all"),
        ("image/webp", b"RIFFxxxxNOPE"),
    ):
        response = _write(live["api"], "/settings/logo", ADMIN, {
            "kind": "logo", "content_type": content_type,
            "data": base64.b64encode(payload).decode(),
        })
        assert response.status == 400, f"{content_type} accepted the wrong bytes"


def test_an_oversized_logo_is_refused(live):
    response = _write(live["api"], "/settings/logo", ADMIN, {
        "kind": "logo", "content_type": "image/png",
        "data": base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"x" * 2_000_000).decode(),
    })
    assert response.status == 400
    assert "kB" in response.body["error"]["message"]


def test_replacing_a_logo_with_another_format_leaves_no_orphan(live):
    api = live["api"]
    _write(api, "/settings/logo", ADMIN, {
        "kind": "logo", "content_type": "image/png",
        "data": base64.b64encode(png_bytes()).decode()})
    _write(api, "/settings/logo", ADMIN, {
        "kind": "logo", "content_type": "image/gif",
        "data": base64.b64encode(b"GIF89a" + b"\x00" * 20).decode()})

    stored = sorted(p.name for p in (live["root"] / "branding").iterdir())
    assert stored == ["logo.gif"], f"an old format was left behind: {stored}"
    assert load_bundle(live["root"]).identity.logo == "branding/logo.gif"


def test_clearing_a_logo_removes_the_file_and_the_reference(live):
    api = live["api"]
    _write(api, "/settings/logo", ADMIN, {
        "kind": "logo", "content_type": "image/png",
        "data": base64.b64encode(png_bytes()).decode()})
    _write(api, "/settings/logo", ADMIN, {"kind": "logo", "data": ""})

    assert not load_bundle(live["root"]).identity.logo
    assert api.handle("/platform/v1/branding/logo").status == 404
    assert not (live["root"] / "branding" / "logo.png").exists()


def test_a_logo_path_escaping_the_bundle_is_not_served(live):
    """The path comes from a YAML file an operator edits, which is not the same as a path
    NOVA wrote."""
    import yaml

    secret = live["root"].parent / "outside.png"
    secret.write_bytes(png_bytes())
    path = live["root"] / "identity.yaml"
    document = yaml.safe_load(path.read_text()) or {}
    document["logo"] = "../outside.png"
    path.write_text(yaml.safe_dump(document))

    api = ControlAPI(load_bundle(live["root"]), live["api"].runtime, audit=live["api"].audit)
    assert api.handle("/platform/v1/branding/logo").status == 404


def test_the_bundle_writer_handles_binary_files(live):
    """Regression: `_sync` decoded every changed file as UTF-8, so the first logo upload
    died on the PNG header. A bundle is not only YAML and Markdown."""
    from nova.spec.writer import edit

    raw = png_bytes()

    def mutate(e):
        e.write_bytes("branding/probe.png", raw)

    edit(live["root"], mutate)
    assert (live["root"] / "branding" / "probe.png").read_bytes() == raw


# -- audit ---------------------------------------------------------------------


def test_branding_changes_are_audited_without_the_image(live):
    import json

    _write(live["api"], "/settings/logo", ADMIN, {
        "kind": "logo", "content_type": "image/png",
        "data": base64.b64encode(png_bytes()).decode()})

    text = (live["home"] / "nova" / "audit.jsonl").read_text()
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    logo = [r for r in records if r.get("kind") == "settings.logo_changed"]
    assert [r["phase"] for r in logo] == ["intent", "committed"]
    assert {r["actor"] for r in logo} == {"ops"}
    # The image is not copied into the log — only that one of this type was stored.
    assert "iVBOR" not in text and "\x89PNG" not in text


def test_a_malformed_settings_body_is_refused(live):
    api = live["api"]
    assert _write(api, "/settings/organization", ADMIN, {}).status == 400
    assert _write(api, "/settings/identity", ADMIN, {"fields": {}}).status == 400
    assert _write(api, "/settings/logo", ADMIN,
                  {"kind": "logo", "data": 42, "content_type": "image/png"}).status == 400
    assert _write(api, "/settings/logo", ADMIN, {"kind": "elsewhere", "data": ""}).status == 400


def test_the_settings_read_says_what_is_settable_and_what_is_not(live):
    body = _settings(live["api"])
    assert body["immutable"] == ["tenant_id"]
    assert "legal_name" in body["settable"]["organization"]
    assert "theme" in body["settable"]["identity"]
    assert body["organization"]["tenant_id"] == "acme"
    assert body["logo"] == {"logo": False, "favicon": False}


# -- values that would be wrong everywhere they are used are refused -------------------


@pytest.mark.parametrize("path, fields, words", [
    ("/settings/identity", {"support": {"url": "javascript:alert(1)"}}, "https://"),
    ("/settings/identity", {"support": {"url": "data:text/html,<script>1</script>"}}, "https://"),
    ("/settings/identity", {"support": {"email": "not-an-email"}}, "email address"),
    ("/settings/organization", {"timezone": "Mars/Olympus"}, "time zone"),
    ("/settings/organization", {"contact_email": "platform at acme"}, "email address"),
])
def test_a_value_that_is_wrong_everywhere_is_refused_by_name(live, path, fields, words):
    """Found sweeping the Settings screen: each of these saved, and a javascript: support
    link would run code the day anything renders it as a link."""
    before = _settings(live["api"])
    response = _write(live["api"], path, ADMIN, {"fields": fields})
    assert response.status == 400, response.body
    assert words in response.body["error"]["message"]
    assert _settings(live["api"]) == before, "a refused edit must change nothing"


@pytest.mark.parametrize("path, fields", [
    ("/settings/identity", {"support": {"url": "https://help.northwind.example/support"}}),
    ("/settings/identity", {"support": {"url": "mailto:help@northwind.example"}}),
    ("/settings/organization", {"timezone": "America/New_York", "contact_email": "ops@northwind.example"}),
])
def test_ordinary_contact_details_still_save(live, path, fields):
    assert _write(live["api"], path, ADMIN, {"fields": fields}).status == 200


def test_a_blank_product_name_falls_back_to_the_default_rather_than_an_empty_title(live):
    from nova.spec.identity import DEFAULT_PRODUCT_NAME

    assert _write(live["api"], "/settings/identity", ADMIN, {"fields": {"product_name": "  "}}).status == 200
    assert live["api"].handle("/platform/v1/identity").body["product_name"] == DEFAULT_PRODUCT_NAME


def test_a_bundle_already_holding_such_a_value_still_loads(live):
    """The checks guard the form, not the file. A tenant whose bundle predates them must not
    find the control plane refusing to start after an upgrade — found when a local control
    plane, restarted on the new code, would not load a bundle an earlier test had written."""
    org = live["root"] / "organization.yaml"
    org.write_text(org.read_text().replace("Europe/London", "Mars/Olympus"))
    assert load_bundle(live["root"]).organization.timezone == "Mars/Olympus"
