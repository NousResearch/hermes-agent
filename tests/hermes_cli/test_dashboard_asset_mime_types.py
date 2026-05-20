import mimetypes


def test_dashboard_registers_javascript_mime_type(monkeypatch):
    from hermes_cli.web_server import _register_dashboard_asset_mime_types

    monkeypatch.setitem(mimetypes.types_map, ".js", "text/plain")

    _register_dashboard_asset_mime_types()

    assert mimetypes.guess_type("/assets/index-abc123.js")[0] == "application/javascript"


def test_dashboard_registers_wasm_mime_type(monkeypatch):
    from hermes_cli.web_server import _register_dashboard_asset_mime_types

    monkeypatch.delitem(mimetypes.types_map, ".wasm", raising=False)

    _register_dashboard_asset_mime_types()

    assert mimetypes.guess_type("/assets/module.wasm")[0] == "application/wasm"
