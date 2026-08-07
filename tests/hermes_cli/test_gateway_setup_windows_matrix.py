from types import SimpleNamespace

from hermes_cli import gateway as gateway_mod


def test_matrix_stays_visible_but_unavailable_on_native_windows(monkeypatch):
    monkeypatch.setattr(gateway_mod.sys, "platform", "win32")
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", lambda: None)

    matrix_entry = SimpleNamespace(
        name="matrix",
        label="Matrix",
        emoji="🟩",
        required_env=["MATRIX_ACCESS_TOKEN"],
        install_hint="",
    )
    monkeypatch.setattr(
        "gateway.platform_registry.platform_registry.all_entries",
        lambda: [matrix_entry],
    )

    matrix = next(p for p in gateway_mod._all_platforms() if p["key"] == "matrix")

    assert matrix["label"] == "Matrix"
    assert matrix["token_var"] == "MATRIX_ACCESS_TOKEN"
    assert "python-olm" in matrix["unavailable_reason"]
    assert gateway_mod._platform_status(matrix) == "unavailable on this OS"


def test_selecting_windows_matrix_row_explains_wsl_path(capsys):
    platform = gateway_mod._windows_matrix_placeholder()

    gateway_mod._configure_platform(platform)

    out = capsys.readouterr().out
    assert "Matrix" in out
    assert "native Windows" in out
    assert "WSL" in out
