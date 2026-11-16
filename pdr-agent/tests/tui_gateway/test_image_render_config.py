"""``image.render`` reads ``display.inline_images`` through the gateway's real config loader."""

import pytest

from tui_gateway import server

PIL = pytest.importorskip("PIL.Image")


def _call(params):
    response = server.handle_request({"id": "img", "method": "image.render", "params": params})
    assert "error" not in response, response
    return response["result"]


@pytest.mark.parametrize(("value", "expected"), [("off", "off"), ("unicode", "unicode")])
def test_config_yaml_inline_images_reaches_image_render(tmp_path, monkeypatch, value, expected):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(f"display:\n  inline_images: {value}\n")
    monkeypatch.setattr(server, "_hermes_home", home)  # the launch home, bound at import
    photo = tmp_path / "big.png"
    PIL.new("RGBA", (4096, 4096), (10, 20, 30, 255)).save(photo)

    result = _call({"path": str(photo), "cols": 5000, "rows": 5000})

    assert result["mode"] == expected
    if expected == "unicode":  # a preview, whatever budget the caller asked for
        assert result["cols"] <= 64 and result["rows"] <= 16
