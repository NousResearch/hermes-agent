from gateway.platforms.base import BasePlatformAdapter


def test_extract_media_strips_placeholder_without_delivery():
    media, cleaned = BasePlatformAdapter.extract_media(
        "Reporte listo. MEDIA:/absolute/path/to/chart.png Fin."
    )

    assert media == []
    assert "MEDIA:" not in cleaned
    assert "/absolute/path" not in cleaned
    assert cleaned == "Reporte listo.  Fin."


def test_extract_media_strips_extensionless_placeholder_without_delivery():
    media, cleaned = BasePlatformAdapter.extract_media(
        "Ejemplo: MEDIA:/absolute/path/to/file listo"
    )

    assert media == []
    assert "MEDIA:" not in cleaned
    assert "/absolute/path" not in cleaned
    assert cleaned == "Ejemplo: listo"


def test_extract_media_keeps_real_looking_missing_path_for_warning():
    media, cleaned = BasePlatformAdapter.extract_media(
        "Adjunto MEDIA:/tmp/generated-chart.png"
    )

    assert media == [("/tmp/generated-chart.png", False)]
    assert "MEDIA:" not in cleaned


def test_placeholder_detector_covers_spanish_and_file_urls():
    assert BasePlatformAdapter._is_placeholder_media_path("/ruta/absoluta.png")
    assert BasePlatformAdapter._is_placeholder_media_path("file:///absolute/path.png")
    assert not BasePlatformAdapter._is_placeholder_media_path("/tmp/real-output.png")


def test_placeholder_media_inside_protected_spans_stays_visible_text():
    media, cleaned = BasePlatformAdapter.extract_media(
        "```text\nMEDIA:/absolute/path/to/chart.png\n```\n> MEDIA:/ruta/absoluta.png"
    )

    assert media == []
    assert "MEDIA:/absolute/path/to/chart.png" in cleaned
    assert "MEDIA:/ruta/absoluta.png" in cleaned
