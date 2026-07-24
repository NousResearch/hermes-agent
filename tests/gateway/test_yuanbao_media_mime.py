from gateway.platforms.yuanbao_media import get_image_format, is_image


def test_yuanbao_image_mime_parameters_are_ignored():
    assert is_image("upload.bin", "image/gif; charset=binary") is True
    assert get_image_format("image/gif; charset=binary") == 2
    assert get_image_format(" IMAGE/PNG ; charset=utf-8") == 3
