"""``FILE:`` must alias ``MEDIA:`` on the extension-less / unknown-extension path too.

Such paths go through MEDIA_EXTENSIONLESS_TAG_RE and deliver only after
``validate_media_delivery_path`` accepts them. If only the known-extension pattern learned
the alias, ``FILE:`` + an unknown suffix extracted as nothing and the raw tag was printed to
the user, while the same path tagged ``MEDIA:`` delivered.
"""

from gateway.platforms.base import BasePlatformAdapter


def _extract(text):
    media, cleaned = BasePlatformAdapter.extract_media(text)
    return [path for path, _ in media], cleaned


def test_file_and_media_agree_on_unknown_extension(tmp_path):
    rom = tmp_path / "ROMD82T4.30"  # ``.30`` is not a deliverable extension
    rom.write_bytes(b"\xff" * 16)

    via_media = _extract(f"a\n\nMEDIA:{rom}\n\nb")
    via_file = _extract(f"a\n\nFILE:{rom}\n\nb")

    assert via_media == ([str(rom)], "a\n\nb")
    assert via_file == via_media


def test_file_alias_on_extensionless_basename_with_spaces(tmp_path):
    dockerfile = tmp_path / "my build" / "Dockerfile"
    dockerfile.parent.mkdir()
    dockerfile.write_text("FROM scratch\n")

    assert _extract(f"x\n\nFILE:{dockerfile}\n\ny") == ([str(dockerfile)], "x\n\ny")


def test_glued_file_tag_does_not_absorb_the_next_path(tmp_path):
    first = tmp_path / "a.weird"
    second = tmp_path / "b.weird"
    first.write_text("a")
    second.write_text("b")

    assert _extract(f"MEDIA:{first}FILE:{second}")[0] == [str(first), str(second)]


def test_nonexistent_path_stays_visible(tmp_path):
    missing = tmp_path / "NOPE1.99"
    text = f"x\n\nFILE:{missing}\n\ny"

    assert _extract(text) == ([], text)
