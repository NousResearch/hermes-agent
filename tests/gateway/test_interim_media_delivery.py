"""Regression coverage for current-turn media emitted in interim commentary."""

from gateway.platforms.base import BasePlatformAdapter
from gateway.run import _build_interim_media_delivery_payload


def test_interim_media_is_preserved_and_final_duplicate_is_deduplicated(tmp_path):
    interim_path = tmp_path / "interim.png"
    final_path = tmp_path / "final.png"
    interim_path.write_bytes(b"interim")
    final_path.write_bytes(b"final")

    payload, cleaned_final = _build_interim_media_delivery_payload(
        [f"first\nMEDIA:{interim_path}"],
        f"done\nMEDIA:{interim_path}\nMEDIA:{final_path}",
        BasePlatformAdapter,
    )

    assert payload is not None
    media, _cleaned_payload = BasePlatformAdapter.extract_media(payload)
    assert media == [
        (str(interim_path), False),
        (str(final_path), False),
    ]
    assert cleaned_final == "done"


def test_final_only_media_keeps_established_delivery_rail(tmp_path):
    final_path = tmp_path / "final.png"
    final_path.write_bytes(b"final")
    final_response = f"done\nMEDIA:{final_path}"

    payload, cleaned_final = _build_interim_media_delivery_payload(
        ["commentary without media"],
        final_response,
        BasePlatformAdapter,
    )

    assert payload is None
    assert cleaned_final == final_response