"""One ``MEDIA:`` keyword listing several references delivers every file (openclaw#150958/#163448/#163627)."""

import os

import pytest

from gateway.platforms.base import BasePlatformAdapter


@pytest.fixture
def files(tmp_path):
    paths = {}
    for name in ("first image.png", "second image.png", "a.png", "b.png", "notes.log"):
        p = tmp_path / name
        p.write_bytes(b"x")
        paths[name] = str(p)
    return paths


def _names(media):
    return [os.path.basename(path) for path, _voice in media]


def test_list_shaped_media_tags_deliver_every_reference_in_order(files):
    f1, f2, a, b, log = (files[k] for k in ("first image.png", "second image.png", "a.png", "b.png", "notes.log"))
    cases = {
        f'MEDIA:"{f1}" "{f2}"': ["first image.png", "second image.png"],
        f'Done: MEDIA:"{f1}", "{f2}".': ["first image.png", "second image.png"],
        f'MEDIA:["{a}", "{b}"]': ["a.png", "b.png"],
        f'MEDIA:["{log}","{a}"]': ["notes.log", "a.png"],  # unknown ext validates on disk
        f"MEDIA:`{a}` `{b}`": ["a.png", "b.png"],
        f"MEDIA:{a}, {b}": ["a.png", "b.png"],
    }
    for text, expected in cases.items():
        media, cleaned = BasePlatformAdapter.extract_media(text)
        assert _names(media) == expected, text
        assert "MEDIA:" not in cleaned and ".png" not in cleaned, (text, cleaned)
    # The streamed/display seam agrees with delivery: nothing of the list leaks as text.
    assert BasePlatformAdapter.strip_media_directives_for_display(f'MEDIA:["{a}", "{b}"] done').strip() == "done"


def test_non_list_payloads_keep_single_reference_reading(files, tmp_path):
    f1, a = files["first image.png"], files["a.png"]
    apostrophe = tmp_path / "team's.png"
    apostrophe.write_bytes(b"x")
    # A spaced bare path is one reference, not two.
    assert _names(BasePlatformAdapter.extract_media(f"MEDIA:{f1}")[0]) == ["first image.png"]
    # A comma inside one quoted name stays inside it.
    comma = tmp_path / "Hello, World.png"
    comma.write_bytes(b"x")
    assert _names(BasePlatformAdapter.extract_media(f'MEDIA:"{comma}"')[0]) == ["Hello, World.png"]
    # Prose after a single reference is prose, and the second quoted token of a list never
    # absorbs an apostrophe-bearing neighbour into a welded path.
    media, cleaned = BasePlatformAdapter.extract_media(f'MEDIA:"{a}", the first one')
    assert _names(media) == ["a.png"] and "the first one" in cleaned
    media, cleaned = BasePlatformAdapter.extract_media(f"MEDIA:'{apostrophe}' '{a}'")
    assert "team's.png" not in _names(media) and "team's.png" in cleaned
    # Examples inside code are never rewritten or delivered.
    example = f'```\nMEDIA:["{a}", "{f1}"]\n```\nreal: MEDIA:{a}'
    media, cleaned = BasePlatformAdapter.extract_media(example)
    assert _names(media) == ["a.png"] and f'MEDIA:["{a}", "{f1}"]' in cleaned
