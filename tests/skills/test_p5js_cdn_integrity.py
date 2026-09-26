"""The bundled p5.js skill ships an external executable dependency; the CDN
script tag must carry subresource integrity so the browser refuses to execute
any bytes that are not the pinned p5.js 1.11.3 build (#96888).

Both authoritative entry points are covered: the viewer template and the
bare-HTML scaffold embedded in SKILL.md. The test is deliberately a static
source assertion — SRI is enforced by the browser at load time, so the
regression we pin is "the attribute pair is present and matches the official
cdnjs digest", not runtime behavior.
"""
import re
from pathlib import Path

import pytest

_P5_SCRIPT = re.compile(r"<script\s+[^>]*>")
_P5_CDN_PREFIX = (
    "https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.11.3/p5.min.js"
)
_OFFICIAL_SRI = (
    "sha512-I0Pwwz3PPNQkWes+rcSoQqikKFfRmTfGQrcNzZbm8ALaUyJuFdyRinl805shE8xT6iEWsWgvRxdXb3yhQNXKoA=="
)

_SKILL_ROOT = Path(__file__).resolve().parents[2] / "skills" / "creative" / "p5js"
_ENTRY_POINTS = [
    _SKILL_ROOT / "templates" / "viewer.html",
    _SKILL_ROOT / "SKILL.md",
]


def _p5_cdn_tags(source: str) -> list[str]:
    return [
        tag
        for tag in _P5_SCRIPT.findall(source)
        if f'src="{_P5_CDN_PREFIX}"' in tag
    ]


@pytest.mark.parametrize("path", _ENTRY_POINTS, ids=lambda p: p.name)
def test_p5_cdn_script_carries_integrity(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tags = _p5_cdn_tags(source)
    assert tags, (
        f"{path.name} no longer references the pinned p5.js CDN script"
    )

    for tag in tags:
        assert f'integrity="{_OFFICIAL_SRI}"' in tag, (
            f"{path.name}: p5.js script tag lost its SRI attribute or the "
            "digest no longer matches the official cdnjs digest for 1.11.3"
        )
        assert 'crossorigin="anonymous"' in tag, (
            f"{path.name}: SRI requires the crossorigin attribute or the "
            "browser skips the integrity check for cross-origin scripts"
        )
