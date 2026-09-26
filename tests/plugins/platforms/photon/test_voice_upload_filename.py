from pathlib import Path


SIDECAR = (
    Path(__file__).resolve().parents[4]
    / "plugins"
    / "platforms"
    / "photon"
    / "sidecar"
    / "index.mjs"
)


def test_voice_upload_uses_m4a_name_without_overwriting_source_mime() -> None:
    source = SIDECAR.read_text(encoding="utf-8")
    voice_branch = 'if (kind === "voice") {'
    m4a_fallback = 'opts.name = name && /\\.m4a$/i.test(name) ? name : "voice.m4a";'
    mime_assignment = "if (mimeType) opts.mimeType = mimeType;"

    assert voice_branch in source
    assert m4a_fallback in source
    assert mime_assignment in source
    assert source.index(voice_branch) < source.index(m4a_fallback) < source.index(mime_assignment)
