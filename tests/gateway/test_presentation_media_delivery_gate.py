from gateway.platforms import base


def _clear_fake_gate(monkeypatch):
    monkeypatch.delitem(__import__('sys').modules, 'telegram_core_presentation_gate', raising=False)


def test_presentation_gate_blocks_disallowed_ppt(monkeypatch, tmp_path):
    _clear_fake_gate(monkeypatch)
    gate_dir = tmp_path / "gate"
    gate_dir.mkdir()
    (gate_dir / "telegram_core_presentation_gate.py").write_text(
        "def allow_media_delivery(path, channel):\n"
        "    assert channel == 'telegram'\n"
        "    return False, 'blocked-for-test'\n"
    )
    ppt = tmp_path / "deck.pptx"
    ppt.write_bytes(b"ppt")

    monkeypatch.setattr(base, "_PRESENTATION_GATE_PATH", str(gate_dir))

    assert base.validate_media_delivery_path(str(ppt)) is None


def test_presentation_gate_allows_non_presentation_files_when_gate_missing(monkeypatch, tmp_path):
    _clear_fake_gate(monkeypatch)
    text = tmp_path / "notes.txt"
    text.write_text("hello")

    monkeypatch.setattr(base, "_PRESENTATION_GATE_PATH", str(tmp_path / "missing"))

    assert base.validate_media_delivery_path(str(text)) == str(text.resolve())


def test_presentation_gate_fails_closed_for_presentation_pdf_when_gate_errors(monkeypatch, tmp_path):
    _clear_fake_gate(monkeypatch)
    pdf = tmp_path / "verified_final_deck.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    monkeypatch.setattr(base, "_PRESENTATION_GATE_PATH", str(tmp_path / "missing"))

    assert base.validate_media_delivery_path(str(pdf)) is None
