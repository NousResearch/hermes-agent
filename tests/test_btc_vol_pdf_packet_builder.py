from pathlib import Path


def test_pdf_packet_builder_does_not_hardcode_local_checkout_path():
    source = Path("scripts/build_btc_vol_pdf_packet.py").read_text(encoding="utf-8")

    assert "/Users/assistant" not in source
    assert "ROOT = Path(__file__).resolve().parents[1]" in source
