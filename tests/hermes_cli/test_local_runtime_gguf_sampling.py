"""Publisher sampling metadata must survive managed-local preset generation."""

import struct

from hermes_cli.local_runtime.gguf import read_gguf_header


def test_gguf_penalty_repeat_uses_the_repeat_penalty_preset_key(tmp_path):
    """Liquid's GGUFs use llama.cpp's canonical penalty_repeat metadata spelling."""
    path = tmp_path / "sampling.gguf"
    key = b"general.sampling.penalty_repeat"
    path.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
        + struct.pack("<Q", len(key)) + key + struct.pack("<If", 6, 1.1)
    )

    assert read_gguf_header(path).sampling_defaults == {"repeat-penalty": "1.1"}
