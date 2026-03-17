from __future__ import annotations

import wave
from pathlib import Path


def write_wav(path: str | Path, samples, sample_rate: int) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to write NeuTTS audio output") from exc

    data = np.asarray(samples, dtype=np.float32).flatten()
    clipped = np.clip(data, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())

    return output_path
