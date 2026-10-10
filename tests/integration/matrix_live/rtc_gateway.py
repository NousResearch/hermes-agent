"""Run the real gateway with deterministic speech providers for native RTC tests."""

from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path


def tone(frequency: int, sample_rate: int, seconds: float) -> bytes:
    return b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * frequency * index / sample_rate)))
                    for index in range(int(sample_rate * seconds)))


def audio_summary(pcm: bytes, sample_rate: int) -> dict:
    import numpy as np
    samples = np.frombuffer(pcm, dtype="<i2").astype(float)
    if len(samples) < sample_rate // 4:
        return {"sample_rate": sample_rate, "duration": len(samples) / sample_rate,
                "rms": 0, "frequency": 0}
    window = samples[-min(len(samples), sample_rate):]
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
    frequency = np.argmax(spectrum) * sample_rate / len(window)
    return {"sample_rate": sample_rate, "duration": len(samples) / sample_rate,
            "rms": float(np.sqrt(np.mean(window ** 2))), "frequency": float(frequency)}


def main() -> None:
    from hermes_constants import get_hermes_home
    import tools.transcription_tools
    import tools.tts_tool

    def transcribe(file_path, **kwargs):
        with wave.open(file_path, "rb") as wav:
            summary = audio_summary(wav.readframes(wav.getnframes()), wav.getframerate())
            summary["channels"] = wav.getnchannels()
        home = get_hermes_home()
        receipts = home / "rtc-stt.json"
        records = json.loads(receipts.read_text()) if receipts.exists() else []
        records.append({**summary, "home": str(home)})
        receipts.write_text(json.dumps(records))
        assert summary["channels"] == 1 and summary["sample_rate"] == 16000, summary
        assert abs(summary["frequency"] - 660) < 12 and summary["rms"] > 500, summary
        return {"success": True, "transcript": "Describe the room's current call."}

    def synthesize(text, output_path, **kwargs):
        with wave.open(output_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)
            wav.writeframes(tone(880, 48000, 1.2))
        return json.dumps({"success": True, "file_path": output_path})

    tools.transcription_tools.transcribe_audio = transcribe
    tools.tts_tool.text_to_speech_tool = synthesize
    tools.tts_tool.check_tts_requirements = lambda: True
    sys.argv = ["hermes", "gateway", "run"]
    from hermes_cli.main import main as gateway_main
    gateway_main()


if __name__ == "__main__":
    main()
