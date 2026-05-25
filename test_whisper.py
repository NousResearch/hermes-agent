from faster_whisper import WhisperModel

model = WhisperModel(
    "Systran/faster-whisper-large-v3",  # oppure "large-v3"
    device="cuda",                      # o "cpu"
    compute_type="float16"             # su CPU spesso "int8" o "int8_float16"
)

segments, info = model.transcribe("audio.mp3", beam_size=5)

for segment in segments:
    print(f"[{segment.start:.2f} -> {segment.end:.2f}] {segment.text}")