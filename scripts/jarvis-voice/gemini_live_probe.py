import asyncio
import sys
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash-native-audio-latest"
OUT_WAV = "/tmp/gemini-probe.wav"
OUT_RATE = 24000  # Gemini Live native audio ส่งกลับ PCM 24kHz mono 16-bit


def load_key() -> str:
    env = Path(__file__).resolve().parent / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("ไม่พบ GEMINI_API_KEY ใน .env")


async def probe() -> int:
    client = genai.Client(api_key=load_key())
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction="คุณคือจาร์วิส ตอบสั้นๆ เป็นภาษาไทย 1 ประโยค",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    turns_text = ["สวัสดีครับ ทักทายกลับสั้นๆ", "สองบวกสองเท่ากับเท่าไหร่", "ขอบคุณครับ ตอบสั้นๆ"]
    audio = bytearray()
    results = []
    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            for idx, text in enumerate(turns_text):
                ttfb = None
                t0 = time.monotonic()
                await session.send_client_content(
                    turns=types.Content(role="user", parts=[types.Part(text=text)]),
                    turn_complete=True,
                )
                async for message in session.receive():
                    data = getattr(message, "data", None)
                    if data:
                        if ttfb is None:
                            ttfb = time.monotonic() - t0
                        if idx == 0:
                            audio.extend(data)
                    sc = getattr(message, "server_content", None)
                    if sc is not None and getattr(sc, "turn_complete", False):
                        break
                label = "เย็น(ครั้งแรก)" if idx == 0 else f"อุ่น(ครั้งที่{idx+1})"
                results.append((label, ttfb))
                print(f"เสียงแรก {label}: {ttfb:.2f} วินาที" if ttfb else f"{label}: ไม่มีเสียง")
    except Exception as exc:
        text = str(exc)
        if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
            print("โควตาฟรีเต็ม (429) — รุ่นเสียงสดถูกจำกัดบน free tier")
        else:
            print(f"ต่อ Gemini Live ไม่สำเร็จ: {text[:200]}")
        return 1

    if not audio:
        print("เชื่อมได้แต่ไม่มีเสียงกลับมา (อาจถูกจำกัด modality บน free tier)")
        return 1

    with wave.open(OUT_WAV, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(OUT_RATE)
        w.writeframes(bytes(audio))

    warm = [t for lbl, t in results if "อุ่น" in lbl and t]
    if warm:
        print(f"เฉลี่ยตอนอุ่นแล้ว: {sum(warm)/len(warm):.2f} วินาที")
    return 0


def main() -> int:
    try:
        return asyncio.run(asyncio.wait_for(probe(), timeout=60))
    except asyncio.TimeoutError:
        print("หมดเวลา 60 วินาที — ไม่มีเสียงตอบกลับ")
        return 1


if __name__ == "__main__":
    sys.exit(main())
