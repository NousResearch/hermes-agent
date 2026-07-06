"""JARVIS v2 เฟส 2 — คุยเสียงจริงกับ Gemini Live + ต่อสายใหม่อัตโนมัติเมื่อหลุด.

รันด้วย: scripts/jarvis-voice/.venv-gemini/bin/python jarvis_live.py
พูดใส่ไมค์ได้เลย · กด Ctrl-C เพื่อหยุด.
"""
import asyncio
import os
import queue
import signal
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from google import genai
from google.genai import types

# P6-I1: สลับรุ่นได้ด้วย env — ทดลอง 3.1 ด้วย JARVIS_MODEL=gemini-3.1-flash-live-preview (สลับกลับได้ทันที)
MODEL = os.getenv("JARVIS_MODEL", "gemini-2.5-flash-native-audio-latest")
IN_RATE, OUT_RATE, BLOCK = 16000, 24000, 1600
SYSTEM = "คุณคือจาร์วิส ผู้ช่วยเสียงภาษาไทยของเจ้านาย ตอบสั้น กระชับ เป็นธรรมชาติ เป็นภาษาไทยเสมอ"
# รอเงียบกี่ ms ถึงถือว่า "พูดจบ" — ต่ำเกินไปจะตอบสวนตอนภาษาไทยเว้นวรรค
def _read_vad_silence_ms() -> int:
    try:
        return max(100, int(os.getenv("JARVIS_VAD_SILENCE_MS", "900")))
    except ValueError:
        return 900


VAD_SILENCE_MS = _read_vad_silence_ms()


def load_key() -> str:
    for line in (Path(__file__).resolve().parent / ".env").read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("ไม่พบ GEMINI_API_KEY ใน .env")


def build_config(handle, manual_activity=False):
    activity_detection = types.AutomaticActivityDetection(
        disabled=True,
    ) if manual_activity else types.AutomaticActivityDetection(
        # HIGH เป็นค่าเริ่มต้นของ Live อยู่แล้ว — ใส่ชัดๆ กัน default เปลี่ยนภายหลัง
        end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
        silence_duration_ms=VAD_SILENCE_MS,
    )
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        session_resumption=types.SessionResumptionConfig(handle=handle),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=activity_detection,
            activity_handling=types.ActivityHandling.NO_INTERRUPTION,
            turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
        ),
    )


class Audio:
    """ไมค์เข้า (callback→คิว) และลำโพงออก (คิว→callback) แยกจาก event loop."""

    def __init__(self):
        self.mic_q: "queue.Queue[bytes]" = queue.Queue()
        self.play_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._buf = np.zeros(0, dtype=np.int16)

    def _mic_cb(self, indata, frames, t, status):
        self.mic_q.put(bytes(indata))

    def _spk_cb(self, outdata, frames, t, status):
        while self._buf.size < frames and not self.play_q.empty():
            self._buf = np.concatenate([self._buf, self.play_q.get_nowait()])
        take = min(frames, self._buf.size)
        outdata[:take, 0] = self._buf[:take]
        if take < frames:
            outdata[take:, 0] = 0
        self._buf = self._buf[take:]

    def playing(self) -> bool:
        """ลำโพงกำลังมีเสียงค้างเล่นอยู่ไหม (ใช้กันเสียงย้อนเข้าไมค์ปนเป็นเสียง user)."""
        return self._buf.size > 0 or not self.play_q.empty()

    def clear_playback(self):
        """ทิ้งเสียงตอบที่ค้างคิว — ใช้ตอน user พูดแทรก จะได้ไม่พูดทับต่อ (Codex #3)."""
        while not self.play_q.empty():
            try:
                self.play_q.get_nowait()
            except queue.Empty:
                break
        self._buf = np.zeros(0, dtype=np.int16)

    @staticmethod
    def _resolve_device(env_name, kind):
        """หาอุปกรณ์เสียงจากชื่อใน env (จับแบบมีคำนั้นอยู่ในชื่อ) — ไม่ตั้ง = ใช้ค่าเริ่มต้นระบบ.

        มีไว้เพราะเครื่องที่ลงอุปกรณ์เสียงเสมือน (BoomAudio/Lark) อาจถูกตั้งเป็นค่าเริ่มต้น
        แล้วเสียงตอบหายเข้ากล่องดำเงียบๆ — ชี้ลำโพงจริงด้วย JARVIS_OUT_DEVICE / ไมค์ด้วย JARVIS_IN_DEVICE
        """
        name = os.getenv(env_name, "").strip()
        if not name:
            return None
        key = "max_output_channels" if kind == "output" else "max_input_channels"
        for i, d in enumerate(sd.query_devices()):
            if d[key] > 0 and name.lower() in d["name"].lower():
                print(f"[audio] {env_name} → ใช้ '{d['name']}'", file=sys.stderr, flush=True)
                return i
        print(f"[audio] ไม่พบอุปกรณ์ '{name}' — ใช้ค่าเริ่มต้นระบบแทน", file=sys.stderr, flush=True)
        return None

    def streams(self):
        return (
            sd.InputStream(samplerate=IN_RATE, channels=1, dtype="int16",
                           blocksize=BLOCK, callback=self._mic_cb,
                           device=self._resolve_device("JARVIS_IN_DEVICE", "input")),
            sd.OutputStream(samplerate=OUT_RATE, channels=1, dtype="int16",
                            blocksize=BLOCK, callback=self._spk_cb,
                            device=self._resolve_device("JARVIS_OUT_DEVICE", "output")),
        )


async def run_session(client, audio, handle, stop):
    async with client.aio.live.connect(model=MODEL, config=build_config(handle)) as session:
        print("[live] เชื่อมแล้ว — พูดได้เลยครับ", file=sys.stderr, flush=True)
        loop = asyncio.get_running_loop()

        async def send():
            while not stop.is_set():
                try:  # timeout กันค้างที่ get() หลัง stop/หลุด (Codex review #1)
                    data = await loop.run_in_executor(None, audio.mic_q.get, True, 0.2)
                except queue.Empty:
                    continue
                await session.send_realtime_input(
                    audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={IN_RATE}"))

        async def recv():
            new_handle = handle
            async for msg in session.receive():
                sc = getattr(msg, "server_content", None)
                if sc is not None and getattr(sc, "interrupted", None):
                    if not audio.playing():
                        audio.clear_playback()
                if getattr(msg, "data", None):
                    audio.play_q.put(np.frombuffer(msg.data, dtype=np.int16))
                upd = getattr(msg, "session_resumption_update", None)
                if upd is not None and getattr(upd, "new_handle", None):
                    new_handle = upd.new_handle
            return new_handle

        send_task = asyncio.create_task(send())
        try:
            return await recv()
        finally:
            send_task.cancel()


async def main() -> int:
    client = genai.Client(api_key=load_key())
    audio = Audio()
    stop = asyncio.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    mic_stream, spk_stream = audio.streams()
    handle = None
    fail = 0
    with mic_stream, spk_stream:
        while not stop.is_set():
            try:
                handle = await run_session(client, audio, handle, stop)
                fail = 0  # ปิดสายปกติ (Gemini ตัดที่ 10-15 นาที) → ต่อใหม่ทันทีด้วย handle เดิม
                print("[live] Gemini ปิดสาย — ต่อใหม่ทันที", file=sys.stderr, flush=True)
            except Exception as exc:
                fail += 1
                if fail > 8:  # หลุดรัวจริง (เน็ต/กุญแจ/โควตาวันเต็ม) → หยุด ไม่ยิง API รัว (Codex review #2)
                    print("[live] หลุดซ้ำเกิน 8 ครั้ง — หยุด · ตรวจเน็ต/กุญแจ/โควตา", file=sys.stderr, flush=True)
                    break
                delay = min(2 ** fail, 30)  # ถอยหลังเพิ่มทีละช่วง เพดาน 30 วิ
                text = str(exc)
                why = "โควตาฟรีเต็มชั่วคราว (429)" if ("429" in text or "RESOURCE_EXHAUSTED" in text) else f"หลุด: {text[:100]}"
                print(f"[live] {why} — รอ {delay} วิ แล้วต่อใหม่เอง", file=sys.stderr, flush=True)
                await asyncio.sleep(delay)
    print("[live] ปิดแล้ว ลาก่อนครับ", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(0)
