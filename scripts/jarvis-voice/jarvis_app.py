"""JARVIS v2 — เปลือกใช้งานจริง: กดปุ่มลัดเปิด/ปิดพูด + ปิดเองเมื่อเงียบนาน.

ค่าเริ่มต้นปุ่มลัด = Cmd+Shift+J (เปลี่ยนได้ด้วย env JARVIS_HOTKEY)
หมายเหตุ: ไม่ใช้ Shift+A ล้วน เพราะจะเด้งทุกครั้งที่พิมพ์ตัว A ใหญ่ และชนกับ Wispr Flow.
รันเบื้องหลังได้ด้วย jarvis-start.command (ไม่ต้องเปิด Terminal ค้าง).
"""
import asyncio
import os
import queue
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types
from pynput import keyboard, mouse

from jarvis_live import IN_RATE, MODEL, OUT_RATE, VAD_SILENCE_MS, Audio, build_config, load_key
from turn_taking import (
    LocalSpeechTurnDetector,
    TurnTakingGate,
    read_bool_env,
    read_float_env,
    read_int_env,
)

IDLE_MIN = float(os.getenv("JARVIS_IDLE_MIN", "30"))      # เงียบกี่นาทีแล้วถามจะปิด
GRACE_SEC = float(os.getenv("JARVIS_IDLE_GRACE", "20"))   # ถามแล้วรอตอบกี่วินาที
HOTKEY = os.getenv("JARVIS_HOTKEY", "<cmd>+<shift>+j")
START_ACTIVE = read_bool_env("JARVIS_START_ACTIVE")
MIC_SPEECH_RMS = read_float_env("JARVIS_MIC_SPEECH_RMS", 1000.0, minimum=0.0)
LOCAL_SILENCE_MS = read_int_env("JARVIS_LOCAL_SILENCE_MS", VAD_SILENCE_MS, minimum=100)
LOCAL_MIN_SPEECH_FRAMES = read_int_env("JARVIS_MIN_SPEECH_FRAMES", 3, minimum=1)
MANUAL_ACTIVITY = read_bool_env("JARVIS_MANUAL_ACTIVITY", True)
PLAYBACK_TAIL_SEC = read_float_env("JARVIS_PLAYBACK_TAIL_SEC", 1.5, minimum=0.0)
INTERRUPT_RMS = read_float_env("JARVIS_INTERRUPT_RMS", 2400.0, minimum=0.0)
INTERRUPT_FRAMES = read_int_env("JARVIS_INTERRUPT_FRAMES", 3, minimum=1)
VOICE_INTERRUPT_ENABLED = os.getenv("JARVIS_VOICE_INTERRUPT", "0") == "1"
# half-duplex: ตอนลำโพงพูด ปิดหูไมค์ กันเสียงตัวเองย้อนไปตัดคำตอบกลางคัน
# ถ้าเจ้านายพูดแทรกชัด ๆ ระหว่างลำโพงพูด จะหยุดเสียงตอบก่อน แล้วค่อยฟังรอบใหม่
# ค่าเริ่มต้นไม่ตัดเสียงตอบเอง เพราะเสียงลำโพงสะท้อนอาจถูกเข้าใจผิดว่าเจ้านายพูด
# อยากให้เสียงพูดแทรกหยุดจาร์วิสอัตโนมัติ → export JARVIS_VOICE_INTERRUPT=1
# อยากส่งเสียงเข้าระหว่างลำโพงพูดแบบเดิม (เสี่ยงเสียงสะท้อนมาก) → export JARVIS_BARGE_IN=1
BARGE_IN = os.getenv("JARVIS_BARGE_IN", "0") == "1"
# P1-I2: ปุ่มข้างเมาส์ = สวิตช์เปิด/ปิดพูด (ปิดด้วย JARVIS_MOUSE_TOGGLE=0)
# ปุ่มกลาง (ล้อ) ไม่ผูก — ชนกับเปิดแท็บ/วางข้อความในเบราว์เซอร์ · แต่ log ให้เห็นไว้ debug
MOUSE_TOGGLE = read_bool_env("JARVIS_MOUSE_TOGGLE", True)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class Jarvis:
    def __init__(self, loop):
        self.loop = loop
        self.active = asyncio.Event()
        self.client = genai.Client(api_key=load_key())
        self.audio = Audio()
        self.last_activity = time.monotonic()
        self.last_voice = 0.0        # เวลาเฟรมเสียงพูดล่าสุดของ user
        self.awaiting_reply = False  # true = user พูดแล้ว กำลังรอเสียงตอบก้อนแรก
        self.turn_gate = TurnTakingGate(
            playback_tail_sec=PLAYBACK_TAIL_SEC,
            interrupt_rms=INTERRUPT_RMS,
            interrupt_frames=INTERRUPT_FRAMES,
            voice_interrupt_enabled=VOICE_INTERRUPT_ENABLED,
        )

    def toggle_from_hotkey(self):
        self.loop.call_soon_threadsafe(self._toggle)

    def _beep(self, freq_hz, ms=180):
        """เสียงบอกสถานะ — เล่นผ่านท่อลำโพงเดียวกับเสียงตอบ Gemini (JARVIS_OUT_DEVICE)
        กันหายเข้าอุปกรณ์เสมือนแบบ afplay/BoomAudio (Codex #1)."""
        try:
            t = np.arange(int(OUT_RATE * ms / 1000), dtype=np.float32) / OUT_RATE
            w = np.sin(2 * np.pi * freq_hz * t) * 0.35
            n = max(1, w.size // 8)          # ขอบจางเข้า-ออก กันเสียงแตกป๊อก
            w[:n] *= np.linspace(0, 1, n, dtype=np.float32)
            w[-n:] *= np.linspace(1, 0, n, dtype=np.float32)
            self.audio.play_q.put((w * 32767).astype(np.int16))
        except Exception:
            pass

    def _toggle(self):
        now = time.monotonic()   # กันกดซ้อน (เมาส์+ปุ่มลัดยิงพร้อมกันจากปุ่มเดียว — Grok #2)
        if now - getattr(self, "_last_toggle", 0.0) < 0.3:
            return
        self._last_toggle = now
        if self.active.is_set():
            self.active.clear()
            self._beep(494)   # เสียงต่ำ = ปิดหูแล้ว
            log("[jarvis] ปิดไมค์แล้ว (กดปุ่มลัดเพื่อเปิดใหม่)")
        else:
            self._drain_mic()
            self.last_activity = time.monotonic()
            self.last_voice = 0.0          # ล้างสถานะเก่า กัน log เวลาเพี้ยนรอบแรก (Grok #6)
            self.awaiting_reply = False
            self.turn_gate.reset()
            self.active.set()
            self._beep(988)   # เสียงสูง = เปิดหูแล้ว พูดได้เลย
            self.turn_gate.note_playback(time.monotonic())  # กันเสียงติ๊งย้อนเข้าไมค์ถูกนับเป็นพูด (Codex)
            log("[jarvis] เปิดแล้ว — พูดได้เลยครับ")

    def _drain_mic(self):
        while not self.audio.mic_q.empty():
            try:
                self.audio.mic_q.get_nowait()
            except Exception:
                break

    async def speak_text(self, text):
        """พูดข้อความสั้นด้วยเสียงในเครื่อง (speak.sh) — ไม่เปิด session Gemini ซ้อน."""
        speak = Path(__file__).resolve().parent / "speak.sh"
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: subprocess.run([str(speak), text], check=False, timeout=30))
        except Exception:
            pass

    async def idle_guard(self):
        """เงียบครบ IDLE_MIN → ถามจะปิดไหม → เงียบต่อ → ปิดเอง.

        last_activity อัปเดตตอน user พูดเข้าไมค์จริง (ใน send) → วัดถูก.
        """
        while self.active.is_set():
            await asyncio.sleep(5)
            if not self.active.is_set():
                return
            if time.monotonic() - self.last_activity > IDLE_MIN * 60:
                log("[jarvis] เงียบนานแล้ว — ถามเจ้านาย")
                await self.speak_text("ไม่ได้ใช้งานสักพักแล้วครับ ถ้ายังอยู่ให้พูดต่อได้เลย ไม่งั้นผมขอปิดไมค์นะครับ")
                self._drain_mic()               # ทิ้งเสียงคำถามที่ย้อนเข้าไมค์ (Codex #1)
                mark = time.monotonic() + 0.5   # นับ grace หลังพูดจบ + กันชนเฟรมที่ลอยค้าง 0.5 วิ
                await asyncio.sleep(GRACE_SEC)
                if not self.active.is_set():
                    return
                if self.last_activity > mark:   # user พูดช่วง grace → เปิดต่อ
                    continue
                log("[jarvis] เงียบต่อ — ปิดไมค์อัตโนมัติ")
                self.active.clear()
                return

    async def one_session(self, handle):
        async with self.client.aio.live.connect(
            model=MODEL,
            config=build_config(handle, manual_activity=MANUAL_ACTIVITY),
        ) as session:
            loop = asyncio.get_running_loop()
            new_handle = handle
            speech_turn = LocalSpeechTurnDetector(
                speech_rms=MIC_SPEECH_RMS,
                silence_ms=LOCAL_SILENCE_MS,
                min_speech_frames=LOCAL_MIN_SPEECH_FRAMES,
            )
            pre_speech_audio = deque(maxlen=max(1, LOCAL_MIN_SPEECH_FRAMES))

            async def send():
                while self.active.is_set():
                    try:  # มี timeout เพื่อไม่ให้ thread ค้างตอนกดปิด (Grok review #1)
                        data = await loop.run_in_executor(
                            None, lambda: self.audio.mic_q.get(timeout=0.2))
                    except queue.Empty:
                        continue
                    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
                    frame_ms = (arr.size / IN_RATE) * 1000.0 if arr.size else 0.0
                    now = time.monotonic()
                    if not BARGE_IN:
                        if self.audio.playing():
                            self.turn_gate.note_playback(now)
                            if self.turn_gate.should_interrupt_playback(rms):
                                self.audio.clear_playback()
                                self.turn_gate.reset()
                                self.last_activity = now
                                self.awaiting_reply = False
                                self._drain_mic()
                                log("[jarvis] ได้ยินเจ้านายพูดแทรก — หยุดเสียงตอบแล้วฟังใหม่")
                            continue   # ลำโพงกำลังพูด → ทิ้งเฟรมไมค์ กันเสียงย้อนตัดคำตอบ
                        if self.turn_gate.in_playback_tail(now):
                            continue   # หางกันเสียงหลังลำโพง — ปิดช่องว่างระหว่างก้อนเสียง (Codex รอบ 3)

                    if MANUAL_ACTIVITY and self.awaiting_reply and not speech_turn.active:
                        continue   # ส่งจบคำพูดแล้ว → รอเสียงตอบก่อน กันเสียงรอบข้างเปิดเทิร์นซ้อน
                    if MANUAL_ACTIVITY and not speech_turn.active:
                        pre_speech_audio.append(data)
                    event = speech_turn.observe(rms, frame_ms)
                    if event == "start":
                        self.last_activity = now   # นับเฉพาะตอน user พูดจริง
                        self.last_voice = now
                        if not self.awaiting_reply:
                            log("[jarvis] 🎤 ได้ยินเสียงพูด — รอคำตอบ")
                        self.awaiting_reply = True
                        if MANUAL_ACTIVITY:
                            await session.send_realtime_input(activity_start=types.ActivityStart())
                            for chunk in pre_speech_audio:
                                await session.send_realtime_input(
                                    audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={IN_RATE}"))
                            pre_speech_audio.clear()
                            continue
                    elif event == "speech":
                        self.last_activity = now
                        self.last_voice = now

                    if MANUAL_ACTIVITY and not speech_turn.active and event != "end":
                        continue

                    await session.send_realtime_input(
                        audio=types.Blob(data=data, mime_type=f"audio/pcm;rate={IN_RATE}"))
                    if MANUAL_ACTIVITY and event == "end":
                        await session.send_realtime_input(activity_end=types.ActivityEnd())
                        log("[jarvis] ส่งสัญญาณจบคำพูดแล้ว — รอเสียงตอบ")

            async def recv():
                nonlocal new_handle
                async for msg in session.receive():
                    sc = getattr(msg, "server_content", None)
                    if sc is not None and getattr(sc, "interrupted", None):
                        now = time.monotonic()
                        if self.turn_gate.should_clear_for_server_interrupt(now, self.audio.playing()):
                            self.audio.clear_playback()
                    if getattr(msg, "data", None):
                        if self.awaiting_reply and self.last_voice:
                            log(f"[jarvis] ⏱ เริ่มตอบใน {time.monotonic() - self.last_voice:.1f} วิ (นับจากเฟรมพูดสุดท้าย)")
                            self.awaiting_reply = False
                        # ไม่อัปเดต last_activity ที่นี่ — idle นับเฉพาะตอน user พูดจริง (Grok #7)
                        self.audio.play_q.put(np.frombuffer(msg.data, dtype=np.int16))
                    upd = getattr(msg, "session_resumption_update", None)
                    if upd is not None and getattr(upd, "new_handle", None):
                        new_handle = upd.new_handle

            async def until_off():
                while self.active.is_set():
                    await asyncio.sleep(0.2)

            tasks = [asyncio.create_task(send()), asyncio.create_task(recv()),
                     asyncio.create_task(until_off())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)   # รอ cancel จริง (Grok review #3)
            for t in done:
                exc = t.exception()
                if exc and not isinstance(exc, asyncio.CancelledError):
                    raise exc   # error จริง → ให้ชั้นนอกจัดการ/ต่อสายใหม่ (Grok review #4)
            return new_handle

    async def run(self):
        mic, spk = self.audio.streams()
        with mic, spk:
            log(f"[jarvis] พร้อม · รุ่น {MODEL} · ปุ่มลัด {HOTKEY} หรือปุ่มข้างเมาส์ = เปิด/ปิดพูด · เงียบ {IDLE_MIN:.0f} นาที = ปิดเอง")
            while True:
                await self.active.wait()
                guard = asyncio.create_task(self.idle_guard())
                handle = None
                try:
                    fast = 0
                    while self.active.is_set():
                        t0 = time.monotonic()
                        handle = await self.one_session(handle)
                        if time.monotonic() - t0 < 2:   # ปิดเร็วผิดปกติ → กันวนต่อสายรัว (Codex review)
                            fast += 1
                            if fast >= 5:
                                log("[jarvis] ต่อสายรัวผิดปกติ — พัก 5 วิ")
                                await asyncio.sleep(5)
                                fast = 0
                        else:
                            fast = 0
                except Exception as exc:
                    log(f"[jarvis] หลุด: {str(exc)[:100]} — กดปุ่มลัดเปิดใหม่")
                    self.active.clear()
                guard.cancel()


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    jarvis = Jarvis(loop)
    if START_ACTIVE:
        jarvis.active.set()
        jarvis.last_activity = time.monotonic()
        log("[jarvis] เปิดไมค์อัตโนมัติ — พูดได้เลยครับ")

    hk = keyboard.GlobalHotKeys({HOTKEY: jarvis.toggle_from_hotkey})
    hk.start()

    def on_click(x, y, button, pressed):
        """ปุ่มเมาส์ที่ไม่ใช่ซ้าย/ขวา = สวิตช์ (ปุ่มข้างส่วนใหญ่เข้ามาเป็น 'unknown' บน mac)."""
        if not pressed or button in (mouse.Button.left, mouse.Button.right):
            return
        log(f"[jarvis] 🖱 เห็นปุ่มเมาส์พิเศษ: {getattr(button, 'name', button)}")
        if MOUSE_TOGGLE and button != mouse.Button.middle:
            jarvis.toggle_from_hotkey()

    ml = mouse.Listener(on_click=on_click)
    ml.start()
    try:
        loop.run_until_complete(jarvis.run())
    except KeyboardInterrupt:
        pass
    finally:
        hk.stop()
        ml.stop()


if __name__ == "__main__":
    main()
