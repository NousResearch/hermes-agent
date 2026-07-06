"""JARVIS v2 เฟส 3 — ทดสอบสั่งงานผ่านด่านปลอดภัย (function calling) ด้วยข้อความ.

พิสูจน์ว่า Gemini เรียกเครื่องมือ open_url แล้วด่านความปลอดภัยเปิดเว็บจริง โดยไม่ต้องใช้ไมค์.
รัน: scripts/jarvis-voice/.venv-gemini/bin/python jarvis_tools_probe.py
"""
import asyncio
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from google import genai
from google.genai import types

MODEL = "gemini-2.5-flash-native-audio-latest"
SYSTEM = "คุณคือจาร์วิส ถ้าเจ้านายสั่งเปิดเว็บ ให้เรียกเครื่องมือ open_url เสมอ แล้วตอบสั้นๆ เป็นไทย"

OPEN_URL = types.FunctionDeclaration(
    name="open_url",
    description="เปิดเว็บไซต์ในเบราว์เซอร์ของเจ้านาย",
    parameters=types.Schema(
        type="OBJECT",
        properties={"url": types.Schema(type="STRING", description="ลิงก์ http/https")},
        required=["url"],
    ),
)

# ---- ด่านความปลอดภัย (gateway): อนุญาตเฉพาะ http/https เท่านั้น ----
ALLOWED_SCHEMES = {"http", "https"}


def _is_private_host(host: str) -> bool:  # กัน localhost/IP ภายใน (Codex review #3)
    host = host.lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1", ""}:
        return True
    if host.startswith(("10.", "192.168.", "127.", "169.254.")):
        return True
    return any(host.startswith(f"172.{i}.") for i in range(16, 32))


def gateway_open_url(url: str) -> dict:
    p = urlparse(url)
    if p.scheme.lower() not in ALLOWED_SCHEMES:
        print(f"[gateway] ปฏิเสธ (scheme ไม่อนุญาต): {url}", file=sys.stderr)
        return {"ok": False, "reason": "อนุญาตเฉพาะ http/https"}
    if _is_private_host(p.hostname or ""):
        print(f"[gateway] ปฏิเสธ (host ภายใน/ส่วนตัว): {url}", file=sys.stderr)
        return {"ok": False, "reason": "บล็อกเว็บภายในเครื่อง/เครือข่ายส่วนตัว"}
    print(f"[gateway] อนุญาต + เปิดเว็บจริง: {url}", file=sys.stderr)
    subprocess.run(["open", url], check=False)
    return {"ok": True, "opened": url}


def load_key() -> str:
    for line in (Path(__file__).resolve().parent / ".env").read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("ไม่พบ GEMINI_API_KEY")


async def probe() -> int:
    client = genai.Client(api_key=load_key())
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM,
        tools=[types.Tool(function_declarations=[OPEN_URL])],
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    called = {"open_url": False, "url": None}
    reply = []

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(
                text="ช่วยเปิดเว็บ https://www.google.com ให้หน่อยครับ")]),
            turn_complete=True,
        )
        async for msg in session.receive():
            tc = getattr(msg, "tool_call", None)
            if tc and tc.function_calls:
                responses = []
                for fc in tc.function_calls:
                    if fc.name == "open_url":
                        url = (fc.args or {}).get("url", "")
                        result = gateway_open_url(url)
                        called["open_url"] = True
                        called["url"] = url
                        responses.append(types.FunctionResponse(id=fc.id, name=fc.name, response=result))
                await session.send_tool_response(function_responses=responses)
                continue
            if getattr(msg, "text", None):
                reply.append(msg.text)
            sc = getattr(msg, "server_content", None)
            if sc is not None and getattr(sc, "turn_complete", False):
                break

    print(f"เรียกเครื่องมือ open_url: {'ใช่' if called['open_url'] else 'ไม่'}")
    print(f"เว็บที่เปิด: {called['url']}")
    print(f"จาร์วิสตอบ: {''.join(reply).strip()[:120]}")
    return 0 if called["open_url"] else 1


def main() -> int:
    try:
        return asyncio.run(asyncio.wait_for(probe(), timeout=60))
    except asyncio.TimeoutError:
        print("หมดเวลา 60 วินาที")
        return 1
    except Exception as exc:
        text = str(exc)
        print("โควตาฟรีเต็ม (429)" if "429" in text else f"ผิดพลาด: {text[:150]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
