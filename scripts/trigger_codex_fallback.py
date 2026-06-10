#!/usr/bin/env python3
"""Live fallback trigger: Codex → local-claude.

실제 hermesAgent gateway로 요청을 보낸 후,
Codex 엔드포인트를 차단해서 fallback이 발동되도록 만든다.

사용법:
    python scripts/trigger_codex_fallback.py [--gateway-url URL]

기본 gateway URL: http://localhost:5050
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error

DEFAULT_GATEWAY = "http://localhost:5050"


def chat(gateway_url: str, message: str, session_id: str = "") -> dict:
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{gateway_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        return {"error": str(e), "body": body, "status": e.code}
    except Exception as e:
        return {"error": str(e)}


def get_session_model(gateway_url: str, session_id: str) -> dict:
    req = urllib.request.Request(
        f"{gateway_url}/api/sessions/{session_id}",
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Trigger Codex→fallback scenario")
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY)
    args = parser.parse_args()
    gw = args.gateway_url.rstrip("/")

    print(f"Gateway: {gw}")
    print("=" * 60)

    # ── Step 1: Codex session — create a skill ──────────────────────
    print("\n[1] Codex session: 간단한 스킬 생성 요청 전송...")
    resp1 = chat(gw, "안녕! 지금 어떤 provider로 동작 중인지 알려줘.")
    session_id = resp1.get("session_id", "")
    print(f"    session_id : {session_id}")
    print(f"    provider   : {resp1.get('model', '?')} / {resp1.get('provider', '?')}")
    print(f"    응답 미리보기: {str(resp1.get('response', resp1))[:120]}")

    if resp1.get("error"):
        print(f"\n⚠️  Gateway 오류: {resp1['error']}")
        print("   hermesAgent gateway가 실행 중인지 확인하세요.")
        sys.exit(1)

    time.sleep(1)

    # ── Step 2: Send a message that produces codex_reasoning_items ──
    print("\n[2] 작업 요청 (Codex가 reasoning items 생성하도록)...")
    resp2 = chat(
        gw,
        "간단한 Python 함수 하나만 작성해줘: def add(a, b): 로 시작하는 덧셈 함수.",
        session_id=session_id,
    )
    print(f"    응답 미리보기: {str(resp2.get('response', resp2))[:120]}")

    time.sleep(1)

    # ── Step 3: Force fallback via /model switch or error injection ─
    print("\n[3] Fallback 시뮬레이션: /model 명령으로 local-claude로 전환...")
    resp3 = chat(
        gw,
        "/model claude-opus-4-8 local-claude",
        session_id=session_id,
    )
    print(f"    응답: {str(resp3.get('response', resp3))[:200]}")

    time.sleep(1)

    # ── Step 4: Verify the new provider sees context from step 2 ───
    print("\n[4] Fallback 후 이전 Codex 작업 컨텍스트 확인...")
    resp4 = chat(
        gw,
        "방금 전에 내가 요청한 Python 함수가 뭐였지? 그리고 지금 어떤 provider로 동작 중이야?",
        session_id=session_id,
    )
    print(f"    provider   : {resp4.get('provider', '?')}")
    print(f"    응답:\n    {str(resp4.get('response', resp4))[:400]}")

    # ── Result ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("테스트 완료.")
    print(f"  session_id  : {session_id}")
    print(f"  최종 provider: {resp4.get('provider', '?')}")
    response_text = str(resp4.get("response", ""))
    if "add" in response_text.lower() or "덧셈" in response_text or "plus" in response_text.lower():
        print("  ✅ 이전 Codex 세션 컨텍스트가 fallback provider에 전달됨!")
    else:
        print("  ⚠️  이전 컨텍스트 전달 여부 불명확 — 응답을 직접 확인하세요.")


if __name__ == "__main__":
    main()
