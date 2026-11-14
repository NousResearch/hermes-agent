# Plan 005: Shift voice integration ownership to agent-meow

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Move local voice orchestration, model selection, and bridge lifecycle into `agent-meow`, while keeping Hermes on one stable TTS provider contract.

**Architecture:** `agent-meow` owns a host-side voice gateway that exposes one Hermes-facing HTTP contract (`GET /health`, `POST /tts`). Hermes does not own Edge/Piper/Qwen orchestration anymore; it only calls the gateway. For Omnigent-managed Hermes sessions, `agent-meow` injects the gateway config into per-session `HERMES_HOME`. For standalone Docker Hermes, `data/config.yaml` points one command provider at the same gateway.

**Tech Stack:** Python 3.12+, FastAPI or equivalent HTTP app in `agent-meow`, `edge-tts`, `piper-tts`, `qwen_tts`, `pytest`, Dockerized Hermes gateway.

## Global Constraints

- Do **not** add new `agent-meow` imports or local-path assumptions to Hermes core runtime files such as `tools/tts_tool.py`.
- Hermes keeps only the stable provider contract and optional sample config; all local orchestration belongs in `agent-meow`.
- Preserve the sidecar separation: Qwen remains host-side, not baked into Hermes containers.
- Support two launch modes with one contract:
  - Omnigent-managed Hermes on the host uses `http://127.0.0.1:17494/tts`
  - Dockerized Hermes containers use `http://host.docker.internal:17494/tts`
- Keep fallback order explicit in `agent-meow` config; do not encode `edge -> piper -> qwen` ordering inside Hermes `tts.providers`.
- Maintain the current intended Chinese behavior unless the user explicitly changes it:
  - primary Edge voice: `zh-CN-XiaoxiaoNeural`
  - offline fallback: `zh_CN-huayan-medium`
  - offline neural fallback: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

---

## Why this matters

The current repo state mixes two responsibilities:

1. Hermes as a generic provider runtime.
2. Local operator logic for one specific desktop stack (`agent-meow` host + Docker Hermes + Edge/Piper/Qwen).

That mixing is why drift is accumulating in Hermes docs, helper scripts, and local config. The clean fix is not to keep teaching Hermes more about the local stack. The clean fix is to move the local stack behind an `agent-meow`-owned gateway and make Hermes consume one stable provider contract.

## Current state

- Hermes still has a generic provider surface in `c:\Users\1\github-pr\hermes-agent\tools\tts_tool.py`:
  - command providers are generic (`tts.providers.<name>`)
  - the built-in default path is still Edge with NeuTTS fallback
  - there is no real `edge -> piper-zh -> qwen3-tts` chain in Hermes core
- Hermes local config currently carries the multi-provider operator stack in `c:\Users\1\github-pr\hermes-agent\data\config.yaml`:
  - `tts.provider: edge`
  - `providers.piper-zh`
  - `providers.qwen3-tts`
  - `qwen3-tts` points to `http://host.docker.internal:17494/tts`
- Hermes helper scripts are agent-meow-specific today:
  - `c:\Users\1\github-pr\hermes-agent\scripts\qwen3-tts-server.py` assumes the `agent-meow` venv
  - `c:\Users\1\github-pr\hermes-agent\scripts\setup_tts_triple.bat` hardcodes `c:\Users\1\github-pr\agent-meow`
- `agent-meow` already has the right ownership seam:
  - `c:\Users\1\github-pr\agent-meow\agent_meow\inner\hermes_executor.py` creates per-session `HERMES_HOME`
  - `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_native_bridge.py` writes `config.yaml` into that home and already merges model/provider settings
  - `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_native.py` and `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_native_bridge.py` are the natural home for Hermes-specific local integration behavior

## Commands you will need

| Purpose                             | Command                                                                                                                                                                                                                                                  | Expected on success                      |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Agent-meow gateway tests            | `uv run pytest c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_gateway.py c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_backends.py -q`                                                                                                | all pass                                 |
| Agent-meow Hermes integration tests | `uv run pytest c:\Users\1\github-pr\agent-meow\tests\test_hermes_native_bridge.py c:\Users\1\github-pr\agent-meow\tests\inner\test_hermes_executor.py c:\Users\1\github-pr\agent-meow\tests\inner\test_hermes_native_executor.py -q`                     | all pass                                 |
| Gateway health smoke                | `uv run python -m agent_meow.hermes_voice_gateway --port 17494` then `Invoke-RestMethod -Uri "http://127.0.0.1:17494/health"`                                                                                                                            | returns JSON health payload              |
| Gateway synth smoke                 | `Invoke-WebRequest -Uri "http://127.0.0.1:17494/tts" -Method POST -Body '{"text":"你好世界"}' -ContentType "application/json" -OutFile "$env:TEMP\agent_meow_voice_test.wav"`                                                                            | writes non-empty WAV                     |
| Standalone Hermes smoke             | `docker exec hermes-gateway python3 -c "import json; from tools.tts_tool import text_to_speech_tool; result=json.loads(text_to_speech_tool(text='你好世界')); print(result.get('success')); print(result.get('error')); print(result.get('file_path'))"` | prints `True`, no error, valid file path |

## Scope

**Primary implementation repo**:

- `c:\Users\1\github-pr\agent-meow`

**Minimal contract-change repo**:

- `c:\Users\1\github-pr\hermes-agent`

**In scope in agent-meow**:

- `agent_meow/hermes_native.py`
- `agent_meow/hermes_native_bridge.py`
- `agent_meow/inner/hermes_executor.py`
- new voice-gateway modules under `agent_meow/`
- tests for the gateway and Hermes integration

**In scope in hermes-agent**:

- `data/config.yaml`
- helper scripts and docs that currently encode agent-meow-specific voice behavior

**Out of scope**:

- Adding Hermes-core fallback orchestration to `tools/tts_tool.py`
- Baking Qwen, Piper, or Edge into the Hermes image as the long-term solution
- Making Hermes core import Omnigent or `agent-meow`

## Git workflow

- Track this plan in the Hermes repo under `plans/`.
- Implement most code changes in `c:\Users\1\github-pr\agent-meow`.
- Keep Hermes-side changes as small contract/documentation cleanup commits.
- Do not execute plans 002 or 003 as written if this plan is chosen; re-scope them afterward.

## Tasks

### Task 1A: Build the minimal Hermes voice gateway skeleton

**Files:**

- Create: `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_voice_gateway.py`
- Test: `c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_gateway.py`

**Interfaces:**

- Consumes: `agent_meow.config.load_effective_config()`
- Produces:
  - `create_app() -> FastAPI`
  - `load_hermes_voice_settings() -> HermesVoiceSettings`
  - `GET /health` returning provider/model state
  - `POST /tts` accepting the same payload shape Hermes already uses today
  - `synthesize_stub(text: str, settings: HermesVoiceSettings) -> SynthesisResult`

**Payload contract (must stay byte-compatible with the current Hermes-side caller):**
The existing Hermes command provider in `data/config.yaml` POSTs `{"text": "<text>"}` to `http://host.docker.internal:17494/tts` and writes the response body to the output file. The existing Flask bridge in `scripts/qwen3-tts-server.py` also accepts `text_file` and `output_path` in the JSON body. The new gateway must accept the same JSON keys so Hermes-side config does not change in this slice:

```json
{
  "text": "你好世界",
  "text_file": "/optional/path.txt",
  "output_path": "/optional/out.wav",
  "speaker": "Vivian",
  "language": "English"
}
```

Response: raw `audio/wav` bytes, `Content-Type: audio/wav`, HTTP 200.

- [ ] **Step 1: Write the failing tests**

Create `c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_gateway.py` with exactly these tests:

```python
"""Contract tests for the agent-meow Hermes voice gateway skeleton.

No real Edge/Piper/Qwen calls — the stub synthesizer returns a tiny valid WAV.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agent_meow import hermes_voice_gateway as gw


def _is_valid_wav_header(data: bytes) -> bool:
    """Minimal WAV validation: RIFF header + 'fmt ' chunk present."""
    return (
        len(data) >= 44
        and data[:4] == b"RIFF"
        and data[8:12] == b"WAVE"
        and data[12:16] == b"fmt "
    )


def test_create_app_returns_fastapi_with_health_and_tts_routes() -> None:
    app = gw.create_app()
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/tts" in paths


def test_health_returns_ok_with_stub_mode_and_attempt_order() -> None:
    client = TestClient(gw.create_app())
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] == "stub"
    assert isinstance(body["attempt_order"], list)
    assert len(body["attempt_order"]) >= 1


def test_tts_with_direct_text_returns_wav_bytes() -> None:
    client = TestClient(gw.create_app())
    r = client.post("/tts", json={"text": "你好世界"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"
    assert _is_valid_wav_header(r.content)
    assert len(r.content) > 44


def test_tts_with_text_file_and_output_path_writes_file(tmp_path: Path) -> None:
    text_file = tmp_path / "input.txt"
    text_file.write_text("你好世界", encoding="utf-8")
    out = tmp_path / "out.wav"
    client = TestClient(gw.create_app())
    r = client.post(
        "/tts",
        json={"text_file": str(text_file), "output_path": str(out)},
    )
    assert r.status_code == 200
    assert _is_valid_wav_header(r.content)
    assert out.is_file()
    assert out.stat().st_size > 44
    assert out.read_bytes() == r.content


def test_tts_missing_text_and_text_file_returns_400() -> None:
    client = TestClient(gw.create_app())
    r = client.post("/tts", json={})
    assert r.status_code == 400
    body = r.json()
    assert "text" in body["detail"].lower() or "text_file" in body["detail"].lower()


def test_tts_empty_text_returns_400() -> None:
    client = TestClient(gw.create_app())
    r = client.post("/tts", json={"text": "   "})
    assert r.status_code == 400


def test_load_hermes_voice_settings_defaults() -> None:
    settings = gw.load_hermes_voice_settings({})
    assert settings.mode == "stub"
    assert isinstance(settings.attempt_order, tuple)
    assert settings.attempt_order == ("stub",)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_gateway.py -q`
Expected: all tests FAIL with `ModuleNotFoundError: No module named 'agent_meow.hermes_voice_gateway'` (or `ImportError`).

- [ ] **Step 3: Implement the gateway module skeleton**

Create `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_voice_gateway.py` with exactly this content:

```python
"""agent-meow-owned Hermes voice gateway — minimal skeleton (Task 1A).

Exposes one stable Hermes-facing HTTP contract:
  GET  /health  → JSON status + provider-chain metadata
  POST /tts      → raw audio/wav bytes

This slice uses a stub synthesizer that returns a tiny valid WAV payload
without calling Edge, Piper, or Qwen. Real backends arrive in Task 1B.

Payload is byte-compatible with the existing Hermes-side Qwen bridge
(scripts/qwen3-tts-server.py) so Hermes config does not change here.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

_logger = logging.getLogger(__name__)

# --- Minimal settings ---------------------------------------------------

_DEFAULT_ATTEMPT_ORDER = ("stub",)


@dataclass(frozen=True)
class HermesVoiceSettings:
    """Voice gateway settings for the current slice.

    Only enough structure to prove the contract boundary. Real provider
    names and fallback order arrive in Task 1B.
    """

    mode: str = "stub"
    attempt_order: tuple[str, ...] = _DEFAULT_ATTEMPT_ORDER


def load_hermes_voice_settings(config: dict[str, Any] | None = None) -> HermesVoiceSettings:
    """Load voice settings from the effective Omnigent config.

    This slice ignores most config keys; it only proves the boundary.
    Task 1B will read real provider/voice/model settings here.
    """
    _ = config or {}
    return HermesVoiceSettings(mode="stub", attempt_order=_DEFAULT_ATTEMPT_ORDER)


# --- Stub synthesizer ---------------------------------------------------

@dataclass(frozen=True)
class SynthesisResult:
    """Result of one synthesis attempt."""

    audio_bytes: bytes
    sample_rate: int = 16000
    provider: str = "stub"
    attempted: tuple[str, ...] = ("stub",)


def _tiny_wav(sample_rate: int = 16000, duration_s: float = 0.05) -> bytes:
    """Return a tiny valid WAV file (silence) for contract tests."""
    import wave

    n_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()


def synthesize_stub(text: str, settings: HermesVoiceSettings) -> SynthesisResult:
    """Return a tiny valid WAV without calling any real provider."""
    _ = text  # stub ignores text content
    _ = settings
    return SynthesisResult(audio_bytes=_tiny_wav(), provider="stub", attempted=("stub",))


# --- HTTP app -----------------------------------------------------------

def create_app() -> FastAPI:
    """Build the FastAPI app exposing /health and /tts."""
    app = FastAPI(title="agent-meow Hermes voice gateway")
    settings = load_hermes_voice_settings()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": settings.mode,
            "attempt_order": list(settings.attempt_order),
        }

    @app.post("/tts")
    async def tts(request: Request) -> Response:
        data = await request.json()
        text = data.get("text")
        text_file = data.get("text_file")

        if text:
            text = text.strip()
        elif text_file:
            try:
                text = Path(text_file).read_text(encoding="utf-8").strip()
            except (OSError, FileNotFoundError):
                raise HTTPException(status_code=400, detail=f"Cannot read text_file: {text_file}")
        else:
            raise HTTPException(status_code=400, detail="Missing 'text' or 'text_file'")

        if not text:
            raise HTTPException(status_code=400, detail="Empty text")

        result = synthesize_stub(text, settings)

        output_path = data.get("output_path")
        if output_path:
            try:
                Path(output_path).write_bytes(result.audio_bytes)
            except OSError:
                pass  # cross-platform path — ignore

        return Response(content=result.audio_bytes, media_type="audio/wav")

    return app


def main() -> None:
    """Run the gateway with uvicorn for local/operator smoke tests."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="agent-meow Hermes voice gateway")
    parser.add_argument("--port", type=int, default=17494, help="Listen port")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    args = parser.parse_args()
    uvicorn.run(create_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_gateway.py -q`
Expected: all tests PASS with no real network/provider dependency.

- [ ] **Step 5: Smoke-test the gateway manually**

Run: `uv run python -m agent_meow.hermes_voice_gateway --port 17494`
Then in another terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:17494/health"
```

Expected: `status=ok`, `mode=stub`, `attempt_order=["stub"]`

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:17494/tts" -Method POST -Body '{"text":"你好世界"}' -ContentType "application/json" -OutFile "$env:TEMP\agent_meow_voice_test.wav"
```

Expected: non-empty `.wav` file written.

- [ ] **Step 6: Commit**

```bash
cd c:\Users\1\github-pr\agent-meow
git add agent_meow/hermes_voice_gateway.py tests/test_hermes_voice_gateway.py
git commit -m "feat: add minimal Hermes voice gateway skeleton (Plan 005 Task 1A)"
```

### Task 1B: Replace the stub with real host-side backends

**Files:**

- Modify: `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_voice_gateway.py`
- Create: `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_voice_backends.py`
- Test: `c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_backends.py`
- Modify: `c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_gateway.py`

**Interfaces:**

- Consumes: `HermesVoiceSettings` and the HTTP contract from Task 1A
- Produces:
  - `synthesize_with_chain(text: str, settings: HermesVoiceSettings) -> SynthesisResult`
  - real backend adapters for Edge, Piper, and Qwen
  - real fallback reporting in `/health` and failure responses

- [ ] Replace the stub synthesizer with the real backend dispatcher.
  - Edge backend owns `zh-CN-XiaoxiaoNeural` selection.
  - Piper backend owns `zh_CN-huayan-medium` voice/model resolution.
  - Qwen backend owns model selection and can start with `1.7B` as the supported model.
  - Fallback order lives here, not in Hermes.

- [ ] Expand the gateway tests to cover attempted-provider reporting and final-fallback errors.

- [ ] Add backend unit tests with fakes/mocks only.
  - Run: `uv run pytest c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_gateway.py c:\Users\1\github-pr\agent-meow\tests\test_hermes_voice_backends.py -q`
  - Expected: all pass with no real network/provider dependency.

### Task 2: Make agent-meow inject the voice contract into Hermes sessions

**Files:**

- Create: `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_voice_overlay.py`
- Modify: `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_native_bridge.py`
- Modify: `c:\Users\1\github-pr\agent-meow\agent_meow\inner\hermes_executor.py`
- Modify: `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_native.py` only if launch-time gateway checks belong there
- Test: `c:\Users\1\github-pr\agent-meow\tests\test_hermes_native_bridge.py`
- Test: `c:\Users\1\github-pr\agent-meow\tests\inner\test_hermes_executor.py`

**Interfaces:**

- Consumes:
  - `write_policy_hook_config(...)`
  - per-session `HERMES_HOME`
  - effective Omnigent config
- Produces:
  - `build_hermes_voice_overlay(base_tts_url: str) -> dict`
  - `merge_hermes_voice_overlay(config: dict, overlay: dict) -> dict`
  - optional `ensure_hermes_voice_gateway_running() -> GatewayHandle | None`

- [ ] Add a small overlay builder instead of hardcoding voice config in the bridge writer.
  - Host-launched Hermes must use `http://127.0.0.1:17494/tts`.
  - The overlay must define exactly one Hermes-facing provider name, for example `agent-meow-voice`.
  - The overlay must set `tts.provider: agent-meow-voice` and one `tts.providers.agent-meow-voice` command entry.

- [ ] Merge the overlay into the per-session `HERMES_HOME/config.yaml` that `agent-meow` already writes.
  - Keep existing model/provider/auth copy-through behavior.
  - Do not break the policy-hook or MCP-server registration that already lives in `write_policy_hook_config()`.

- [ ] Add tests that assert the generated `config.yaml` contains the overlay and the correct host URL.
  - One test for headless `HermesExecutor`.
  - One test for `hermes-native` session setup.

- [ ] Run the Hermes integration tests in agent-meow.
  - Run: `uv run pytest c:\Users\1\github-pr\agent-meow\tests\test_hermes_native_bridge.py c:\Users\1\github-pr\agent-meow\tests\inner\test_hermes_executor.py c:\Users\1\github-pr\agent-meow\tests\inner\test_hermes_native_executor.py -q`
  - Expected: all pass.

### Task 3: Centralize gateway lifecycle in agent-meow, not Hermes scripts

**Files:**

- Modify: `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_voice_gateway.py`
- Modify: `c:\Users\1\github-pr\agent-meow\agent_meow\hermes_native.py`
- Optional create: `c:\Users\1\github-pr\agent-meow\scripts\run_hermes_voice_gateway.py`
- Test: `c:\Users\1\github-pr\agent-meow\tests\test_hermes_native.py`

**Interfaces:**

- Consumes: the gateway app from Task 1
- Produces:
  - a documented way to launch the gateway for standalone Docker Hermes
  - an optional auto-start/health-check path for Omnigent-managed Hermes

- [ ] Decide one supported startup path.
  - Preferred: `python -m agent_meow.hermes_voice_gateway --port 17494`
  - Optional enhancement: `omnigent hermes` verifies the gateway is running and starts it if needed.

- [ ] Remove the need for Hermes-side operator scripts to know agent-meow internals.
  - New lifecycle logic must make `c:\Users\1\github-pr\hermes-agent\scripts\qwen3-tts-server.py` and `c:\Users\1\github-pr\hermes-agent\scripts\setup_tts_triple.bat` unnecessary as the primary control plane.

- [ ] Add a smoke test path for operators.
  - Run: `uv run python -m agent_meow.hermes_voice_gateway --port 17494`
  - Then run: `Invoke-RestMethod -Uri "http://127.0.0.1:17494/health"`
  - Then run: `Invoke-WebRequest -Uri "http://127.0.0.1:17494/tts" -Method POST -Body '{"text":"你好世界"}' -ContentType "application/json" -OutFile "$env:TEMP\agent_meow_voice_test.wav"`
  - Expected: non-empty WAV file.

### Task 4: Shrink Hermes to one stable provider contract

**Files:**

- Modify: `c:\Users\1\github-pr\hermes-agent\data\config.yaml`
- Modify or deprecate: `c:\Users\1\github-pr\hermes-agent\scripts\qwen3-tts-server.py`
- Modify or deprecate: `c:\Users\1\github-pr\hermes-agent\scripts\setup_tts_triple.bat`
- Optional later docs: `c:\Users\1\github-pr\hermes-agent\README.md`, `c:\Users\1\github-pr\hermes-agent\INSTALL.md`

**Interfaces:**

- Consumes: the agent-meow gateway contract from Tasks 1-3
- Produces:
  - one Hermes-side provider name, for example `agent-meow-voice`
  - one Docker URL target: `http://host.docker.internal:17494/tts`

- [ ] Replace the triple-provider local stack in Hermes config with one provider contract.
  - `data/config.yaml` should stop advertising `edge`, `piper-zh`, and `qwen3-tts` as Hermes-owned selection/fallback choices.
  - Keep Hermes config to one command provider that calls the agent-meow gateway over HTTP.

- [ ] Do not add fallback-chain logic to `tools/tts_tool.py` for this route.
  - Hermes already has a generic command-provider feature; use it.
  - The fallback chain lives entirely behind the gateway.

- [ ] Demote Hermes helper scripts from “primary runtime path” to either:
  - deleted,
  - archived,
  - or clearly marked migration shims.

- [ ] Verify standalone Docker Hermes against the new contract.
  - Start the gateway from the agent-meow repo.
  - Run: `docker exec hermes-gateway python3 -c "import json; from tools.tts_tool import text_to_speech_tool; result=json.loads(text_to_speech_tool(text='你好世界')); print(result.get('success')); print(result.get('error')); print(result.get('file_path'))"`
  - Expected: `success == True`, no error, audio file path printed.

### Task 5: Re-scope the old Hermes-local voice plans

**Files:**

- Modify: `c:\Users\1\github-pr\hermes-agent\plans\README.md`
- Modify: `c:\Users\1\github-pr\hermes-agent\plans\002-implement-real-tts-fallback-chain.md`
- Modify: `c:\Users\1\github-pr\hermes-agent\plans\003-stabilize-qwen-0.6b-model-management.md`
- Modify: `c:\Users\1\github-pr\hermes-agent\plans\004-document-triple-tts-switching.md`

**Interfaces:**

- Consumes: the chosen gateway-owned architecture
- Produces:
  - a consistent plan queue that does not keep steering work back into Hermes

- [ ] Mark plan 005 as the preferred path when local customization must live in `agent-meow`.
- [ ] Rewrite plan 002 so it targets the agent-meow gateway fallback chain instead of `tools/tts_tool.py`.
- [ ] Rewrite plan 003 so Qwen model management lives in `agent-meow` instead of Hermes helper scripts.
- [ ] Rewrite plan 004 so docs describe the single-provider contract, not three Hermes-owned providers.

## Test plan

- Agent-meow unit tests must cover gateway contract, fallback behavior, and Hermes config overlay generation.
- No Hermes-core TTS fallback tests should be added for this architecture, because Hermes no longer owns that logic.
- Run one host-side smoke test against `127.0.0.1:17494` and one Docker Hermes smoke test against `host.docker.internal:17494`.

## Done criteria

- [ ] `agent-meow` owns the Edge/Piper/Qwen orchestration and exposes one Hermes-facing gateway contract
- [ ] Omnigent-managed Hermes sessions receive the voice provider contract through per-session `HERMES_HOME`
- [ ] Standalone Docker Hermes can call the same gateway through one command provider
- [ ] Hermes no longer carries local agent-meow path assumptions in primary runtime scripts/config
- [ ] Plans 002-004 are either rewritten or explicitly superseded for the agent-meow-owned route

## STOP conditions

- The user decides the fallback chain should be a general Hermes upstream feature for all users, not a local integration feature
- `agent-meow` cannot reliably own host-side Edge/Piper/Qwen dependencies on the target machine
- The gateway contract adds unacceptable latency or operational fragility compared with the current direct local setup

## Maintenance notes

- Keep the Hermes-facing contract boring and stable. Local experimentation belongs behind the gateway.
- If a provider-specific tweak is only useful for this workstation or Omnigent-managed Hermes sessions, it belongs in `agent-meow`, not Hermes.
- Prefer deleting Hermes-side helper scripts once the gateway path is proven; stale local runbooks are the main source of repo drift here.
