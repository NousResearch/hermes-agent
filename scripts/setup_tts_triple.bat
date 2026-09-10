@echo off
REM ============================================================
REM DEPRECATED: This script is superseded by the agent-meow voice
REM gateway (python -m agent_meow.hermes_voice_gateway) as of Plan 005
REM Task 4. It remains as a migration shim. Do not extend — new voice
REM lifecycle logic belongs in agent-meow.
REM ============================================================
REM Hermes TTS Triple Setup: Edge TTS + Piper + Qwen3-TTS
REM Run this from a NEW PowerShell terminal as Administrator
REM ============================================================
echo === Step 1: Kill stuck processes ===
taskkill /f /im python.exe 2>nul
taskkill /f /im python3.exe 2>nul
echo Done.

echo.
echo === Step 2: Start Qwen3-TTS bridge server (1.7B) ===
cd /d c:\Users\1\github-pr\agent-meow
start "Qwen3-TTS Bridge" cmd /c ".venv\Scripts\python.exe ..\hermes-agent\scripts\qwen3-tts-server.py --port 17494 --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
echo Bridge server starting on port 17494...

echo.
echo === Step 3: Install Piper + Chinese voice in Docker ===
docker exec hermes-gateway python3 -c "import subprocess, sys; subprocess.run([sys.executable, '-m', 'pip', 'install', 'piper-tts'], check=True)"
echo piper-tts installed.

echo.
echo === Step 4: Download Chinese Piper voice ===
docker exec hermes-gateway python3 -m piper.download_voices zh_CN-huayan-medium
echo Chinese voice downloaded.

echo.
echo === Step 5: Install edge-tts in Docker ===
docker exec hermes-gateway python3 -c "from tools.lazy_deps import ensure; ensure('tts.edge', prompt=False); print('edge_tts OK')"
echo edge-tts installed.

echo.
echo === Step 6: Restart Docker containers ===
cd /d c:\Users\1\github-pr\hermes-agent
docker compose -f docker-compose.upstream.yml up -d --force-recreate hermes-gateway hermes-web
echo Docker restarted.

echo.
echo === Step 7: Verify ===
echo.
echo Testing Edge TTS (EN)...
docker exec hermes-gateway python3 -c "from tools.lazy_deps import ensure; ensure('tts.edge', prompt=False); import edge_tts, asyncio, time, os; async def t(): c=edge_tts.Communicate('Hello world','en-US-AriaNeural'); await c.save('/tmp/test_en.mp3'); print(f'Edge EN: {os.path.getsize(\"/tmp/test_en.mp3\")/1024:.0f}KB in {time.time()-t.__code__.co_consts}') "
echo.
echo Testing Edge TTS (ZH)...
docker exec hermes-gateway python3 -c "import edge_tts, asyncio, time, os; async def t(): t0=time.time(); c=edge_tts.Communicate('你好世界','zh-CN-XiaoxiaoNeural'); await c.save('/tmp/test_zh.mp3'); print(f'Edge ZH: {os.path.getsize(\"/tmp/test_zh.mp3\")/1024:.0f}KB in {time.time()-t0:.1f}s'); asyncio.run(t())"
echo.
echo Testing Qwen3-TTS bridge...
curl -s http://127.0.0.1:17494/health
echo.
echo === Setup Complete ===
echo.
echo Config (data/config.yaml):
echo   Default: edge (en-US-AriaNeural)
echo   Fallback 1: piper-zh (zh_CN-huayan-medium, offline)
echo   Fallback 2: qwen3-tts (1.7B, offline, 23-52s)
echo.
echo To switch provider: change tts.provider in data/config.yaml
echo   - "edge" for fast cloud TTS (needs internet)
echo   - "piper-zh" for offline Chinese TTS (~50MB, fast)
echo   - "qwen3-tts" for offline neural TTS (slow on CPU)
pause
