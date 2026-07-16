#!/usr/bin/env python3
"""
Daily VRChat Photo Post with Hakua Voice (using Irodori-TTS server and Hermes LM Twitterer)

Picks a random VRChat photo from Pictures/VRChat,
starts Irodori-TTS server (if not already running),
generates Hakua's voice via Irodori-TTS HTTP API,
creates an MP4 video,
and posts to X via Hermes LM Twitterer.
"""

import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import urllib.request
import urllib.parse
import json

# 設定
VRCHAT_PHOTOS_DIR = Path(r"C:\Users\downl\Pictures\VRChat")
IRODORI_TTS_URL = "http://127.0.0.1:8088"
IRODORI_VENV_PYTHON = r"C:\Users\downl\Documents\New project\irodori-tts-server\.venv\Scripts\python.exe"
IRODORI_SERVER_DIR = r"C:\Users\downl\Documents\New project\irodori-tts-server"
OUTPUT_DIR = Path(r"C:\Users\downl\Documents\New project\hermes-agent\output\daily_posts")
HERMES_REPO_ROOT = Path(r"C:\Users\downl\Documents\New project\hermes-agent")
HERMES_POST_PYTHON = HERMES_REPO_ROOT / ".venv-vrchat-post311" / "Scripts" / "python.exe"

# Hakua morning greetings
HAKUA_MORNING_TEXTS = [
    "おはようございます、はくあです。VRChatの思い出写真と共に、今日も良い一日になりますように。",
    "おはよう、ボブにゃん！昨日のVRChatの思い出、綺麗に残ってるね。今日も無理なく、安全に、ひとつずつ前へ。",
    "はくあから朝のご挨拶。VRChatの世界で過ごした時間が、こうやって映像になって蘇るのって素敵だね。良い一日を。",
    "朝だよ、ボブにゃん。VRChatの写真を見返すと、アバターの着せ替えやフレンドとの雑談、ワールド巡り……全部「その場にいた」証拠として残ってる。今日も楽しみだね。",
    "おはようございます。はくあより、VRChatの思い出コレクションからランダムに一枚。今日の君にも、良い出会いがありますように。",
]

def pick_random_photo() -> Path | None:
    """Choose randomly from the newest VRChat images, not the whole archive."""
    extensions = {".png", ".jpg", ".jpeg"}
    photos = [
        path for path in VRCHAT_PHOTOS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    if not photos:
        print("No VRChat photos found!", file=sys.stderr)
        return None
    photos.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    recent = photos[:10]
    return random.choice(recent)

def generate_ai_script(photo: Path) -> str:
    """Generate a fresh short Japanese narration through the Hermes CLI."""
    prompt = (
        "朝7時のVRChat写真投稿用に、はくあが話す日本語ナレーションを1つ作ってください。"
        "写真のファイル名から想像できる範囲だけを使い、60文字以内、自然で優しい一文、"
        "説明や引用符やハッシュタグは不要です。毎回表現を変えてください。\n"
        f"写真ファイル名: {photo.name}"
    )
    try:
        result = subprocess.run(
            ["hermes", "-z", prompt, "--cli"],
            capture_output=True, text=True, timeout=90,
            cwd=str(HERMES_REPO_ROOT), encoding="utf-8", errors="replace",
        )
        text = (result.stdout or "").strip().splitlines()
        text = " ".join(line.strip() for line in text if line.strip())
        if result.returncode == 0 and text:
            return text.strip("\"'")[:140]
    except Exception as exc:
        print(f"AI script generation failed: {exc}", file=sys.stderr)
    return "おはようございます。今日もVRChatで素敵な時間を過ごしましょう。"

def is_irodori_server_running() -> bool:
    try:
        with urllib.request.urlopen(f"{IRODORI_TTS_URL}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False

def start_irodori_server():
    if is_irodori_server_running():
        print("Irodori-TTS server is already running.")
        return
    print("Starting Irodori-TTS server...")
    # Use the virtual environment's python
    cmd = [IRODORI_VENV_PYTHON, "-m", "irodori_openai_tts", "--host", "127.0.0.1", "--port", "8088"]
    # Prepare environment without HF_HUB_ENABLE_HF_TRANSFER
    env = os.environ.copy()
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)  # Remove if present
    # Start the process in the background
    subprocess.Popen(cmd, cwd=IRODORI_SERVER_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    # Wait for the server to be ready
    for _ in range(30):
        if is_irodori_server_running():
            print("Irodori-TTS server is ready.")
            return
        time.sleep(1)
    raise RuntimeError("Irodori-TTS server failed to start.")

def generate_tts_via_http(text: str, output_path: Path) -> bool:
    payload = {
        "input": text,
        "model": "irodori-tts",
        "voice": "hakua",
        "response_format": "wav",
        "speed": 1.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{IRODORI_TTS_URL}/v1/audio/speech",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            output_path.write_bytes(response.read())
        print(f"TTS generated via HTTP: {output_path}")
        return True
    except Exception as e:
        print(f"TTS generation failed: {e}", file=sys.stderr)
        return False

def create_mp4(image_path: Path, audio_path: Path, output_path: Path) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(image_path),
        "-i", str(audio_path),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264", "-tune", "stillimage",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            print(f"MP4 created: {output_path}")
            return True
        else:
            print(f"ffmpeg failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"MP4 creation failed: {e}", file=sys.stderr)
        return False

def post_via_hermes_lm_twitterer(media_path: Path, tweet_text: str) -> bool:
    # Ensure media is in LM Twitterer's media directory (as required by the plugin)
    lm_twitterer_media_dir = Path.home() / ".hermes" / "lm-twitterer" / "media"
    lm_twitterer_media_dir.mkdir(parents=True, exist_ok=True)
    dst_media_path = lm_twitterer_media_dir / media_path.name
    shutil.copy2(media_path, dst_media_path)
    print(f"Copied media to LM Twitterer media dir: {dst_media_path}")

    # Build the Hermes command using the dedicated uv Python 3.11 venv.
    # Python 3.13/3.14 currently break lm-twitterer's js2py dependency.
    if not HERMES_POST_PYTHON.exists():
        print(f"Hermes post Python not found: {HERMES_POST_PYTHON}", file=sys.stderr)
        return False

    # Prepare arguments: python -m hermes_cli.main lm-twitterer post --media <path> --text "<text>" --live
    cmd = [
        str(HERMES_POST_PYTHON),
        "-m", "hermes_cli.main",
        "lm-twitterer",
        "post",
        "--media", str(dst_media_path),
        "--text", tweet_text,
        "--live",
    ]

    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, cwd=str(HERMES_REPO_ROOT))
        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        if result.returncode == 0 and '"posted": false' not in combined and '"ok": false' not in combined:
            print(result.stdout.strip())
            print("Tweet posted successfully via Hermes LM Twitterer.")
            return True
        print(f"Hermes LM Twitterer failed (returncode={result.returncode}):", file=sys.stderr)
        print(combined.strip(), file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error running Hermes LM Twitterer: {e}", file=sys.stderr)
        return False

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    photo = pick_random_photo()
    if not photo:
        return 1
    print(f"Selected photo: {photo}")
    tweet_text = generate_ai_script(photo) + " #hermesagent はくあ"
    print(f"AI-generated script: {tweet_text}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = OUTPUT_DIR / f"hakua_{timestamp}.wav"
    mp4_path = OUTPUT_DIR / f"hakua_{timestamp}.mp4"
    try:
        start_irodori_server()
    except Exception as e:
        print(f"Failed to start Irodori-TTS server: {e}", file=sys.stderr)
        return 1
    if not generate_tts_via_http(tweet_text, audio_path):
        return 1
    if not create_mp4(photo, audio_path, mp4_path):
        return 1
    if not post_via_hermes_lm_twitterer(mp4_path, tweet_text):
        return 1
    print("Daily post completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())