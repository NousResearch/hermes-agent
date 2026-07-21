#!/usr/bin/env python3
"""
Daily VRChat Photo Post with Hakua Voice

Picks a random VRChat photo from Pictures/VRChat,
generates Hakua's voice via irodoriTTS,
creates an MP4 video,
and posts to X via xurl with media upload.
"""

import os
import random
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Configuration
VRCHAT_PHOTOS_DIR = Path(r"C:\Users\downl\Pictures\VRChat")
IRODORI_TTS_URL = "http://127.0.0.1:8088"
IRODORI_VOICE = "hakua"
OUTPUT_DIR = Path(r"C:\Users\downl\Documents\New project\hermes-agent\output\daily_posts")
XURL_APP = "hermes"

# Hakua morning greetings
HAKUA_MORNING_TEXTS = [
    "おはようございます、はくあです。VRChatの思い出写真と共に、今日も良い一日になりますように。",
    "おはよう、ボブにゃん！昨日のVRChatの思い出、綺麗に残ってるね。今日も無理なく、安全に、ひとつずつ前へ。",
    "はくあから朝のご挨拶。VRChatの世界で過ごした時間が、こうやって映像になって蘇るのって素敵だね。良い一日を。",
    "朝だよ、ボブにゃん。VRChatの写真を見返すと、アバターの着せ替えやフレンドとの雑談、ワールド巡り……全部「その場にいた」証拠として残ってる。今日も楽しみだね。",
    "おはようございます。はくあより、VRChatの思い出コレクションからランダムに一枚。今日の君にも、良い出会いがありますように。",
]

def pick_random_photo() -> Path | None:
    extensions = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
    photos = []
    for ext in extensions:
        photos.extend(VRCHAT_PHOTOS_DIR.rglob(f"*{ext}"))
    if not photos:
        print("No VRChat photos found!", file=sys.stderr)
        return None
    return random.choice(photos)

def generate_tts(text: str, output_path: Path) -> bool:
    import urllib.request
    import json
    payload = {
        "input": text,
        "model": "irodori-tts",
        "voice": IRODORI_VOICE,
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
        print(f"TTS generated: {output_path}")
        return True
    except Exception as e:
        print(f"TTS generation failed: {e}", file=sys.stderr)
        return False

def create_mp4(image_path: Path, audio_path: Path, output_path: Path) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(image_path),
        "-i", str(audio_path),
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

def upload_media_and_post(media_path: Path, tweet_text: str) -> bool:
    upload_cmd = [
        "xurl", "media", "upload",
        "--app", XURL_APP,
        "--media-type", "video/mp4",
        "--category", "tweet_video",
        "--wait",
        str(media_path),
    ]
    try:
        print("Uploading media...")
        result = subprocess.run(upload_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"Media upload failed: {result.stderr}", file=sys.stderr)
            return False
        import json
        upload_result = json.loads(result.stdout)
        media_id = upload_result.get("media_id_string") or upload_result.get("media_id")
        if not media_id:
            print(f"Could not get media_id from: {result.stdout}", file=sys.stderr)
            return False
        print(f"Media uploaded: {media_id}")
        tweet_cmd = [
            "xurl", "post",
            "--app", XURL_APP,
            "--media", media_id,
            tweet_text,
        ]
        print("Posting tweet...")
        result = subprocess.run(tweet_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"Tweet posted: {result.stdout.strip()}")
            return True
        else:
            print(f"Tweet post failed: {result.stderr}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"X posting failed: {e}", file=sys.stderr)
        return False

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    photo = pick_random_photo()
    if not photo:
        return 1
    print(f"Selected photo: {photo}")
    tweet_text = random.choice(HAKUA_MORNING_TEXTS) + " #hermesagent"
    print(f"Tweet text: {tweet_text}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_path = OUTPUT_DIR / f"hakua_{timestamp}.wav"
    mp4_path = OUTPUT_DIR / f"hakua_{timestamp}.mp4"
    if not generate_tts(tweet_text, audio_path):
        return 1
    if not create_mp4(photo, audio_path, mp4_path):
        return 1
    if not upload_media_and_post(mp4_path, tweet_text):
        return 1
    print("Daily post completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
