import asyncio
import json
import logging
import os
import re
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

YT_REGEX = re.compile(r"(https?://(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)[a-zA-Z0-9_-]+)")

MAX_DURATION_SECONDS = 600
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

def extract_youtube_url(text: str) -> Optional[str]:
    match = YT_REGEX.search(text)
    if match:
        return match.group(1)
    return None

async def get_youtube_info(url: str) -> Optional[dict]:
    cmd = ["yt-dlp", "-J", "--no-playlist", url]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(f"[YouTube] yt-dlp info failed: {stderr.decode()}")
            return None
        return json.loads(stdout)
    except Exception as e:
        logger.error(f"[YouTube] get_youtube_info exception: {e}")
        return None

async def download_youtube_audio(url: str, output_dir: str) -> Tuple[bool, str]:
    info = await get_youtube_info(url)
    if not info:
        return False, "Failed to fetch video info."
    
    duration = info.get("duration", 0)
    if duration > MAX_DURATION_SECONDS:
        return False, f"Video is too long ({duration}s > {MAX_DURATION_SECONDS}s)."

    output_template = os.path.join(output_dir, "%(id)s.%(ext)s")
    
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f", "bestaudio",
        "-x",
        "--audio-format", "mp3",
        "-o", output_template,
        url
    ]
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(f"[YouTube] yt-dlp download failed: {stderr.decode()}")
            return False, "Failed to download audio."
            
        expected_path = os.path.join(output_dir, f"{info['id']}.mp3")
        if os.path.exists(expected_path):
            size = os.path.getsize(expected_path)
            if size > MAX_FILE_SIZE_BYTES:
                os.remove(expected_path)
                return False, "Downloaded file is too large."
            return True, expected_path
            
        return False, "Downloaded file not found."
    except Exception as e:
        logger.error(f"[YouTube] download_youtube_audio exception: {e}")
        return False, f"Exception: {e}"
