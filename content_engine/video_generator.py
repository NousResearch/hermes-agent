"""
FFMPEG-based video generation for simple animated content.
No AI video generation needed — this handles:
- Text reveal countdowns
- Stat fly-in animations
- Image slideshows with transitions
- Quiz answer reveals
"""

import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int = 48, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Load a font for video frames."""
    font_paths = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _sanitize_filename(text: str) -> str:
    """Remove characters unsafe for filenames."""
    return re.sub(r'[^\w\-_.]', '_', text)[:50]


def create_countdown_video(
    title: str,
    countdown_items: list[str],
    reveal_text: str,
    output_path: str = None,
    width: int = 1080,
    height: int = 1920,  # 9:16 for TikTok/Instagram Reels
    duration_per_item: float = 2.0,
    fps: int = 30,
    bg_color: tuple = (18, 18, 18),
    accent_color: tuple = (220, 38, 38),
    text_color: tuple = (255, 255, 255),
) -> str:
    """
    Create a countdown reveal video.
    
    Args:
        title: Top headline text
        countdown_items: List of items to count through (e.g., ["3...", "2...", "1..."])
        reveal_text: Final reveal text
        output_path: Output MP4 path
        width/height: Video dimensions
        duration_per_item: Seconds per countdown item
        fps: Frames per second
    
    Returns:
        Path to generated MP4
    """
    # Create temp dir for frames
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        title_font = _get_font(64, bold=True)
        item_font = _get_font(120, bold=True)
        reveal_font = _get_font(72, bold=True)
        small_font = _get_font(32)
        
        frame_idx = 0
        total_duration = len(countdown_items) * duration_per_item + 3.0  # +3s for reveal
        total_frames = int(total_duration * fps)
        
        for i, item in enumerate(countdown_items):
            # Generate frames for this item
            item_frames = int(duration_per_item * fps)
            
            for f in range(item_frames):
                img = Image.new("RGB", (width, height), bg_color)
                draw = ImageDraw.Draw(img)
                
                # Title at top
                draw.text((width // 2, 100), title, fill=text_color, font=title_font, anchor="mt")
                
                # Countdown item, centered, with scale animation
                progress = f / item_frames
                scale = 1.0 + 0.2 * (1 - progress)  # Shrink slightly over time
                size = int(120 * scale)
                
                # Big number in center
                draw.text((width // 2, height // 2 - 50), item, fill=accent_color, font=item_font, anchor="mm")
                
                # Progress bar at bottom
                bar_width = int(width * 0.8 * progress)
                draw.rectangle([width * 0.1, height - 100, width * 0.1 + bar_width, height - 80], fill=accent_color)
                
                # Save frame
                img.save(tmpdir / f"frame_{frame_idx:05d}.png")
                frame_idx += 1
        
        # Reveal frames (3 seconds)
        reveal_frames = int(3.0 * fps)
        for f in range(reveal_frames):
            img = Image.new("RGB", (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            
            # Title
            draw.text((width // 2, 100), title, fill=text_color, font=title_font, anchor="mt")
            
            # Reveal text with fade-in
            alpha = min(1.0, f / (reveal_frames * 0.3))
            
            # "Reveal" label
            draw.text((width // 2, height // 2 - 120), "The answer:", fill=(180, 180, 180), font=small_font, anchor="mm")
            
            # Actual reveal
            draw.text((width // 2, height // 2), reveal_text, fill=accent_color, font=reveal_font, anchor="mm")
            
            # Brand footer
            draw.text((width // 2, height - 120), "MatchdayMaestro", fill=accent_color, font=small_font, anchor="mm")
            draw.text((width // 2, height - 80), "Predict. Compete. Win.", fill=(180, 180, 180), font=_get_font(24), anchor="mm")
            
            img.save(tmpdir / f"frame_{frame_idx:05d}.png")
            frame_idx += 1
        
        # Build video with ffmpeg
        if output_path is None:
            output_dir = Path.home() / "apps" / "content-engine" / "assets" / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"countdown_{_sanitize_filename(title)}_{uuid.uuid4().hex}.mp4"
        
        # FFMPEG command
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",  # Quality (lower = better, 23 is default)
            "-preset", "fast",
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"FFMPEG failed: {result.stderr}")
        
        return str(output_path)


def create_stat_reveal_video(
    headline: str,
    stats: list[dict],  # [{"label": "", "value": "", "prefix": ""}]
    output_path: str = None,
    width: int = 1080,
    height: int = 1920,
    duration_per_stat: float = 3.0,
    fps: int = 30,
    bg_color: tuple = (18, 18, 18),
    accent_color: tuple = (251, 191, 36),
    text_color: tuple = (255, 255, 255),
) -> str:
    """
    Create a stat reveal video with animated fly-ins.
    Each stat flies in from the side with a counter animation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        headline_font = _get_font(56, bold=True)
        label_font = _get_font(36)
        value_font = _get_font(96, bold=True)
        small_font = _get_font(24)
        
        frame_idx = 0
        
        for stat in stats:
            label = stat["label"]
            value = stat["value"]
            prefix = stat.get("prefix", "")
            
            stat_frames = int(duration_per_stat * fps)
            
            for f in range(stat_frames):
                img = Image.new("RGB", (width, height), bg_color)
                draw = ImageDraw.Draw(img)
                
                # Headline
                draw.text((width // 2, 120), headline, fill=text_color, font=headline_font, anchor="mt")
                
                # Animation progress
                progress = min(1.0, f / (stat_frames * 0.6))
                
                # Label slides in from left
                label_x = int(width * 0.1 + (width * 0.4) * (1 - progress))
                draw.text((label_x, height // 2 - 80), label, fill=(180, 180, 180), font=label_font)
                
                # Value counts up
                # Try to parse as number for counting animation
                try:
                    target_num = float(value.replace(",", "").replace("%", ""))
                    is_number = True
                    suffix = "%" if "%" in value else ""
                    current_num = target_num * progress
                    if target_num == int(target_num):
                        display_value = f"{prefix}{int(current_num)}{suffix}"
                    else:
                        display_value = f"{prefix}{current_num:.1f}{suffix}"
                except ValueError:
                    display_value = f"{prefix}{value}"
                    is_number = False
                
                # Value bounces in
                bounce = 1.0 + 0.1 * (1 - progress) if progress < 1.0 else 1.0
                size = int(96 * bounce)
                value_font_scaled = _get_font(size, bold=True)
                
                draw.text((width // 2, height // 2 + 40), display_value, fill=accent_color, font=value_font_scaled, anchor="mm")
                
                # Save
                img.save(tmpdir / f"frame_{frame_idx:05d}.png")
                frame_idx += 1
        
        # Build video
        if output_path is None:
            output_dir = Path.home() / "apps" / "content-engine" / "assets" / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"stats_{_sanitize_filename(headline)}_{uuid.uuid4().hex}.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(tmpdir / "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-preset", "fast",
            str(output_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"FFMPEG failed: {result.stderr}")
        
        return str(output_path)


def create_image_slideshow_video(
    image_paths: list[str],
    captions: list[str] = None,
    output_path: str = None,
    width: int = 1080,
    height: int = 1920,
    duration_per_image: float = 3.0,
    transition_frames: int = 10,
    fps: int = 30,
    bg_color: tuple = (18, 18, 18),
) -> str:
    """Full-bleed Ken Burns clip from one or more stills, with optional music.

    Each image is scaled to cover the full 9:16 frame (no letterbox bars) and
    given a slow push-in via ffmpeg zoompan, so the output reads as motion
    rather than a frozen frame. If any audio file exists under output/music/,
    one is picked deterministically and mixed in, trimmed to the clip length;
    TikTok/Reels punish silent video hard.
    """
    if output_path is None:
        output_dir = Path.home() / "apps" / "content-engine" / "assets" / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"slideshow_{uuid.uuid4().hex}.mp4"

    frames = int(duration_per_image * fps)
    # Slow push-in: ~8% zoom over the clip. Upscale first so zoompan's
    # integer-pixel pan doesn't shimmer.
    kb_filter = (
        f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
        f"crop={width * 2}:{height * 2},"
        f"zoompan=z='min(zoom+0.0014,1.08)':d={frames}"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps},"
        f"format=yuv420p"
    )

    music_dir = Path(__file__).resolve().parent / "output" / "music"
    tracks = sorted(music_dir.glob("*.mp3")) + sorted(music_dir.glob("*.m4a")) if music_dir.exists() else []

    clip_paths = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for i, img_path in enumerate(image_paths):
            clip = tmpdir / f"clip_{i:03d}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(img_path),
                "-vf", kb_filter,
                "-t", str(duration_per_image),
                "-c:v", "libx264", "-crf", "21", "-preset", "fast",
                "-an",
                str(clip),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                raise RuntimeError(f"FFMPEG failed: {result.stderr[-400:]}")
            clip_paths.append(clip)

        if len(clip_paths) == 1:
            visual = clip_paths[0]
        else:
            concat_list = tmpdir / "concat.txt"
            concat_list.write_text("".join(f"file '{c}'\n" for c in clip_paths))
            visual = tmpdir / "joined.mp4"
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                 "-c", "copy", str(visual)],
                capture_output=True, text=True, timeout=180,
            )
            if result.returncode != 0:
                raise RuntimeError(f"FFMPEG concat failed: {result.stderr[-400:]}")

        total = duration_per_image * len(image_paths)
        if tracks:
            # Deterministic per output name so re-runs are reproducible.
            track = tracks[hash(Path(output_path).stem) % len(tracks)]
            cmd = [
                "ffmpeg", "-y", "-i", str(visual), "-i", str(track),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                "-af", f"afade=t=out:st={max(total - 1.0, 0)}:d=1.0",
                "-t", str(total), "-shortest",
                str(output_path),
            ]
        else:
            cmd = ["ffmpeg", "-y", "-i", str(visual), "-c", "copy", str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"FFMPEG mux failed: {result.stderr[-400:]}")

    return str(output_path)


def get_video_stats(video_path: str) -> dict:
    """Get video duration and dimensions using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        return {}
    import json
    return json.loads(result.stdout)
