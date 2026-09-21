"""FFMPEG video generator for text animations and stat reveals.
No AI involved, pure code rendering. Free. Works on 8GB RAM VPS."""
import os
import random
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

FFMPEG_BIN = os.getenv("FFMPEG_PATH", "ffmpeg")

OUTPUT_BASE = Path("/home/kensei/repos/KenseiAgent/content_engine/output")


def _drawtext_clause(text: str, tempfiles: List[str], **opts) -> str:
    """Build a drawtext filter clause using textfile= form.

    The text content goes into a temp file so FFmpeg reads it raw, avoiding
    filter-graph escape issues with chars like , % [ ] \\ ; that broke the
    earlier text='...' form (C2).

    Caller passes a list to accumulate temp paths in; cleanup is the caller's
    responsibility (typically `finally: for p in tempfiles: os.unlink(p)`).
    """
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="ffmpeg_drawtext_", text=True)
    try:
        os.write(fd, (text or "").encode("utf-8"))
    finally:
        os.close(fd)
    tempfiles.append(path)
    opt_str = ":".join(f"{k}={v}" for k, v in opts.items() if v is not None)
    return f"drawtext=textfile='{path}':{opt_str}" if opt_str else f"drawtext=textfile='{path}'"


def _cleanup_tempfiles(tempfiles: List[str]) -> None:
    for p in tempfiles:
        try:
            os.unlink(p)
        except OSError:
            pass


def _run_ffmpeg(args: List[str]) -> bool:
    import subprocess
    try:
        subprocess.run(
            [FFMPEG_BIN] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFMPEG failed: {e.stderr.decode()[:200]}")
        return False
    except Exception as e:
        print(f"FFMPEG error: {e}")
        return False


def make_stat_reveal_video(
    brand: str,
    stat_line: str,
    sub_line: str,
    duration: int = 4,
    resolution: str = "1080x1080",
    out_path: Optional[str] = None,
) -> Optional[str]:
    """Animated stat reveal: number pops in, sub-line fades. Returns path."""
    bg_colour = "#0A0A0A"
    accent = "#FBBF24" if brand == "matchdaymaestro" else "#10B981" if brand == "plenishd" else "#8B0000"

    out_dir = OUTPUT_BASE / brand / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_path or f"{brand}_stat_reveal_{str(uuid.uuid4())[:8]}.mp4"
    fpath = out_dir / fname

    tempfiles: List[str] = []
    try:
        # Two-pass: stat text then sub text. textfile= form avoids filter-graph
        # escape issues with commas, percents, brackets etc in the body copy (C2).
        clauses = [
            _drawtext_clause(
                stat_line, tempfiles,
                fontcolor=accent, fontsize=72,
                x="(w-text_w)/2", y="(h-text_h)/2-40",
                alpha="'if(lt(t,0.5),t*2,if(lt(t,3.5),1,(4-t)*2))'",
                font="DejaVuSans-Bold",
            ),
            _drawtext_clause(
                sub_line, tempfiles,
                fontcolor="white", fontsize=36,
                x="(w-text_w)/2", y="(h-text_h)/2+60",
                alpha="'if(lt(t,1.0),0,if(lt(t,1.5),(t-1)*2,if(lt(t,3.5),1,(4-t)*2)))'",
                font="DejaVuSans",
            ),
        ]
        filter_txt = ",".join(clauses)

        args = [
            "-f", "lavfi",
            "-i", f"color=c={bg_colour}:s={resolution}:r=30:d={duration}",
            "-vf", filter_txt,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-t", str(duration),
            "-y",
            str(fpath),
        ]

        if _run_ffmpeg(args):
            return str(fpath)
        return None
    finally:
        _cleanup_tempfiles(tempfiles)


def make_countdown_video(
    brand: str,
    question: str,
    countdown_from: int = 3,
    resolution: str = "1080x1080",
) -> Optional[str]:
    """Countdown video for quiz reveals or prediction windows."""
    bg_colour = "#0F0F1A" if brand == "matchdaymaestro" else "#2C2A28" if brand == "plenishd" else "#0A0A0A"
    accent = "#E11D48" if brand == "matchdaymaestro" else "#FBBF24" if brand == "plenishd" else "#8B0000"

    duration = countdown_from + 2
    out_dir = OUTPUT_BASE / brand / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{brand}_countdown_{str(uuid.uuid4())[:8]}.mp4"
    fpath = out_dir / fname

    # Build expression-based countdown text
    # Each number shows for 1 second
    fps = 30

    tempfiles: List[str] = []
    # Build drawtext with enable per segment. Numbers are small literals so
    # text='{num}' is safe and avoids creating tempfiles for each frame.
    texts = []
    y_pos = "(h-text_h)/2"
    for i, num in enumerate(range(countdown_from, 0, -1)):
        start = i
        end = i + 1
        texts.append(
            f"drawtext=text='{num}':fontcolor={accent}:fontsize=180:"
            f"x=(w-text_w)/2:y={y_pos}:"
            f"enable='between(t\\,{start}\\,{end})':"
            f"font=DejaVuSans-Bold"
        )

    # Question text (persistent, small). textfile= form for body safety (C2).
    # Alpha expression (C25 fix): the previous form had a malformed `lt(t,X,Y,Z)`
    # collapsed-args bug AND referenced FFmpeg-unknown `duration` variable. Now
    # pre-computes duration into the f-string and uses proper if(lt(),Y,Z) shape.
    question_filter = _drawtext_clause(
        question, tempfiles,
        fontcolor="white", fontsize=32,
        x="(w-text_w)/2", y="80",
        alpha=f"'if(lt(t,0.5),t*2,if(lt(t,{duration-0.5}),1,({duration}-t)*2))'",
        font="DejaVuSans",
    )

    filter_chain = ",".join(texts) + "," + question_filter

    args = [
        "-f", "lavfi",
        "-i", f"color=c={bg_colour}:s={resolution}:r={fps}:duration={duration}",
        "-vf", filter_chain,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",
        str(fpath),
    ]

    try:
        if _run_ffmpeg(args):
            return str(fpath)
        return None
    finally:
        _cleanup_tempfiles(tempfiles)


def make_battle_challenge_video(
    brand: str,
    challenger: str,
    stake: int,
    your_rank: str,
    resolution: str = "1080x1080",
) -> Optional[str]:
    """MMaestro-specific: battle challenge teaser."""
    bg_colour = "#0F0F1A"
    accent = "#E11D48"

    duration = 3
    out_dir = OUTPUT_BASE / brand / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{brand}_battle_{str(uuid.uuid4())[:8]}.mp4"
    fpath = out_dir / fname

    # Write user-controlled text to tempfiles; FFmpeg textfile= reads raw (C2).
    # stake (int) and the fixed "Your move." string don't need it.
    tempfiles: List[str] = []
    fd, challenger_path = tempfile.mkstemp(suffix=".txt", prefix="ffmpeg_drawtext_", text=True)
    try:
        os.write(fd, f"{challenger} challenged you".encode("utf-8"))
    finally:
        os.close(fd)
    tempfiles.append(challenger_path)

    filter_complex = (
        f"color=c={bg_colour}:s={resolution}:duration={duration}[bg];"
        f"[bg]drawtext=textfile='{challenger_path}':fontcolor=white:fontsize=48:"
        f"x=(w-text_w)/2:y=200:"
        f"alpha='if(lt(t,0.3),t*3.3,if(lt(t,2.7),1,(3-t)*3.3))':"
        f"font=DejaVuSans-Bold[txt1];"
        f"[txt1]drawtext=text='{stake} coins on the line':fontcolor={accent}:fontsize=64:"
        f"x=(w-text_w)/2:y=300:"
        f"alpha='if(lt(t,0.6),0,if(lt(t,1.0),(t-0.6)*2.5,if(lt(t,2.7),1,(3-t)*3.3)))':"
        f"font=DejaVuSans-Bold[txt2];"
        f"[txt2]drawtext=text='Your move.':fontcolor=white:fontsize=36:"
        f"x=(w-text_w)/2:y=500:"
        f"alpha='if(lt(t,1.5),0,if(lt(t,1.8),(t-1.5)*3.3,if(lt(t,2.7),1,(3-t)*3.3)))':"
        f"font=DejaVuSans[txt3]"
    )

    args = [
        "-f", "lavfi",
        "-i", f"color=c={bg_colour}:s={resolution}:r=30:duration={duration}",
        "-vf", filter_complex,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",
        str(fpath),
    ]

    try:
        if _run_ffmpeg(args):
            return str(fpath)
        return None
    finally:
        _cleanup_tempfiles(tempfiles)
