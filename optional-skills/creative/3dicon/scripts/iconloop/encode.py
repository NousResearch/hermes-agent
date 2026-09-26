"""Encode the matted frames.

The default is the SOURCE frame rate. That sounds obvious and is exactly the
mistake this tool exists to stop: sampling a 24fps render down to 8fps to hit a
file-size target produces judder that is easy to blame on the model, the
matting, or the player, and is none of those. Smooth motion is where a low
frame rate shows worst, so the icons that most want animating suffer most.

If the full-rate file is too heavy, `sweep()` prints what each tradeoff
actually costs so the choice is made on numbers rather than a guess.
"""
import glob
import io
import os
import subprocess

from PIL import Image


def load(master_dir):
    return [Image.open(p).convert("RGBA") for p in sorted(glob.glob(f"{master_dir}/*.png"))]


def resample(frames, n):
    """Pick n frames evenly around the loop; wrap spacing stays uniform."""
    total = len(frames)
    return [frames[int(round(i * total / n)) % total] for i in range(n)]


def webp(frames, out_path, fps, size=288, quality=50, n=None):
    """Animated WebP with real alpha — what expo-image, Chrome and Safari all play."""
    sel = resample(frames, n) if n else frames
    sel = [f.resize((size, size), Image.LANCZOS) for f in sel]
    sel[0].save(out_path, save_all=True, append_images=sel[1:],
                duration=int(round(1000 / fps)), loop=0, format="WEBP",
                lossless=False, quality=quality, method=4)
    return os.path.getsize(out_path)


def webm(master_dir, out_path, fps, size=288, crf=28):
    """VP9 with alpha, for the web. Smaller than WebP, but video not image.

    Note for anyone verifying this file: WebM keeps VP9 alpha in a side channel
    flagged `alpha_mode=1`. ffprobe reports pix_fmt=yuv420p and a default
    decode looks opaque. Force the libvpx decoder to see it:
        ffmpeg -c:v libvpx-vp9 -i out.webm -pix_fmt rgba f.png
    """
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
        "-i", f"{master_dir}/%04d.png", "-vf", f"scale={size}:{size}",
        "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
        "-crf", str(crf), "-b:v", "0", "-an", out_path], check=True)
    return os.path.getsize(out_path)


def mp4_flat(master_dir, out_path, fps, size=288, bg="0xFFFFFF", crf=20):
    """H.264 on a flat background — the universal fallback.

    There is no transparent-MP4 option here on purpose: that needs HEVC with an
    alpha layer, and libx265 as shipped by most distros reports "does not
    support alpha layer encoding". See docs/limitations.md.
    """
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-framerate", str(fps),
        "-i", f"{master_dir}/%04d.png", "-f", "lavfi", "-i", f"color=c={bg}:s={size}x{size}",
        "-filter_complex", f"[0:v]scale={size}:{size}[fg];[1:v][fg]overlay=shortest=1,format=yuv420p",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
        "-movflags", "+faststart", "-an", out_path], check=True)
    return os.path.getsize(out_path)


def sweep(frames, source_fps, sizes=(288, 256), fractions=(1, 2, 3), quality=50):
    """Print size against frame rate so the tradeoff is visible, not implied."""
    rows = []
    n_src = len(frames)
    for size in sizes:
        scaled = [f.resize((size, size), Image.LANCZOS) for f in frames]
        for div in fractions:
            n = max(2, round(n_src / div))
            fps = source_fps / div
            sel = resample(scaled, n)
            buf = io.BytesIO()
            sel[0].save(buf, save_all=True, append_images=sel[1:],
                        duration=int(round(1000 / fps)), loop=0, format="WEBP",
                        lossless=False, quality=quality, method=4)
            rows.append((size, n, fps, buf.tell()))
    print("\n  px    frames    fps      size")
    print("  " + "-" * 34)
    for size, n, fps, b in rows:
        mark = "  <- source rate" if abs(fps - source_fps) < 0.01 else ""
        print(f"  {size:<5} {n:<9} {fps:<8.1f} {b/1024:6.0f} KB{mark}")
    print()
    return rows
