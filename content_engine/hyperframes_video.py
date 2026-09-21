"""HyperFrames video generator for KENSEI Content Engine.

Generates animated HTML-based videos using HyperFrames CLI.
Uses screenshot-capture mode (PRODUCER_FORCE_SCREENSHOT=true) for CPU-only VPS.

Features:
- Dynamic HTML templates per band/animation type
- Renders to temp dir via subprocess
- Saves to output/videos/hyperfaces/<band>/<id>.mp4
- Falls back to ffmpeg_video.py on failure
- Resource guards (max renders, cleanup)
"""

import html
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

# Import config for brand colours
from config import BRANDS

# Constants
OUTPUT_BASE = Path("/home/kensei/repos/KenseiAgent/content_engine/output")
HYPERFRAMES_OUTPUT = OUTPUT_BASE / "videos" / "hyperframes"
MAX_RENDERS_PER_RUN = 2  # Safety guard
RENDER_TIMEOUT = 120  # seconds
TEMP_DIR_BASE = Path("/tmp/hyperframes_render")

# Ensure output directories exist
HYPERFRAMES_OUTPUT.mkdir(parents=True, exist_ok=True)
TEMP_DIR_BASE.mkdir(parents=True, exist_ok=True)

# Track renders this run (simple counter)
_render_count = 0


def _reset_render_counter():
    """Reset the render counter (call at start of each batch)."""
    global _render_count
    _render_count = 0


def _can_render() -> bool:
    """Check if we can render more videos this run."""
    global _render_count
    return _render_count < MAX_RENDERS_PER_RUN


def _increment_render_counter():
    """Increment the run counter."""
    global _render_count
    _render_count += 1


def _cleanup_temp_dir(temp_dir: Path):
    """Clean up a temporary directory."""
    if temp_dir and temp_dir.exists():
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temp dir {temp_dir}: {e}")


def _chromium_executable() -> Optional[str]:
    """Locate an arm64 headless Chromium for puppeteer.

    HyperFrames' own cached chrome-headless-shell is x86-64 (Google ships no
    linux-arm64 Chrome for Testing), so on this box we point puppeteer at the
    Playwright-built arm64 headless shell instead. Env override wins.
    """
    env_path = os.getenv("PUPPETEER_EXECUTABLE_PATH", "")
    if env_path and Path(env_path).exists():
        return env_path
    for pattern in (
        ".cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell",
        ".cache/ms-playwright/chromium-*/chrome-linux/chrome",
    ):
        candidates = sorted(Path.home().glob(pattern))
        if candidates:
            return str(candidates[-1])
    return None


def _run_hyperframes(html_dir: Path, output_path: Path) -> bool:
    """
    Run HyperFrames CLI to render an HTML composition directory to video.

    html_dir must contain an index.html file with the composition.
    Returns True on success, False on failure.
    """
    # Ensure we're using screenshot-capture mode and CPU fallback
    env = os.environ.copy()
    env.update({
        "PRODUCER_FORCE_SCREENSHOT": "true",
        "ORT_ALLOW_CPU_FALLBACK": "1",
        "ORT_USE_CUDA": "0",
    })
    chromium = _chromium_executable()
    if chromium:
        env["PUPPETEER_EXECUTABLE_PATH"] = chromium
    
    # Build command — hyperframes expects a directory, not a file
    cmd = [
        "hyperframes",
        "render",
        str(html_dir),
        "--output", str(output_path),
        "--quality", "draft",  # Faster, lower quality for frequent use
        "--fps", "30",
        "--format", "mp4",
    ]
    
    try:
        # Run with timeout
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT
        )
        
        if result.returncode == 0:
            # Verify output exists and has size
            if output_path.exists() and output_path.stat().st_size > 0:
                return True
            else:
                print(f"HyperFrames: output file empty or missing: {output_path}")
                return False
        else:
            print(f"HyperFrames failed (exit {result.returncode}): {result.stderr[-500:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"HyperFrames timeout after {RENDER_TIMEOUT}s")
        return False
    except Exception as e:
        print(f"HyperFrames error: {e}")
        return False


def _generate_stat_reveal_html(band: str, stat_line: str, sub_line: str = "") -> str:
    """
    Generate HTML for a stat reveal animation.
    
    Template: stat line animates up, sub-line fades in below.
    """
    brand_config = BRANDS.get(band, BRANDS["matchdaymaestro"])
    bg_color = brand_config["bg"]
    accent_color = brand_config["accent"]
    text_color = "#FFFFFF"
    
    # Escape HTML special chars
    stat_line = stat_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    sub_line = sub_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0;
      overflow: hidden;
      background: {bg_color};
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .scene {{
      width: 1080px;
      height: 1920px;
      position: relative;
    }}
    .stat-content {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
    }}
    .stat-line {{
      font-size: 72px;
      font-weight: 900;
      color: {accent_color};
      margin-bottom: 24px;
      letter-spacing: -1px;
    }}
    .sub-line {{
      font-size: 36px;
      font-weight: 500;
      color: {text_color};
      opacity: 0;
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
</head>
<body>
  <div class="scene" 
       data-composition-id="stat-reveal" 
       data-width="1080" 
       data-height="1920" 
       data-duration="4"
       data-start="0">
    <div class="stat-content">
      <div class="stat-line clip" id="stat-line">{stat_line}</div>
      <div class="sub-line clip" id="sub-line">{sub_line}</div>
    </div>
  </div>
  
  <script>
    window.__timelines = window.__timelines || {{}};
    const tl = gsap.timeline({{ paused: true }});
    
    // Stat line: slide up and fade in
    tl.from("#stat-line", {{ 
      y: 50, 
      opacity: 0, 
      duration: 0.8, 
      ease: "power3.out" 
    }});
    
    // Sub line: fade in slightly after
    tl.from("#sub-line", {{ 
      opacity: 0, 
      duration: 0.6, 
      ease: "power2.out" 
    }}, "-=0.4");
    
    window.__timelines["stat-reveal"] = tl;
  </script>
</body>
</html>"""


def _generate_countdown_html(brand: str, question: str, count_from: int = 3) -> str:
    """
    Generate HTML for a countdown timer.
    
    Shows number counting down, then reveals question.
    """
    brand_config = BRANDS.get(brand, BRANDS["matchdaymaestro"])
    bg_color = brand_config["bg"]
    accent_color = brand_config["accent"]
    text_color = "#FFFFFF"
    
    # Escape HTML
    question = question.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0;
      overflow: hidden;
      background: {bg_color};
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .scene {{
      width: 1080px;
      height: 1920px;
      position: relative;
    }}
    .content {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
    }}
    .number {{
      font-size: 200px;
      font-weight: 900;
      color: {accent_color};
      line-height: 1;
      margin-bottom: 40px;
      opacity: 0;
    }}
    .question {{
      font-size: 42px;
      font-weight: 600;
      color: {text_color};
      max-width: 800px;
      line-height: 1.4;
      opacity: 0;
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
</head>
<body>
  <div class="scene" 
       data-composition-id="countdown" 
       data-width="1080" 
       data-height="1920" 
       data-duration="{count_from + 2}"
       data-start="0">
    <div class="content">
      <div class="number clip" id="number"></div>
      <div class="question clip" id="question">{question}</div>
    </div>
  </div>
  
  <script>
    window.__timelines = window.__timelines || {{}};
    const tl = gsap.timeline({{ paused: true }});
    
    // Three-number countdown ending at 1 (never below)
    const numbers = [{count_from}, {max(count_from - 1, 1)}, {max(count_from - 2, 1)}];
    numbers.forEach((num, index) => {{
      const start = index;
      const end = index + 1;
      tl.from("#number", {{ 
        text: {{ value: num }},
        duration: 0.5,
        ease: "power2.out"
      }}, start);
      tl.to("#number", {{ 
        opacity: 0,
        duration: 0.2
      }}, end - 0.2);
    }});
    
    // Question: fade in after countdown
    tl.from("#question", {{ 
      opacity: 0,
      duration: 0.8,
      ease: "power3.out"
    }}, "-=0.5");
    
    window.__timelines["countdown"] = tl;
  </script>
</body>
</html>"""


def _generate_screenshot_overlay_html(band: str, overlay_text: str = "", 
                                    screenshot_path: Optional[str] = None) -> str:
    """
    Generate HTML for showcasing a screenshot with animated overlay.
    
    Shows screenshot full-screen with optional text overlay that animates in.
    """
    brand_config = BRANDS.get(band, BRANDS["matchdaymaestro"])
    bg_color = brand_config["bg"]
    accent_color = brand_config["accent"]
    text_color = "#FFFFFF"
    
    # Escape HTML
    overlay_text = overlay_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Default to solid color if no screenshot
    bg_element = f'<div class="screenshot-placeholder"></div>'
    if screenshot_path and Path(screenshot_path).exists():
        # Use the screenshot as background. Escape for attribute safety (paths can contain ' or &).
        safe_path = html.escape(screenshot_path, quote=True)
        bg_element = f'<img class="screenshot-image" src="file://{safe_path}" alt="Screenshot">'
    
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{
      margin: 0;
      overflow: hidden;
      background: {bg_color};
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .scene {{
      width: 1080px;
      height: 1920px;
      position: relative;
      overflow: hidden;
    }}
    .screenshot-image {{
      width: 1080px;
      height: 1920px;
      object-fit: cover;
    }}
    .screenshot-placeholder {{
      width: 1080px;
      height: 1920px;
      background: {bg_color};
    }}
    .overlay {{
      position: absolute;
      bottom: 200px;
      left: 0;
      right: 0;
      text-align: center;
      padding: 0 40px;
    }}
    .overlay-text {{
      font-size: 48px;
      font-weight: 700;
      color: {accent_color};
      text-shadow: 0 2px 4px rgba(0,0,0,0.5);
      margin-bottom: 24px;
      opacity: 0;
    }}
    .brand-tag {{
      font-size: 24px;
      font-weight: 500;
      color: {text_color};
      opacity: 0;
      background: rgba(0,0,0,0.3);
      padding: 8px 16px;
      border-radius: 20px;
      display: inline-block;
    }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>
</head>
<body>
  <div class="scene" 
       data-composition-id="screenshot-overlay" 
       data-width="1080" 
       data-height="1920" 
       data-duration="5"
       data-start="0">
    {bg_element}
    <div class="overlay">
      <div class="overlay-text clip" id="overlay-text">{overlay_text}</div>
      <div class="brand-tag clip" id="brand-tag">@{band}</div>
    </div>
  </div>
  
  <script>
    window.__timelines = window.__timelines || {{}};
    const tl = gsap.timeline({{ paused: true }});
    
    // Overlay text: fade in and slight up movement
    tl.from("#overlay-text", {{ 
      y: 30, 
      opacity: 0, 
      duration: 0.8, 
      ease: "power3.out" 
    }});
    
    // Brand tag: fade in slightly later
    tl.from("#brand-tag", {{ 
      opacity: 0, 
      duration: 0.6, 
      ease: "power2.out" 
    }}, "-=0.4");
    
    window.__timelines["screenshot-overlay"] = tl;
  </script>
</body>
</html>"""


def generate_stat_reveal_video(band: str, stat_line: str, sub_line: str = "",
                              draft_id: Optional[str] = None) -> Optional[str]:
    """
    Generate a stat reveal video using HyperFrames.
    
    Returns path to video file or None on failure.
    """
    global _render_count
    
    if not _can_render():
        print(f"HyperFrames: render limit reached ({MAX_RENDERS_PER_RUN}/run)")
        return None
    
    # Create temp directory for this render
    temp_dir = TEMP_DIR_BASE / f"render_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate HTML
        html_content = _generate_stat_reveal_html(band, stat_line, sub_line)
        html_path = temp_dir / "index.html"
        html_path.write_text(html_content, encoding="utf-8")
        
        # Determine output path
        if not draft_id:
            draft_id = f"hr_{uuid.uuid4().hex[:8]}"
        output_dir = HYPERFRAMES_OUTPUT / band
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{draft_id}_stat_reveal.mp4"
        
        # Run HyperFrames
        start_time = time.time()
        success = _run_hyperframes(html_path.parent, output_path)
        render_time = time.time() - start_time
        
        if success:
            _increment_render_counter()
            file_size_kb = output_path.stat().st_size // 1024
            print(f"HyperFrames stat_reveal: {output_path.name} "
                  f"({file_size_kb}KB, {render_time:.1f}s)")
            return str(output_path)
        else:
            print(f"HyperFrames stat_reveal failed after {render_time:.1f}s")
            return None
            
    finally:
        # Clean up temp directory
        _cleanup_temp_dir(temp_dir)


def generate_countdown_video(band: str, question: str, count_from: int = 3,
                            draft_id: Optional[str] = None) -> Optional[str]:
    """
    Generate a countdown video using HyperFrames.
    
    Returns path to video file or None on failure.
    """
    global _render_count
    
    if not _can_render():
        print(f"HyperFrames: render limit reached ({MAX_RENDERS_PER_RUN}/run)")
        return None
    
    # Create temp directory for this render
    temp_dir = TEMP_DIR_BASE / f"render_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate HTML
        html_content = _generate_countdown_html(band, question, count_from)
        html_path = temp_dir / "index.html"
        html_path.write_text(html_content, encoding="utf-8")
        
        # Determine output path
        if not draft_id:
            draft_id = f"hr_{uuid.uuid4().hex[:8]}"
        output_dir = HYPERFRAMES_OUTPUT / band
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{draft_id}_countdown.mp4"
        
        # Run HyperFrames
        start_time = time.time()
        success = _run_hyperframes(html_path.parent, output_path)
        render_time = time.time() - start_time
        
        if success:
            _increment_render_counter()
            file_size_kb = output_path.stat().st_size // 1024
            print(f"HyperFrames countdown: {output_path.name} "
                  f"({file_size_kb}KB, {render_time:.1f}s)")
            return str(output_path)
        else:
            print(f"HyperFrames countdown failed after {render_time:.1f}s")
            return None
            
    finally:
        # Clean up temp directory
        _cleanup_temp_dir(temp_dir)


def generate_screenshot_overlay_video(band: str, overlay_text: str = "",
                                     screenshot_path: Optional[str] = None,
                                     draft_id: Optional[str] = None) -> Optional[str]:
    """
    Generate a screenshot overlay video using HyperFrames.
    
    Returns path to video file or None on failure.
    """
    global _render_count
    
    if not _can_render():
        print(f"HyperFrames: render limit reached ({MAX_RENDERS_PER_RUN}/run)")
        return None
    
    # Create temp directory for this render
    temp_dir = TEMP_DIR_BASE / f"render_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate HTML
        html_content = _generate_screenshot_overlay_html(band, overlay_text, screenshot_path)
        html_path = temp_dir / "index.html"
        html_path.write_text(html_content, encoding="utf-8")
        
        # Determine output path
        if not draft_id:
            draft_id = f"hr_{uuid.uuid4().hex[:8]}"
        output_dir = HYPERFRAMES_OUTPUT / band
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{draft_id}_screenshot_overlay.mp4"
        
        # Run HyperFrames
        start_time = time.time()
        success = _run_hyperframes(html_path.parent, output_path)
        render_time = time.time() - start_time
        
        if success:
            _increment_render_counter()
            file_size_kb = output_path.stat().st_size // 1024
            print(f"HyperFrames screenshot_overlay: {output_path.name} "
                  f"({file_size_kb}KB, {render_time:.1f}s)")
            return str(output_path)
        else:
            print(f"HyperFrames screenshot_overlay failed after {render_time:.1f}s")
            return None
            
    finally:
        # Clean up temp directory
        _cleanup_temp_dir(temp_dir)


def get_render_count() -> int:
    """Get the number of renders performed this run."""
    global _render_count
    return _render_count


# For testing
if __name__ == "__main__":
    # Simple test
    print("Testing HyperFrames stat reveal...")
    result = generate_stat_reveal_video(
        "matchdaymaestro", 
        "7 GOALS", 
        "Season 2025/26",
        "test_001"
    )
    if result:
        print(f"Success: {result}")
    else:
        print("Failed")
    
    print(f"Render count: {get_render_count()}")