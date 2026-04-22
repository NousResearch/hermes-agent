"""
ComfyUI-VolcEngine install script

This script is automatically called by ComfyUI-Manager to:
1. Install Python dependencies from requirements.txt
2. Run post-install verification
"""

import subprocess
import sys
import os


def install():
    """Install dependencies and verify setup."""
    print("[ComfyUI-VolcEngine] Installing dependencies...")
    
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(script_dir, "requirements.txt")
    
    if os.path.exists(requirements_file):
        print(f"[ComfyUI-VolcEngine] Found requirements.txt at {requirements_file}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", requirements_file
            ], cwd=script_dir)
            print("[ComfyUI-VolcEngine] Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-VolcEngine] Warning: Failed to install dependencies: {e}")
    else:
        print("[ComfyUI-VolcEngine] Warning: requirements.txt not found.")
    
    # Verify critical dependencies
    print("[ComfyUI-VolcEngine] Verifying dependencies...")
    try:
        import imageio
        import imageio_ffmpeg
        print(f"[ComfyUI-VolcEngine] imageio version: {imageio.__version__}")
        print("[ComfyUI-VolcEngine] imageio-ffmpeg: OK")
    except ImportError as e:
        print(f"[ComfyUI-VolcEngine] Warning: {e}")
    
    # Check for ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            print("[ComfyUI-VolcEngine] ffmpeg: OK")
        else:
            print("[ComfyUI-VolcEngine] Warning: ffmpeg returned non-zero exit code")
    except FileNotFoundError:
        print("[ComfyUI-VolcEngine] Warning: ffmpeg not found in PATH.")
        print("[ComfyUI-VolcEngine] Video processing may not work without ffmpeg.")
        print("[ComfyUI-VolcEngine] Install it via: sudo apt install ffmpeg (Debian/Ubuntu)")
    except subprocess.TimeoutExpired:
        print("[ComfyUI-VolcEngine] Warning: ffmpeg command timed out")
    
    print("[ComfyUI-VolcEngine] Installation complete.")


def uninstall():
    """Uninstall dependencies."""
    print("[ComfyUI-VolcEngine] Uninstalling dependencies...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(script_dir, "requirements.txt")
    
    if os.path.exists(requirements_file):
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "uninstall", "-y", "-r", requirements_file
            ], cwd=script_dir)
            print("[ComfyUI-VolcEngine] Dependencies uninstalled.")
        except subprocess.CalledProcessError as e:
            print(f"[ComfyUI-VolcEngine] Warning: Failed to uninstall dependencies: {e}")


if __name__ == "__main__":
    if "--uninstall" in sys.argv:
        uninstall()
    else:
        install()
