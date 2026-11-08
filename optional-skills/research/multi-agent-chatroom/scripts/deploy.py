#!/usr/bin/env python3
"""Deploy the multi-agent-chatroom skill to a target directory.

Usage:
    python scripts/deploy.py /path/to/target
    python scripts/deploy.py /mnt/e/develop/OpenOne/multi-agent-chatroom

This script reads all source files from the skill's templates/ directory
and deploys them to the target location, preserving directory structure.
"""

import os
import sys
import shutil
from pathlib import Path


def deploy(target_dir: str):
    """Deploy the multi-agent-chatroom to target directory."""
    target = Path(target_dir).resolve()
    templates = Path(__file__).parent.parent / "templates"

    if not templates.exists():
        print(f"❌ Templates not found at {templates}")
        sys.exit(1)

    print(f"🚀 Deploying Multi-Agent Research Chatroom")
    print(f"   Source: {templates}")
    print(f"   Target: {target}")
    print()

    # Create target directory
    target.mkdir(parents=True, exist_ok=True)

    files_copied = 0
    for root, dirs, files in os.walk(templates):
        rel_root = Path(root).relative_to(templates)
        for f in files:
            src = Path(root) / f
            rel = rel_root / f
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files_copied += 1
            print(f"  ✅ {rel}")

    print(f"\n📦 Deployed {files_copied} files to {target}")

    # Post-deploy checks
    req_file = target / "requirements.txt"
    config_file = target / "config.yaml"
    launch_file = target / "launch.sh"

    print("\n📋 Post-deploy checks:")
    print(f"   requirements.txt: {'✅' if req_file.exists() else '❌'} ")
    print(f"   config.yaml:      {'✅' if config_file.exists() else '❌'} ")
    print(f"   launch.sh:        {'✅' if launch_file.exists() else '❌'} ")

    print("\n📝 Next steps:")
    print(f"   1. cd {target}")
    print(f"   2. pip install -r requirements.txt")
    print(f"   3. Set API keys as environment variables:")
    print(f"      DEEPSEEK_API_KEY=sk-...")
    print(f"      OPENAI_API_KEY=sk-...")
    print(f"      ANTHROPIC_API_KEY=sk-ant-...")
    print(f"   4. Edit config.yaml → project.workdir to your research repo")
    print(f"   5. python -m pytest tests/ -v")
    print(f"   6. bash launch.sh")

    # Try to set launch.sh executable
    try:
        os.chmod(launch_file, 0o755)
    except OSError:
        pass  # NTFS may not support chmod

    return target


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/deploy.py <target_directory>")
        print("Example: python scripts/deploy.py /mnt/e/develop/OpenOne/chatroom")
        sys.exit(1)

    deploy(sys.argv[1])
