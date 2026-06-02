#!/usr/bin/env python3
"""Build helper script."""
import subprocess
import sys


def run_build():
    subprocess.run([sys.executable, "-m", "build"], check=True)


if __name__ == "__main__":
    run_build()
