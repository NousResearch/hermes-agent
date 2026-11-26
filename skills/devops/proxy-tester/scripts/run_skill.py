#!/usr/bin/env python3
"""
Hermes skill wrapper for proxy-tester.
Usage: hermes skill proxy-tester test <proxy_file> [options]
"""

import sys
from pathlib import Path

skill_scripts = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(skill_scripts))

from proxy_tester import main

if __name__ == "__main__":
    main()
