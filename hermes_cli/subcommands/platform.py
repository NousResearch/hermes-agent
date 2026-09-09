"""
CLI Subcommand: hermes platform
Inspects current platform, hardware capabilities, and adapter status.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from hermes_platform.factory import get_platform_adapter, detect_platform_type
from core.models import ModelManager


def main(args=None):
    adapter = get_platform_adapter()
    ptype = detect_platform_type()
    mm = ModelManager()
    hw = mm.hardware

    print("=== Hermes Platform Information ===")
    print(f"Platform Type:    {ptype.value}")
    print(f"Adapter Class:    {adapter.__class__.__name__}")
    print(f"OS:               {hw.os}")
    print(f"Architecture:     {hw.architecture}")
    print(f"CPU Count:        {hw.cpu_count}")
    print(f"RAM:              {hw.ram_gb} GB")
    print(f"Storage:          {hw.available_storage_gb} GB")
    print(f"GPU Available:    {hw.gpu_available}")
    if hw.gpu_name:
        print(f"GPU Name:         {hw.gpu_name}")
        print(f"VRAM:             {hw.vram_gb} GB")
    print("===================================")


if __name__ == "__main__":
    main()
