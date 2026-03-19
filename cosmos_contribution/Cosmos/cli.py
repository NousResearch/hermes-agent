#!/usr/bin/env python3
"""
cosmos CLI Entry Point

This module provides the entry point for the 'cosmos' command
when installed via pip install -e .
"""

import sys
from pathlib import Path

def main():
    """Main entry point for cosmos command."""
    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import and run the main function
    try:
        from main import main as run_main
        run_main()
    except ImportError:
        # Fallback: try running from the parent directory
        import os
        os.chdir(project_root)
        sys.path.insert(0, str(project_root))
        from main import main as run_main
        run_main()


if __name__ == "__main__":
    main()
