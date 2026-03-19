"""
Compile Protected Core Module

This script compiles the core.py module to prevent source code distribution.
The compiled .pyd/.so file should be distributed instead of the source.

Methods available:
1. Cython - Compiles to C extension (.pyd on Windows, .so on Linux)
2. PyArmor - Obfuscates Python bytecode
3. Nuitka - Compiles to standalone binary

Usage:
    python compile.py cython    # Recommended for speed
    python compile.py pyarmor   # Recommended for protection
    python compile.py nuitka    # Recommended for standalone
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CORE_FILE = SCRIPT_DIR / "core.py"
DIST_DIR = SCRIPT_DIR / "dist"


def compile_cython():
    """Compile using Cython to C extension."""
    print("=== Compiling with Cython ===\n")

    # Check Cython installed
    try:
        import Cython
        print(f"Cython version: {Cython.__version__}")
    except ImportError:
        print("ERROR: Cython not installed. Run: pip install cython")
        return False

    # Create setup.py
    setup_content = '''
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        "core.py",
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
        }
    ),
    script_args=['build_ext', '--inplace']
)
'''

    setup_file = SCRIPT_DIR / "setup_cython.py"
    with open(setup_file, 'w') as f:
        f.write(setup_content)

    # Run compilation
    result = subprocess.run(
        [sys.executable, str(setup_file)],
        cwd=str(SCRIPT_DIR),
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print("ERRORS:")
        print(result.stderr)
        return False

    # Cleanup
    setup_file.unlink()

    # Move compiled file to dist
    DIST_DIR.mkdir(exist_ok=True)
    for ext in ['.pyd', '.so']:
        for f in SCRIPT_DIR.glob(f"core*{ext}"):
            dest = DIST_DIR / f.name
            shutil.move(str(f), str(dest))
            print(f"\nCompiled: {dest}")

    # Remove intermediate files
    for f in SCRIPT_DIR.glob("core.c"):
        f.unlink()
    for d in SCRIPT_DIR.glob("build"):
        shutil.rmtree(d, ignore_errors=True)

    print("\n=== Cython compilation complete ===")
    print(f"Distribute files from: {DIST_DIR}")
    return True


def compile_pyarmor():
    """Obfuscate using PyArmor."""
    print("=== Obfuscating with PyArmor ===\n")

    # Check PyArmor installed
    result = subprocess.run(
        [sys.executable, "-m", "pyarmor", "--version"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: PyArmor not installed. Run: pip install pyarmor")
        return False

    print(result.stdout.strip())

    # Run obfuscation
    DIST_DIR.mkdir(exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "pyarmor", "gen",
            "--output", str(DIST_DIR),
            str(CORE_FILE)
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print("ERRORS:")
        print(result.stderr)
        return False

    print("\n=== PyArmor obfuscation complete ===")
    print(f"Distribute files from: {DIST_DIR}")
    return True


def compile_nuitka():
    """Compile using Nuitka to standalone binary."""
    print("=== Compiling with Nuitka ===\n")

    # Check Nuitka installed
    result = subprocess.run(
        [sys.executable, "-m", "nuitka", "--version"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("ERROR: Nuitka not installed. Run: pip install nuitka")
        return False

    print(result.stdout.strip())

    # Run compilation
    DIST_DIR.mkdir(exist_ok=True)

    result = subprocess.run(
        [
            sys.executable, "-m", "nuitka",
            "--module",
            "--output-dir=" + str(DIST_DIR),
            str(CORE_FILE)
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print("ERRORS:")
        print(result.stderr)
        return False

    print("\n=== Nuitka compilation complete ===")
    print(f"Distribute files from: {DIST_DIR}")
    return True


def create_stub():
    """Create a stub file that imports from compiled module."""
    stub_content = '''"""
Protected Core Module Stub

This stub imports from the compiled core module.
The actual implementation is in the compiled .pyd/.so file.
"""

try:
    # Try to import compiled module
    from .core import *
except ImportError:
    raise ImportError(
        "Protected core module not found. "
        "Please ensure the compiled core.pyd/.so file is present."
    )
'''

    stub_file = DIST_DIR / "core_stub.py"
    with open(stub_file, 'w') as f:
        f.write(stub_content)

    print(f"Created stub: {stub_file}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    method = sys.argv[1].lower()

    if method == "cython":
        if compile_cython():
            create_stub()
    elif method == "pyarmor":
        compile_pyarmor()
    elif method == "nuitka":
        if compile_nuitka():
            create_stub()
    else:
        print(f"Unknown method: {method}")
        print("Valid methods: cython, pyarmor, nuitka")


if __name__ == "__main__":
    main()
