#!/usr/bin/env python3
"""
Convert Mermaid diagrams to PNG or SVG images.

This script renders Mermaid diagrams to image files using mermaid-cli (mmdc).
Supports single diagrams, batch conversion, and various rendering options.

Usage:
    # Convert single diagram file
    python mermaid_to_image.py diagram.mmd output.png

    # Convert with custom settings
    python mermaid_to_image.py diagram.mmd output.svg --theme dark --background white --width 1200

    # Batch convert all .mmd files in directory
    python mermaid_to_image.py diagrams/ output/ --format png --recursive

    # Convert from stdin
    echo "graph TD; A-->B" | python mermaid_to_image.py - output.png

Requirements:
    - mermaid-cli: npm install -g @mermaid-js/mermaid-cli
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, List


class MermaidRenderer:
    """Render Mermaid diagrams to images using mermaid-cli."""

    VALID_THEMES = ['default', 'forest', 'dark', 'neutral', 'base']
    VALID_FORMATS = ['png', 'svg', 'pdf']

    def __init__(
        self,
        theme: str = 'default',
        background: str = 'transparent',
        width: Optional[int] = None,
        height: Optional[int] = None,
        scale: int = 1,
        config_file: Optional[Path] = None
    ):
        """
        Initialize Mermaid renderer.

        Args:
            theme: Mermaid theme (default, forest, dark, neutral, base)
            background: Background color (transparent, white, etc.)
            width: Output width in pixels
            height: Output height in pixels
            scale: Scale factor (1-3)
            config_file: Path to custom Mermaid config file
        """
        mmdc_cmd = self._get_mmdc_cmd()
        if not mmdc_cmd:
            print("ERROR: mermaid-cli (mmdc) not found.", file=sys.stderr)
            print("Install with: npm install -g @mermaid-js/mermaid-cli or ensure npx is available", file=sys.stderr)
            sys.exit(1)

        self.theme = theme if theme in self.VALID_THEMES else 'default'
        self.background = background
        self.width = width
        self.height = height
        self.scale = max(1, min(3, scale))
        self.config_file = config_file

    def render(self, input_path: Path, output_path: Path) -> bool:
        """
        Render a Mermaid diagram to an image.

        Args:
            input_path: Path to .mmd file or '-' for stdin
            output_path: Path to output image file

        Returns:
            True if successful, False otherwise
        """
        # Build mmdc command
        mmdc_cmd = self._get_mmdc_cmd() or ['mmdc']
        cmd = list(mmdc_cmd) + ['-i', str(input_path), '-o', str(output_path)]

        # Add options
        cmd.extend(['-t', self.theme])
        cmd.extend(['-b', self.background])

        if self.width:
            cmd.extend(['-w', str(self.width)])

        if self.height:
            cmd.extend(['-H', str(self.height)])

        if self.scale != 1:
            cmd.extend(['-s', str(self.scale)])

        if self.config_file and self.config_file.exists():
            cmd.extend(['-c', str(self.config_file)])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                print(f"ERROR: mmdc failed: {result.stderr}", file=sys.stderr)
                return False

            if not output_path.exists():
                print(f"ERROR: Output file not created: {output_path}", file=sys.stderr)
                return False

            if output_path.stat().st_size == 0:
                print(f"ERROR: Output file is empty: {output_path}", file=sys.stderr)
                return False

            return True

        except subprocess.TimeoutExpired:
            print(f"ERROR: Rendering timed out after 60 seconds", file=sys.stderr)
            return False

        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return False

    def render_from_string(self, mermaid_code: str, output_path: Path) -> bool:
        """
        Render Mermaid code string to an image.

        Args:
            mermaid_code: Mermaid diagram syntax as string
            output_path: Path to output image file

        Returns:
            True if successful, False otherwise
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False) as f:
            f.write(mermaid_code)
            temp_input = Path(f.name)

        try:
            success = self.render(temp_input, output_path)
            return success
        finally:
            try:
                temp_input.unlink()
            except:
                pass

    def batch_render(
        self,
        input_dir: Path,
        output_dir: Path,
        output_format: str = 'png',
        recursive: bool = False
    ) -> tuple[int, int]:
        """
        Batch render all .mmd files in a directory.

        Args:
            input_dir: Directory containing .mmd files
            output_dir: Output directory for images
            output_format: Output format (png, svg, pdf)
            recursive: Recursively search subdirectories

        Returns:
            Tuple of (success_count, total_count)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all .mmd files
        if recursive:
            mmd_files = list(input_dir.rglob('*.mmd'))
        else:
            mmd_files = list(input_dir.glob('*.mmd'))

        if not mmd_files:
            print(f"No .mmd files found in {input_dir}")
            return 0, 0

        print(f"Found {len(mmd_files)} diagram(s) to render\n")

        success_count = 0
        for input_file in mmd_files:
            # Determine output path
            if recursive:
                relative_path = input_file.relative_to(input_dir)
                output_file = output_dir / relative_path.with_suffix(f'.{output_format}')
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                output_file = output_dir / input_file.with_suffix(f'.{output_format}').name

            print(f"  Rendering: {input_file.name} -> {output_file.name}...", end=" ")

            if self.render(input_file, output_file):
                print("✅")
                success_count += 1
            else:
                print("❌")

        print(f"\n✓ Successfully rendered {success_count}/{len(mmd_files)} diagram(s)")
        return success_count, len(mmd_files)

    @staticmethod
    def _get_mmdc_cmd() -> Optional[List[str]]:
        """Find working mmdc command (direct binary or npx fallback)."""
        import shutil
        if shutil.which('mmdc'):
            return ['mmdc']
        if shutil.which('npx'):
            return ['npx', '-y', '@mermaid-js/mermaid-cli']
        return None

    @classmethod
    def _check_mmdc_installed(cls) -> bool:
        """Check if mermaid-cli (mmdc or npx) is available."""
        return cls._get_mmdc_cmd() is not None


def main():
    parser = argparse.ArgumentParser(
        description='Convert Mermaid diagrams to PNG or SVG images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single diagram
  python mermaid_to_image.py diagram.mmd output.png

  # With custom theme and background
  python mermaid_to_image.py diagram.mmd output.svg --theme dark --background white

  # Batch convert directory
  python mermaid_to_image.py diagrams/ output/ --format png

  # From stdin
  echo "graph TD; A-->B" | python mermaid_to_image.py - output.png

Themes:
  default, forest, dark, neutral, base
        """
    )

    parser.add_argument('input', type=str,
                        help='Input file (.mmd), directory, or "-" for stdin')
    parser.add_argument('output', type=str,
                        help='Output file or directory')

    # Rendering options
    parser.add_argument('--theme', '-t', choices=MermaidRenderer.VALID_THEMES,
                        default='default', help='Mermaid theme (default: default)')
    parser.add_argument('--background', '-b', default='transparent',
                        help='Background color (default: transparent)')
    parser.add_argument('--width', '-w', type=int,
                        help='Output width in pixels')
    parser.add_argument('--height', '-H', type=int,
                        help='Output height in pixels')
    parser.add_argument('--scale', '-s', type=int, default=1, choices=[1, 2, 3],
                        help='Scale factor (default: 1)')
    parser.add_argument('--config', '-c', type=Path,
                        help='Path to custom Mermaid config file')

    # Batch options
    parser.add_argument('--format', '-f', choices=MermaidRenderer.VALID_FORMATS,
                        default='png', help='Output format for batch conversion (default: png)')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help='Recursively process subdirectories')

    args = parser.parse_args()

    # Initialize renderer
    renderer = MermaidRenderer(
        theme=args.theme,
        background=args.background,
        width=args.width,
        height=args.height,
        scale=args.scale,
        config_file=args.config
    )

    # Handle stdin input
    if args.input == '-':
        if not sys.stdin.isatty():
            mermaid_code = sys.stdin.read()
            output_path = Path(args.output)
            print(f"Rendering from stdin to {output_path}...")

            if renderer.render_from_string(mermaid_code, output_path):
                print(f"✅ Success: {output_path}")
                sys.exit(0)
            else:
                print("❌ Failed", file=sys.stderr)
                sys.exit(1)
        else:
            print("ERROR: No input provided on stdin", file=sys.stderr)
            sys.exit(1)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Handle directory (batch mode)
    if input_path.is_dir():
        success, total = renderer.batch_render(
            input_path,
            output_path,
            output_format=args.format,
            recursive=args.recursive
        )
        sys.exit(0 if success == total else 1)

    # Handle single file
    elif input_path.is_file():
        print(f"Rendering: {input_path} -> {output_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if renderer.render(input_path, output_path):
            print(f"✅ Success: {output_path}")
            print(f"   Size: {output_path.stat().st_size:,} bytes")
            sys.exit(0)
        else:
            print("❌ Failed", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
