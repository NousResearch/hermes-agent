#!/usr/bin/env python3
"""
Extract Mermaid diagrams from Markdown files.

This script finds all Mermaid code blocks in Markdown files and can:
1. Extract them to separate .mmd files
2. Validate syntax by attempting to render
3. Replace them with image references
4. List all diagrams with metadata

Usage:
    # Extract all diagrams to separate files
    python extract_mermaid.py document.md --output-dir diagrams/

    # List all diagrams without extracting
    python extract_mermaid.py document.md --list-only

    # Validate diagrams (requires mmdc installed)
    python extract_mermaid.py document.md --validate

    # Replace diagrams with image references
    python extract_mermaid.py document.md --replace-with-images --image-format png

Requirements:
    - For validation: mermaid-cli (npm install -g @mermaid-js/mermaid-cli)
"""

import argparse
import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import hashlib


class MermaidDiagram:
    """Represents a single Mermaid diagram extracted from Markdown."""

    def __init__(self, content: str, line_number: int, index: int):
        self.content = content.strip()
        self.line_number = line_number
        self.index = index
        self.hash = hashlib.md5(content.encode()).hexdigest()[:8]

    def get_filename(self, prefix: str = "diagram", extension: str = "mmd") -> str:
        """Generate a unique filename for this diagram."""
        return f"{prefix}-{self.index:03d}-{self.hash}.{extension}"

    def get_first_line(self, max_length: int = 50) -> str:
        """Get the first line of the diagram for display."""
        first_line = self.content.split('\n')[0].strip()
        if len(first_line) > max_length:
            return first_line[:max_length] + "..."
        return first_line


class MermaidExtractor:
    """Extract and process Mermaid diagrams from Markdown files."""

    # Pattern to match Mermaid code blocks
    MERMAID_PATTERN = re.compile(
        r'```mermaid\s*\n(.*?)```',
        re.DOTALL | re.MULTILINE
    )

    def __init__(self, markdown_file: Path):
        self.markdown_file = markdown_file
        self.content = markdown_file.read_text(encoding='utf-8')
        self.diagrams: List[MermaidDiagram] = []
        self._extract_diagrams()

    def _extract_diagrams(self):
        """Extract all Mermaid diagrams from the Markdown content."""
        line_number = 1
        for index, match in enumerate(self.MERMAID_PATTERN.finditer(self.content), start=1):
            diagram_content = match.group(1)
            # Count newlines before this match to get line number
            lines_before = self.content[:match.start()].count('\n')
            diagram = MermaidDiagram(diagram_content, lines_before + 1, index)
            self.diagrams.append(diagram)

    def save_diagrams(self, output_dir: Path, prefix: str = "diagram") -> List[Path]:
        """
        Save all diagrams to separate .mmd files.

        Args:
            output_dir: Directory to save diagram files
            prefix: Prefix for diagram filenames

        Returns:
            List of paths to saved diagram files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []

        for diagram in self.diagrams:
            filename = diagram.get_filename(prefix=prefix, extension="mmd")
            output_path = output_dir / filename
            output_path.write_text(diagram.content, encoding='utf-8')
            saved_files.append(output_path)
            print(f"  ✓ Saved: {output_path}")

        return saved_files

    def list_diagrams(self):
        """Print a list of all diagrams with metadata."""
        if not self.diagrams:
            print("No Mermaid diagrams found.")
            return

        print(f"\nFound {len(self.diagrams)} Mermaid diagram(s) in {self.markdown_file}:\n")
        for diagram in self.diagrams:
            print(f"  #{diagram.index} (Line {diagram.line_number}):")
            print(f"    First line: {diagram.get_first_line()}")
            print(f"    Hash: {diagram.hash}")
            print(f"    Lines: {len(diagram.content.splitlines())}")
            print()

    def validate_diagrams(self) -> Dict[int, Optional[str]]:
        """
        Validate all diagrams by attempting to render them with mmdc.

        Returns:
            Dict mapping diagram index to error message (None if valid)
        """
        if not self._check_mmdc_installed():
            print("ERROR: mermaid-cli (mmdc) not found.", file=sys.stderr)
            print("Install with: npm install -g @mermaid-js/mermaid-cli", file=sys.stderr)
            sys.exit(1)

        results = {}
        print(f"\nValidating {len(self.diagrams)} diagram(s)...\n")

        for diagram in self.diagrams:
            print(f"  Validating diagram #{diagram.index}...", end=" ")
            error = self._validate_single_diagram(diagram)
            results[diagram.index] = error

            if error:
                print(f"❌ FAILED")
                print(f"    Error: {error}")
            else:
                print(f"✅ OK")

        # Summary
        failed_count = sum(1 for err in results.values() if err)
        print(f"\nValidation complete: {len(self.diagrams) - failed_count}/{len(self.diagrams)} passed")

        return results

    def _validate_single_diagram(self, diagram: MermaidDiagram) -> Optional[str]:
        """Validate a single diagram. Returns error message if invalid, None if valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_file = tmpdir_path / f"diagram-{diagram.index}.mmd"
            output_file = tmpdir_path / f"diagram-{diagram.index}.svg"

            # Write diagram to temp file
            input_file.write_text(diagram.content, encoding='utf-8')

            try:
                mmdc_cmd = self._get_mmdc_cmd() or ['mmdc']
                cmd = mmdc_cmd + ['-i', str(input_file), '-o', str(output_file), '-b', 'transparent']
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    return result.stderr.strip() or "Unknown rendering error"

                if not output_file.exists() or output_file.stat().st_size == 0:
                    return "Rendering produced no output"

                return None  # Valid

            except subprocess.TimeoutExpired:
                return "Rendering timed out after 30 seconds"
            except Exception as e:
                return str(e)

    def replace_with_images(self, image_format: str = "png", image_dir: str = "diagrams") -> str:
        """
        Replace Mermaid code blocks with image references.

        Args:
            image_format: Image format (png or svg)
            image_dir: Directory path for images (relative to markdown file)

        Returns:
            Modified Markdown content
        """
        def replace_block(match):
            diagram_content = match.group(1)
            # Find which diagram this is
            for diagram in self.diagrams:
                if diagram.content == diagram_content.strip():
                    filename = diagram.get_filename(extension=image_format)
                    image_path = f"{image_dir}/{filename}"
                    return f"![Diagram {diagram.index}]({image_path})"
            return match.group(0)  # If not found, leave unchanged

        return self.MERMAID_PATTERN.sub(replace_block, self.content)

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
        description='Extract Mermaid diagrams from Markdown files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract diagrams to separate files
  python extract_mermaid.py document.md --output-dir diagrams/

  # List all diagrams
  python extract_mermaid.py document.md --list-only

  # Validate diagrams
  python extract_mermaid.py document.md --validate

  # Replace with image references
  python extract_mermaid.py document.md --replace-with-images --image-format png
        """
    )

    parser.add_argument('markdown_file', type=Path, help='Input Markdown file')
    parser.add_argument('--output-dir', '-o', type=Path, help='Output directory for extracted diagrams')
    parser.add_argument('--prefix', default='diagram', help='Prefix for output filenames (default: diagram)')
    parser.add_argument('--list-only', '-l', action='store_true', help='List diagrams without extracting')
    parser.add_argument('--validate', '-v', action='store_true', help='Validate diagrams with mmdc')
    parser.add_argument('--replace-with-images', '-r', action='store_true',
                        help='Replace Mermaid blocks with image references')
    parser.add_argument('--image-format', choices=['png', 'svg'], default='png',
                        help='Image format for replacement (default: png)')
    parser.add_argument('--image-dir', default='docs/architecture',
                        help='Image directory path for references (default: docs/architecture)')
    parser.add_argument('--output-markdown', type=Path,
                        help='Output file for modified markdown (with --replace-with-images)')

    args = parser.parse_args()

    # Validate input file
    if not args.markdown_file.exists():
        print(f"ERROR: File not found: {args.markdown_file}", file=sys.stderr)
        sys.exit(1)

    if not args.markdown_file.is_file():
        print(f"ERROR: Not a file: {args.markdown_file}", file=sys.stderr)
        sys.exit(1)

    # Extract diagrams
    print(f"Processing: {args.markdown_file}")
    extractor = MermaidExtractor(args.markdown_file)

    if not extractor.diagrams:
        print("No Mermaid diagrams found.")
        sys.exit(0)

    # Handle operations
    if args.list_only:
        extractor.list_diagrams()

    elif args.validate:
        results = extractor.validate_diagrams()
        # Exit with error if any validation failed
        if any(results.values()):
            sys.exit(1)

    elif args.replace_with_images:
        modified_content = extractor.replace_with_images(
            image_format=args.image_format,
            image_dir=args.image_dir
        )

        if args.output_markdown:
            args.output_markdown.write_text(modified_content, encoding='utf-8')
            print(f"\n✓ Modified Markdown saved to: {args.output_markdown}")
        else:
            print("\nModified Markdown:\n")
            print(modified_content)

    elif args.output_dir:
        print(f"\nExtracting {len(extractor.diagrams)} diagram(s) to {args.output_dir}/:\n")
        saved_files = extractor.save_diagrams(args.output_dir, prefix=args.prefix)
        print(f"\n✓ Extracted {len(saved_files)} diagram(s)")

    else:
        # Default: list diagrams
        extractor.list_diagrams()
        print("\nUse --output-dir to extract diagrams or --help for more options.")


if __name__ == '__main__':
    main()
