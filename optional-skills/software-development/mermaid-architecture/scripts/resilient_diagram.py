#!/usr/bin/env python3
"""
Resilient Mermaid diagram generation with error recovery.

This script orchestrates the full workflow for creating Mermaid diagrams:
1. Identify diagram type from Mermaid code
2. Save .mmd file with consistent naming convention
3. Generate image file (PNG/SVG) using mmdc
4. On validation failure, search troubleshooting.md for matching fixes
5. Return structured result for Claude to act on errors

The script enables resilient diagram generation by:
- Automatically detecting diagram types
- Following consistent file naming: <markdown>_<num>_<type>_<title>.<ext>
- Parsing troubleshooting.md to find error fixes
- Generating search queries for external tools when no local fix exists

Usage:
    # Generate from code string
    python resilient_diagram.py --code "flowchart TD; A-->B" \\
        --markdown-file design_doc --diagram-num 1 --title "overview"

    # Generate from existing .mmd file
    python resilient_diagram.py --mmd-file diagram.mmd --output-dir ./diagrams/

    # Read from stdin with JSON output
    cat diagram.mmd | python resilient_diagram.py --stdin \\
        --markdown-file doc --diagram-num 2 --title "flow" --json

Requirements:
    - mermaid-cli (npm install -g @mermaid-js/mermaid-cli)
    - Python 3.7+ (stdlib only, no external dependencies)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any


class DiagramType(Enum):
    """Supported Mermaid diagram types."""
    FLOWCHART = "flowchart"
    SEQUENCE = "sequence"
    CLASS = "class"
    STATE = "state"
    ER = "er"
    GANTT = "gantt"
    PIE = "pie"
    MINDMAP = "mindmap"
    TIMELINE = "timeline"
    QUADRANT = "quadrant"
    REQUIREMENT = "requirement"
    JOURNEY = "journey"
    C4 = "c4"
    UNKNOWN = "unknown"


@dataclass
class TroubleshootingMatch:
    """A matching entry from troubleshooting guide."""
    error_number: int
    title: str
    severity: str
    diagram_types: List[str]
    problem: str
    incorrect_example: str
    correct_example: str
    error_messages: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_number": self.error_number,
            "title": self.title,
            "severity": self.severity,
            "diagram_types": self.diagram_types,
            "problem": self.problem,
            "correct_example": self.correct_example
        }


@dataclass
class DiagramResult:
    """Result of diagram generation attempt."""
    success: bool
    mmd_path: Optional[str]
    image_path: Optional[str]
    diagram_type: str
    error_message: Optional[str]
    troubleshooting_matches: List[Dict] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    search_recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "mmd_path": self.mmd_path,
            "image_path": self.image_path,
            "diagram_type": self.diagram_type,
            "error_message": self.error_message,
            "troubleshooting_matches": self.troubleshooting_matches,
            "suggested_fix": self.suggested_fix,
            "search_recommendation": self.search_recommendation
        }


class TroubleshootingParser:
    """Parse troubleshooting.md into searchable entries."""

    # Pattern to match error sections
    ERROR_PATTERN = re.compile(
        r'### ❌ Error (\d+): (.+?)\n+'
        r'\*\*Severity:\*\* (🔴|🟠|🟡|🟢) (\w+)',
        re.MULTILINE
    )

    # Pattern to find incorrect/correct code blocks
    CODE_BLOCK_PATTERN = re.compile(
        r'\*\*(Incorrect|Correct)(?:\s+Solutions?)?:\*\*\s*```mermaid\s*\n(.*?)```',
        re.DOTALL
    )

    def __init__(self, troubleshooting_path: Path):
        self.path = troubleshooting_path
        self.entries: List[TroubleshootingMatch] = []
        if self.path and self.path.exists():
            self._parse()

    def _parse(self):
        """Parse the troubleshooting.md file."""
        content = self.path.read_text(encoding='utf-8')

        # Split by error sections
        sections = re.split(r'(?=### ❌ Error \d+:)', content)

        for section in sections:
            if not section.strip() or '### ❌ Error' not in section:
                continue

            # Extract error number and title
            header_match = re.search(r'### ❌ Error (\d+): (.+)', section)
            if not header_match:
                continue

            error_num = int(header_match.group(1))
            title = header_match.group(2).strip()

            # Extract severity
            severity_match = re.search(r'\*\*Severity:\*\* (🔴|🟠|🟡|🟢) (\w+)', section)
            severity = severity_match.group(2) if severity_match else "Unknown"

            # Extract problem description
            problem_match = re.search(r'\*\*Problem:\*\* (.+?)(?:\n\n|\*\*)', section, re.DOTALL)
            problem = problem_match.group(1).strip() if problem_match else ""

            # Extract diagram types affected
            types_match = re.search(r'\*\*Diagram Types Affected:\*\* (.+)', section)
            diagram_types = []
            if types_match:
                types_text = types_match.group(1)
                # Parse types like "All diagrams", "Flowcharts, state diagrams"
                diagram_types = [t.strip().lower() for t in types_text.split(',')]

            # Extract incorrect/correct examples
            incorrect = ""
            correct = ""
            for match in self.CODE_BLOCK_PATTERN.finditer(section):
                block_type = match.group(1)
                code = match.group(2).strip()
                if block_type == "Incorrect":
                    incorrect = code
                elif block_type == "Correct":
                    correct = code

            # Extract error messages
            error_messages = []
            error_msg_match = re.search(r'\*\*Error Message[s]?:\*\*\s*\n((?:- `.+`\n?)+)', section)
            if error_msg_match:
                for line in error_msg_match.group(1).split('\n'):
                    msg_match = re.search(r'`(.+)`', line)
                    if msg_match:
                        error_messages.append(msg_match.group(1))

            # Also look for inline error messages
            inline_msg = re.search(r'\*\*Error Message:\*\*\s*`?(.+?)`?\n', section)
            if inline_msg and not error_messages:
                error_messages.append(inline_msg.group(1).strip('`'))

            entry = TroubleshootingMatch(
                error_number=error_num,
                title=title,
                severity=severity,
                diagram_types=diagram_types,
                problem=problem,
                incorrect_example=incorrect,
                correct_example=correct,
                error_messages=error_messages
            )
            self.entries.append(entry)

    def search(self, error_message: str, diagram_type: DiagramType) -> List[TroubleshootingMatch]:
        """
        Search for matching troubleshooting entries.

        Args:
            error_message: Error message from mmdc
            diagram_type: Type of diagram that failed

        Returns:
            List of matching entries, ranked by relevance
        """
        matches = []
        error_lower = error_message.lower()
        type_name = diagram_type.value.lower()

        for entry in self.entries:
            score = 0

            # Check error message matches
            for known_error in entry.error_messages:
                if known_error.lower() in error_lower:
                    score += 10
                elif any(word in error_lower for word in known_error.lower().split()):
                    score += 3

            # Check diagram type matches
            if "all" in ' '.join(entry.diagram_types).lower():
                score += 2
            elif any(type_name in t for t in entry.diagram_types):
                score += 5

            # Check problem keywords
            problem_keywords = entry.problem.lower().split()
            for keyword in ['reserved', 'missing', 'invalid', 'incorrect', 'error', 'syntax']:
                if keyword in error_lower and keyword in problem_keywords:
                    score += 2

            # Check title keywords
            title_keywords = entry.title.lower().split()
            for keyword in title_keywords:
                if keyword in error_lower:
                    score += 2

            if score > 0:
                matches.append((score, entry))

        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)

        return [entry for _, entry in matches[:5]]  # Return top 5 matches


class ResilientDiagramGenerator:
    """
    Generate Mermaid diagrams with resilient error recovery.

    Workflow:
    1. Parse and identify diagram type
    2. Generate filename using convention
    3. Save .mmd file
    4. Run mmdc to generate image
    5. On error, search troubleshooting.md
    6. Return structured result with recovery info
    """

    # Patterns to detect diagram type from first line
    DIAGRAM_TYPE_PATTERNS = {
        DiagramType.FLOWCHART: [
            r'^flowchart\s+(TB|TD|BT|RL|LR)',
            r'^graph\s+(TB|TD|BT|RL|LR)',
        ],
        DiagramType.SEQUENCE: [r'^sequenceDiagram'],
        DiagramType.CLASS: [r'^classDiagram'],
        DiagramType.STATE: [r'^stateDiagram(-v2)?'],
        DiagramType.ER: [r'^erDiagram'],
        DiagramType.GANTT: [r'^gantt'],
        DiagramType.PIE: [r'^pie'],
        DiagramType.MINDMAP: [r'^mindmap'],
        DiagramType.TIMELINE: [r'^timeline'],
        DiagramType.QUADRANT: [r'^quadrantChart'],
        DiagramType.REQUIREMENT: [r'^requirementDiagram'],
        DiagramType.JOURNEY: [r'^journey'],
        DiagramType.C4: [r'^C4Context', r'^C4Container', r'^C4Component', r'^C4Deployment'],
    }

    def __init__(self, troubleshooting_path: Optional[Path] = None):
        """
        Initialize generator.

        Args:
            troubleshooting_path: Path to troubleshooting.md guide (auto-detected if not provided)
        """
        self.troubleshooting_path = troubleshooting_path or self._find_troubleshooting_guide()
        self.troubleshooting = TroubleshootingParser(self.troubleshooting_path) if self.troubleshooting_path else None

    def _find_troubleshooting_guide(self) -> Optional[Path]:
        """Find troubleshooting.md relative to script location."""
        script_dir = Path(__file__).parent
        guide_path = script_dir.parent / "references" / "guides" / "troubleshooting.md"
        return guide_path if guide_path.exists() else None

    def detect_diagram_type(self, mermaid_code: str) -> DiagramType:
        """
        Detect diagram type from Mermaid code.

        Args:
            mermaid_code: Raw Mermaid diagram code

        Returns:
            DiagramType enum value
        """
        # Get first non-empty, non-comment line
        lines = mermaid_code.strip().split('\n')
        first_line = ""
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('%%'):
                first_line = stripped
                break

        for diagram_type, patterns in self.DIAGRAM_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, first_line, re.IGNORECASE):
                    return diagram_type

        return DiagramType.UNKNOWN

    def generate_filename(
        self,
        markdown_file: str,
        diagram_num: int,
        diagram_type: str,
        title: str
    ) -> str:
        """
        Generate filename following convention:
        <markdown_file_name>_<diagram_num>_<image_type>_<title>

        Args:
            markdown_file: Source markdown filename (without extension)
            diagram_num: Diagram number in the document
            diagram_type: Detected diagram type
            title: User-provided title

        Returns:
            Base filename string (without extension)
        """
        # Sanitize markdown file name
        safe_md = re.sub(r'[^a-zA-Z0-9_-]', '_', markdown_file.lower())
        safe_md = re.sub(r'_+', '_', safe_md).strip('_')

        # Sanitize title: lowercase, replace spaces/special chars with underscores
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title.lower())
        safe_title = re.sub(r'_+', '_', safe_title).strip('_')
        # Truncate to first 20 chars for reasonable filename length
        safe_title = safe_title[:20].rstrip('_')

        return f"{safe_md}_{diagram_num:02d}_{diagram_type}_{safe_title}"

    def save_mmd_file(
        self,
        mermaid_code: str,
        output_dir: Path,
        base_filename: str
    ) -> Path:
        """
        Save diagram to .mmd file.

        Args:
            mermaid_code: Mermaid diagram code
            output_dir: Directory for output files
            base_filename: Base filename without extension

        Returns:
            Path to saved .mmd file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        mmd_path = output_dir / f"{base_filename}.mmd"
        mmd_path.write_text(mermaid_code.strip() + '\n', encoding='utf-8')

        return mmd_path

    def render_image(
        self,
        mmd_path: Path,
        image_format: str = "png"
    ) -> Tuple[bool, Optional[Path], Optional[str]]:
        """
        Render diagram to image using mmdc.

        Args:
            mmd_path: Path to .mmd file
            image_format: Output format (png, svg, pdf)

        Returns:
            Tuple of (success, image_path, error_message)
        """
        image_path = mmd_path.with_suffix(f".{image_format}")

        # Check mmdc or npx is installed
        mmdc_cmd = self._get_mmdc_cmd()
        if not mmdc_cmd:
            return False, None, "mmdc not found. Install with: npm install -g @mermaid-js/mermaid-cli or ensure npx is available"

        try:
            cmd = mmdc_cmd + ['-i', str(mmd_path), '-o', str(image_path), '-b', 'transparent']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown rendering error"
                return False, None, error_msg

            if not image_path.exists():
                return False, None, f"Output file not created: {image_path}"

            if image_path.stat().st_size == 0:
                return False, None, f"Output file is empty: {image_path}"

            return True, image_path, None

        except subprocess.TimeoutExpired:
            return False, None, "Rendering timed out after 60 seconds"
        except Exception as e:
            return False, None, str(e)

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

    def get_search_recommendation(
        self,
        error_message: str,
        diagram_type: DiagramType
    ) -> str:
        """
        Generate search query for external tools when troubleshooting fails.

        Priority order for search tools:
        1. perplexity_ask MCP
        2. brave_web_search MCP
        3. gemini skill
        4. WebSearch tool

        Returns:
            Search query string
        """
        # Clean up error message for search
        clean_error = re.sub(r'\s+', ' ', error_message[:150]).strip()
        clean_error = re.sub(r'[^\w\s:.-]', '', clean_error)

        return f"mermaid {diagram_type.value} diagram syntax error: {clean_error}"

    def generate(
        self,
        mermaid_code: str,
        markdown_file: str,
        diagram_num: int,
        title: str,
        output_dir: Path,
        image_format: str = "png"
    ) -> DiagramResult:
        """
        Execute full resilient generation workflow.

        Args:
            mermaid_code: Mermaid diagram code
            markdown_file: Source markdown filename
            diagram_num: Diagram number
            title: Diagram title
            output_dir: Output directory
            image_format: png, svg, or pdf

        Returns:
            DiagramResult with full status and error recovery info
        """
        # Step 1: Detect diagram type
        diagram_type = self.detect_diagram_type(mermaid_code)

        # Step 2: Generate filename
        base_filename = self.generate_filename(
            markdown_file, diagram_num, diagram_type.value, title
        )

        # Step 3: Save .mmd file
        mmd_path = self.save_mmd_file(mermaid_code, output_dir, base_filename)

        # Step 4: Render image
        success, image_path, error_message = self.render_image(mmd_path, image_format)

        if success:
            return DiagramResult(
                success=True,
                mmd_path=str(mmd_path),
                image_path=str(image_path),
                diagram_type=diagram_type.value,
                error_message=None
            )

        # Step 5: On error, search troubleshooting guide
        matches = []
        suggested_fix = None

        if self.troubleshooting and error_message:
            found_matches = self.troubleshooting.search(error_message, diagram_type)
            matches = [m.to_dict() for m in found_matches]

            if found_matches:
                # Get suggested fix from best match
                best_match = found_matches[0]
                if best_match.correct_example:
                    suggested_fix = best_match.correct_example

        # Step 6: Generate search recommendation if no good matches
        search_rec = None
        if not matches or (matches and matches[0].get('error_number', 0) == 0):
            search_rec = self.get_search_recommendation(error_message, diagram_type)

        return DiagramResult(
            success=False,
            mmd_path=str(mmd_path),
            image_path=None,
            diagram_type=diagram_type.value,
            error_message=error_message,
            troubleshooting_matches=matches,
            suggested_fix=suggested_fix,
            search_recommendation=search_rec
        )


def main():
    parser = argparse.ArgumentParser(
        description='Resilient Mermaid diagram generation with error recovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from code string
  python resilient_diagram.py --code "flowchart TD; A-->B" \\
      --markdown-file design_doc --diagram-num 1 --title "overview"

  # Generate from existing .mmd file
  python resilient_diagram.py --mmd-file diagram.mmd \\
      --markdown-file doc --diagram-num 1 --title "flow"

  # Read from stdin with JSON output
  cat diagram.mmd | python resilient_diagram.py --stdin \\
      --markdown-file doc --diagram-num 2 --title "sequence" --json

  # Full options with custom output directory
  python resilient_diagram.py --code "sequenceDiagram..." \\
      --markdown-file api_design --diagram-num 1 --title "auth_flow" \\
      --format svg --output-dir ./images/ --json

Output Files:
  The script generates both .mmd (source) and image files:
  - ./diagrams/api_design_01_sequence_auth_flow.mmd
  - ./diagrams/api_design_01_sequence_auth_flow.png

Error Recovery:
  On validation failure, the script:
  1. Searches troubleshooting.md for matching errors
  2. Returns suggested fixes from the guide
  3. Generates search queries for external tools:
     - perplexity_ask MCP (primary)
     - brave_web_search MCP (secondary)
     - gemini skill (tertiary)
     - WebSearch tool (fallback)
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--code', '-c', type=str, help='Mermaid code string')
    input_group.add_argument('--mmd-file', '-i', type=Path, help='Path to existing .mmd file')
    input_group.add_argument('--stdin', action='store_true', help='Read Mermaid code from stdin')

    # Output options
    parser.add_argument('--output-dir', '-o', type=Path, default=Path('docs/architecture'),
                        help='Output directory (default: docs/architecture)')
    parser.add_argument('--markdown-file', '-m', type=str, default='diagram',
                        help='Source markdown filename for naming convention')
    parser.add_argument('--diagram-num', '-n', type=int, default=1,
                        help='Diagram number (default: 1)')
    parser.add_argument('--title', '-t', type=str, default='diagram',
                        help='Diagram title for filename')
    parser.add_argument('--format', '-f', choices=['png', 'svg', 'pdf'], default='png',
                        help='Image format (default: png)')

    # Output format
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output result as JSON (recommended for programmatic use)')

    # Troubleshooting guide override
    parser.add_argument('--troubleshooting', type=Path,
                        help='Path to troubleshooting.md (auto-detected if not specified)')

    args = parser.parse_args()

    # Get mermaid code from input source
    if args.code:
        mermaid_code = args.code
    elif args.mmd_file:
        if not args.mmd_file.exists():
            print(f"ERROR: File not found: {args.mmd_file}", file=sys.stderr)
            sys.exit(1)
        mermaid_code = args.mmd_file.read_text(encoding='utf-8')
    elif args.stdin:
        mermaid_code = sys.stdin.read()

    if not mermaid_code.strip():
        print("ERROR: No Mermaid code provided", file=sys.stderr)
        sys.exit(1)

    # Initialize generator
    generator = ResilientDiagramGenerator(troubleshooting_path=args.troubleshooting)

    # Generate diagram
    result = generator.generate(
        mermaid_code=mermaid_code,
        markdown_file=args.markdown_file,
        diagram_num=args.diagram_num,
        title=args.title,
        output_dir=args.output_dir,
        image_format=args.format
    )

    # Output result
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        if result.success:
            print(f"SUCCESS: Generated diagram")
            print(f"  MMD file:   {result.mmd_path}")
            print(f"  Image file: {result.image_path}")
            print(f"  Type:       {result.diagram_type}")
        else:
            print(f"FAILED: {result.error_message}", file=sys.stderr)
            print(f"  MMD file: {result.mmd_path}")
            print(f"  Type:     {result.diagram_type}")

            if result.troubleshooting_matches:
                print(f"\nTroubleshooting matches found ({len(result.troubleshooting_matches)}):")
                for match in result.troubleshooting_matches[:3]:
                    print(f"  - Error {match['error_number']}: {match['title']} ({match['severity']})")

                if result.suggested_fix:
                    print(f"\nSuggested fix (from Error {result.troubleshooting_matches[0]['error_number']}):")
                    print("```mermaid")
                    print(result.suggested_fix)
                    print("```")

            if result.search_recommendation:
                print(f"\nNo exact match found. Search recommendation:")
                print(f"  Query: {result.search_recommendation}")
                print(f"\n  Search tools (in priority order):")
                print(f"    1. perplexity_ask MCP")
                print(f"    2. brave_web_search MCP")
                print(f"    3. gemini skill")
                print(f"    4. WebSearch tool")

    sys.exit(0 if result.success else 1)


if __name__ == '__main__':
    main()
