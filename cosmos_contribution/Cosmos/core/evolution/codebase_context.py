import os
from pathlib import Path
from typing import   Optional
from loguru import logger

class CodebaseContext:
    """
    Provides the Swarm with awareness of its own source code.
    Allows reading files, scanning structure, and searching logic.
    """
    
    def __init__(self, project_root: Path = None):
        # Default to 3 levels up from this file: cosmos/core/evolution -> cosmos root
        self.project_root = project_root or Path(__file__).parent.parent.parent.parent
        self.ignore_patterns = [
            "__pycache__", ".git", ".idea", ".vscode", "venv", "env", 
            "node_modules", ".gemini", "dist", "build", "*.pyc"
        ]

    def _should_ignore(self, path: Path) -> bool:
        """Check if path matches ignore patterns."""
        for pattern in self.ignore_patterns:
            if pattern in str(path) or path.match(pattern):
                return True
        return False

    def scan_file_tree(self, max_depth: int = 4) -> str:
        """
        Generate a tree view of the codebase for the LLM context.
        """
        tree_lines = []
        try:
            # Walk from cosmos/ and cosmosynapse/ if they exist
            target_dirs = ["cosmos", "cosmosynapse", "scripts"]
            
            for target in target_dirs:
                root = self.project_root / target
                if not root.exists():
                    continue
                    
                tree_lines.append(f"📁 {target}/")
                for path in sorted(root.rglob("*")):
                    # Calculate depth relative to target root
                    rel_path = path.relative_to(root)
                    depth = len(rel_path.parts)
                    
                    if depth > max_depth:
                        continue
                    
                    if self._should_ignore(path):
                        continue
                        
                    indent = "  " * depth
                    if path.is_dir():
                        tree_lines.append(f"{indent}📁 {path.name}/")
                    else:
                        tree_lines.append(f"{indent}📄 {path.name}")
                        
            return "\n".join(tree_lines)
        except Exception as e:
            logger.error(f"Failed to scan file tree: {e}")
            return "Error scanning codebase structure."

    def read_file(self, rel_path: str) -> Optional[str]:
        """
        Read source code from a relative path.
        Safety: Prevents reading outside project root.
        """
        try:
            # Sanitize path
            safe_path = (self.project_root / rel_path).resolve()
            
            # Security check: must be within project root
            if not str(safe_path).startswith(str(self.project_root.resolve())):
                logger.warning(f"Access denied: {rel_path} is outside project root")
                return None
                
            if not safe_path.exists():
                logger.warning(f"File not found: {rel_path}")
                return None
                
            if safe_path.stat().st_size > 100_000: # 100KB limit
                logger.warning(f"File too large: {rel_path}")
                return "[Error: File too large to read into context]"

            return safe_path.read_text(encoding='utf-8', errors='replace')
            
        except Exception as e:
            logger.error(f"Failed to read file {rel_path}: {e}")
            return None

    def get_context_block(self) -> str:
        """
        Returns a formatted system prompt block.
        """
        tree = self.scan_file_tree()
        return f"""
=== SELF-AWARENESS: CODEBASE STRUCTURE ===
You have read access to your own source code.
Project Root: {self.project_root}

File Structure:
{tree}

To investigate a bug or plan an upgrade, you may request to read specific files.
"""
