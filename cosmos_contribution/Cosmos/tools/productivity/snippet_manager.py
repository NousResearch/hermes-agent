"""
cosmos Snippet Manager - Code Snippet Storage

"Store your best code once, use it forever!"

Features:
- Multi-language support
- Tagging and categorization
- Fuzzy search
- Template variables
- Clipboard integration
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import hashlib
import re

from loguru import logger


@dataclass
class Snippet:
    """A reusable code snippet."""
    id: str
    name: str
    code: str
    language: str = "text"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)  # Template variables
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    usage_count: int = 0
    favorite: bool = False

    def render(self, **kwargs) -> str:
        """Render snippet with variable substitution."""
        result = self.code
        # Merge default variables with provided ones
        vars_to_use = {**self.variables, **kwargs}
        for key, value in vars_to_use.items():
            result = result.replace(f"${{{key}}}", str(value))
            result = result.replace(f"${{{{key}}}}", str(value))  # Double braces
        return result


class SnippetManager:
    """Manage reusable code snippets."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir) / "snippets"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.snippets_file = self.data_dir / "snippets.json"
        self.snippets: Dict[str, Snippet] = {}
        self._load()

    def _load(self):
        """Load snippets from disk."""
        if self.snippets_file.exists():
            try:
                with open(self.snippets_file) as f:
                    data = json.load(f)
                for snippet_data in data.get("snippets", []):
                    snippet = Snippet(**snippet_data)
                    self.snippets[snippet.id] = snippet
            except Exception as e:
                logger.error(f"Failed to load snippets: {e}")

    def _save(self):
        """Save snippets to disk."""
        try:
            with open(self.snippets_file, "w") as f:
                json.dump({
                    "snippets": [asdict(s) for s in self.snippets.values()],
                    "updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save snippets: {e}")

    def add(
        self,
        name: str,
        code: str,
        language: str = "text",
        description: str = "",
        tags: List[str] = None,
        variables: Dict[str, str] = None,
    ) -> Snippet:
        """Add a new snippet."""
        snippet_id = hashlib.sha256(
            f"{name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]

        # Auto-detect template variables
        detected_vars = re.findall(r'\$\{(\w+)\}', code)
        default_vars = {v: f"<{v}>" for v in detected_vars}
        if variables:
            default_vars.update(variables)

        snippet = Snippet(
            id=snippet_id,
            name=name,
            code=code,
            language=language,
            description=description,
            tags=tags or [],
            variables=default_vars,
        )
        self.snippets[snippet_id] = snippet
        self._save()
        logger.info(f"SnippetManager: Added '{name}'")
        return snippet

    def get(self, snippet_id: str) -> Optional[Snippet]:
        """Get a snippet by ID."""
        snippet = self.snippets.get(snippet_id)
        if snippet:
            snippet.usage_count += 1
            self._save()
        return snippet

    def get_by_name(self, name: str) -> Optional[Snippet]:
        """Get a snippet by name (fuzzy match)."""
        name_lower = name.lower()
        for snippet in self.snippets.values():
            if snippet.name.lower() == name_lower:
                snippet.usage_count += 1
                self._save()
                return snippet
        return None

    def search(self, query: str) -> List[Snippet]:
        """Search snippets by name, description, or tags."""
        query_lower = query.lower()
        results = []
        for snippet in self.snippets.values():
            score = 0
            if query_lower in snippet.name.lower():
                score += 3
            if query_lower in snippet.description.lower():
                score += 2
            if any(query_lower in tag.lower() for tag in snippet.tags):
                score += 1
            if query_lower in snippet.code.lower():
                score += 1
            if score > 0:
                results.append((score, snippet))

        return [s for _, s in sorted(results, key=lambda x: -x[0])]

    def list_by_language(self, language: str) -> List[Snippet]:
        """Get all snippets for a specific language."""
        return [s for s in self.snippets.values() if s.language.lower() == language.lower()]

    def list_favorites(self) -> List[Snippet]:
        """Get favorite snippets."""
        return sorted(
            [s for s in self.snippets.values() if s.favorite],
            key=lambda s: -s.usage_count
        )

    def list_popular(self, limit: int = 10) -> List[Snippet]:
        """Get most used snippets."""
        return sorted(
            self.snippets.values(),
            key=lambda s: -s.usage_count
        )[:limit]

    def toggle_favorite(self, snippet_id: str) -> bool:
        """Toggle favorite status."""
        if snippet_id in self.snippets:
            self.snippets[snippet_id].favorite = not self.snippets[snippet_id].favorite
            self._save()
            return True
        return False

    def delete(self, snippet_id: str) -> bool:
        """Delete a snippet."""
        if snippet_id in self.snippets:
            del self.snippets[snippet_id]
            self._save()
            return True
        return False

    def get_languages(self) -> List[str]:
        """Get all unique languages."""
        return sorted(set(s.language for s in self.snippets.values()))

    def export_to_file(self, snippet_id: str, output_path: str) -> bool:
        """Export a snippet to a file."""
        snippet = self.snippets.get(snippet_id)
        if not snippet:
            return False
        try:
            Path(output_path).write_text(snippet.code)
            return True
        except Exception:
            return False

    def import_from_file(
        self,
        file_path: str,
        name: str = None,
        language: str = None,
        tags: List[str] = None,
    ) -> Optional[Snippet]:
        """Import a snippet from a file."""
        try:
            path = Path(file_path)
            code = path.read_text()

            # Auto-detect language from extension
            ext_to_lang = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".rs": "rust",
                ".go": "go",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".sh": "bash",
                ".sql": "sql",
                ".html": "html",
                ".css": "css",
                ".json": "json",
                ".yaml": "yaml",
                ".yml": "yaml",
            }
            detected_lang = ext_to_lang.get(path.suffix.lower(), "text")

            return self.add(
                name=name or path.stem,
                code=code,
                language=language or detected_lang,
                tags=tags or [],
            )
        except Exception as e:
            logger.error(f"Failed to import snippet: {e}")
            return None

    def stats(self) -> Dict[str, Any]:
        """Get snippet statistics."""
        return {
            "total": len(self.snippets),
            "favorites": sum(1 for s in self.snippets.values() if s.favorite),
            "languages": len(self.get_languages()),
            "total_usage": sum(s.usage_count for s in self.snippets.values()),
        }


# Global instance
snippet_manager = SnippetManager()
