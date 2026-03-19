"""
cosmos File System Agent - Intelligent File Operations

Novel Approaches:
1. Project Structure Understanding - Semantic code navigation
2. Smart File Search - Natural language to file paths
3. Context-Aware Operations - Understand project conventions
4. Safe Modifications - Preview and validate changes
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any, Callable
import json
import hashlib

from loguru import logger


class FileType(Enum):
    """Types of files."""
    CODE = "code"
    CONFIG = "config"
    DATA = "data"
    DOCUMENTATION = "documentation"
    TEST = "test"
    ASSET = "asset"
    UNKNOWN = "unknown"


class OperationType(Enum):
    """Types of file operations."""
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    RENAME = "rename"
    MOVE = "move"
    COPY = "copy"
    SEARCH = "search"
    ANALYZE = "analyze"


@dataclass
class FileInfo:
    """Information about a file."""
    path: str
    name: str
    extension: str
    file_type: FileType
    size_bytes: int
    modified_at: datetime
    created_at: Optional[datetime] = None

    # Content info
    line_count: int = 0
    encoding: str = "utf-8"
    is_binary: bool = False

    # For code files
    language: str = ""
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)

    # Metadata
    git_status: str = ""  # "modified", "staged", "untracked", etc.
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "name": self.name,
            "extension": self.extension,
            "file_type": self.file_type.value,
            "size_bytes": self.size_bytes,
            "language": self.language,
            "line_count": self.line_count,
        }


@dataclass
class ProjectStructure:
    """Understanding of a project's structure."""
    root_path: str
    name: str
    project_type: str = ""  # "python", "node", "rust", etc.

    # Key directories
    source_dirs: list[str] = field(default_factory=list)
    test_dirs: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    documentation_files: list[str] = field(default_factory=list)

    # Statistics
    total_files: int = 0
    total_lines: int = 0
    file_by_type: dict = field(default_factory=dict)
    language_stats: dict = field(default_factory=dict)

    # Dependencies
    dependencies: list[str] = field(default_factory=list)
    dev_dependencies: list[str] = field(default_factory=list)

    # Convention patterns
    naming_convention: str = ""  # "snake_case", "camelCase", etc.
    import_style: str = ""

    def to_dict(self) -> dict:
        return {
            "root": self.root_path,
            "name": self.name,
            "type": self.project_type,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "languages": self.language_stats,
        }


@dataclass
class FileChange:
    """A proposed file change."""
    operation: OperationType
    path: str
    new_path: Optional[str] = None  # For rename/move
    content: Optional[str] = None   # For write/create
    diff: Optional[str] = None      # Preview of changes
    reason: str = ""

    # Validation
    is_valid: bool = True
    validation_errors: list[str] = field(default_factory=list)

    # Backup
    backup_path: Optional[str] = None
    original_content: Optional[str] = None


@dataclass
class SearchResult:
    """Result of a file search."""
    file_path: str
    matches: list[dict] = field(default_factory=list)  # {"line": N, "content": "...", "context": "..."}
    relevance_score: float = 0.0


class FileSystemAgent:
    """
    Intelligent file system operations agent.

    Features:
    - Natural language file search
    - Project structure understanding
    - Safe file modifications with preview
    - Code-aware operations
    """

    def __init__(
        self,
        llm_fn: Optional[Callable] = None,
        working_dir: Optional[str] = None,
        create_backups: bool = True,
    ):
        self.llm_fn = llm_fn
        self.working_dir = Path(working_dir or os.getcwd())
        self.create_backups = create_backups

        self._project_cache: dict[str, ProjectStructure] = {}
        self._file_cache: dict[str, FileInfo] = {}

        # Extension to language mapping
        self._ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".rs": "rust",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".sql": "sql",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "zsh",
            ".ps1": "powershell",
        }

        # Extension to file type mapping
        self._ext_to_type = {
            **{ext: FileType.CODE for ext in self._ext_to_lang.keys()},
            ".json": FileType.CONFIG,
            ".yaml": FileType.CONFIG,
            ".yml": FileType.CONFIG,
            ".toml": FileType.CONFIG,
            ".ini": FileType.CONFIG,
            ".env": FileType.CONFIG,
            ".md": FileType.DOCUMENTATION,
            ".rst": FileType.DOCUMENTATION,
            ".txt": FileType.DOCUMENTATION,
            ".csv": FileType.DATA,
            ".tsv": FileType.DATA,
            ".xml": FileType.DATA,
            ".png": FileType.ASSET,
            ".jpg": FileType.ASSET,
            ".gif": FileType.ASSET,
            ".svg": FileType.ASSET,
            ".ico": FileType.ASSET,
        }

        self._lock = asyncio.Lock()

    async def analyze_project(
        self,
        root_path: Optional[str] = None,
        max_files: int = 1000,
    ) -> ProjectStructure:
        """
        Analyze a project's structure.

        Returns comprehensive project information.
        """
        root = Path(root_path or self.working_dir)

        if str(root) in self._project_cache:
            return self._project_cache[str(root)]

        structure = ProjectStructure(
            root_path=str(root),
            name=root.name,
        )

        # Detect project type
        structure.project_type = self._detect_project_type(root)

        # Scan files
        language_lines = {}
        file_count = 0

        for path in root.rglob("*"):
            if file_count >= max_files:
                break

            if not path.is_file():
                continue

            # Skip common ignored patterns
            if self._should_ignore(path):
                continue

            file_count += 1

            # Classify file
            ext = path.suffix.lower()
            file_type = self._ext_to_type.get(ext, FileType.UNKNOWN)
            language = self._ext_to_lang.get(ext, "")

            structure.file_by_type[file_type.value] = \
                structure.file_by_type.get(file_type.value, 0) + 1

            # Count lines for code files
            if file_type == FileType.CODE:
                try:
                    content = path.read_text(encoding='utf-8', errors='ignore')
                    lines = content.count('\n') + 1
                    structure.total_lines += lines

                    if language:
                        language_lines[language] = \
                            language_lines.get(language, 0) + lines

                except (OSError, UnicodeDecodeError) as e:
                    # Skip files that can't be read (binary, permissions, etc.)
                    logger.debug(f"Could not read file {path}: {e}")

            # Track special directories/files
            rel_path = str(path.relative_to(root))

            if "test" in rel_path.lower():
                if str(path.parent) not in structure.test_dirs:
                    structure.test_dirs.append(str(path.parent.relative_to(root)))
            elif file_type == FileType.CODE:
                parent = str(path.parent.relative_to(root))
                if parent not in structure.source_dirs and parent != ".":
                    structure.source_dirs.append(parent)

            if file_type == FileType.CONFIG:
                structure.config_files.append(rel_path)

            if file_type == FileType.DOCUMENTATION:
                structure.documentation_files.append(rel_path)

        structure.total_files = file_count
        structure.language_stats = language_lines

        # Detect naming convention
        structure.naming_convention = self._detect_naming_convention(root)

        # Parse dependencies
        structure.dependencies, structure.dev_dependencies = \
            self._parse_dependencies(root, structure.project_type)

        self._project_cache[str(root)] = structure
        return structure

    def _detect_project_type(self, root: Path) -> str:
        """Detect project type from marker files."""
        markers = {
            "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile"],
            "node": ["package.json", "yarn.lock", "package-lock.json"],
            "rust": ["Cargo.toml"],
            "go": ["go.mod", "go.sum"],
            "java": ["pom.xml", "build.gradle"],
            "ruby": ["Gemfile", "Rakefile"],
            "dotnet": ["*.csproj", "*.sln"],
        }

        for project_type, files in markers.items():
            for pattern in files:
                if list(root.glob(pattern)):
                    return project_type

        return "unknown"

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored."""
        ignore_patterns = [
            "__pycache__", ".git", "node_modules", "venv", ".venv",
            "env", ".env", "dist", "build", ".pytest_cache",
            ".mypy_cache", ".tox", "eggs", "*.egg-info",
            ".idea", ".vscode", ".vs", "*.pyc", "*.pyo",
        ]

        path_str = str(path)
        for pattern in ignore_patterns:
            if pattern in path_str:
                return True

        return False

    def _detect_naming_convention(self, root: Path) -> str:
        """Detect naming convention from files."""
        snake_count = 0
        camel_count = 0
        kebab_count = 0

        for path in root.rglob("*.py"):
            if self._should_ignore(path):
                continue

            name = path.stem
            if "_" in name:
                snake_count += 1
            if name != name.lower() and name[0].islower():
                camel_count += 1

        for path in root.rglob("*.js"):
            if self._should_ignore(path):
                continue

            name = path.stem
            if "-" in name:
                kebab_count += 1
            if name != name.lower() and name[0].islower():
                camel_count += 1

        if snake_count > camel_count and snake_count > kebab_count:
            return "snake_case"
        if camel_count > snake_count and camel_count > kebab_count:
            return "camelCase"
        if kebab_count > 0:
            return "kebab-case"

        return "unknown"

    def _parse_dependencies(
        self,
        root: Path,
        project_type: str,
    ) -> tuple[list[str], list[str]]:
        """Parse project dependencies."""
        deps = []
        dev_deps = []

        try:
            if project_type == "python":
                # Try pyproject.toml
                pyproject = root / "pyproject.toml"
                if pyproject.exists():
                    import tomllib
                    data = tomllib.loads(pyproject.read_text())
                    deps = list(data.get("project", {}).get("dependencies", []))
                    dev_deps = list(data.get("project", {}).get("optional-dependencies", {}).get("dev", []))

                # Try requirements.txt
                req_txt = root / "requirements.txt"
                if req_txt.exists():
                    lines = req_txt.read_text().splitlines()
                    deps.extend([
                        l.split("==")[0].split(">=")[0].strip()
                        for l in lines if l.strip() and not l.startswith("#")
                    ])

            elif project_type == "node":
                pkg_json = root / "package.json"
                if pkg_json.exists():
                    data = json.loads(pkg_json.read_text())
                    deps = list(data.get("dependencies", {}).keys())
                    dev_deps = list(data.get("devDependencies", {}).keys())

        except Exception as e:
            logger.debug(f"Dependency parsing error: {e}")

        return deps, dev_deps

    async def search(
        self,
        query: str,
        search_content: bool = True,
        file_types: Optional[list[FileType]] = None,
        max_results: int = 20,
    ) -> list[SearchResult]:
        """
        Search for files using natural language or patterns.

        Args:
            query: Search query (natural language or pattern)
            search_content: Whether to search file contents
            file_types: Filter by file types
            max_results: Maximum results to return

        Returns:
            List of search results with matches
        """
        results = []

        # Parse query
        if self.llm_fn:
            search_params = await self._parse_search_query(query)
        else:
            search_params = self._basic_query_parse(query)

        patterns = search_params.get("patterns", [query])
        keywords = search_params.get("keywords", [])

        # Search files
        for path in self.working_dir.rglob("*"):
            if not path.is_file():
                continue

            if self._should_ignore(path):
                continue

            ext = path.suffix.lower()
            file_type = self._ext_to_type.get(ext, FileType.UNKNOWN)

            if file_types and file_type not in file_types:
                continue

            # Check filename
            name_match = any(
                p.lower() in path.name.lower()
                for p in patterns
            )

            content_matches = []

            # Check content
            if search_content and file_type != FileType.ASSET:
                try:
                    content = path.read_text(encoding='utf-8', errors='ignore')

                    for keyword in keywords:
                        for i, line in enumerate(content.splitlines(), 1):
                            if keyword.lower() in line.lower():
                                content_matches.append({
                                    "line": i,
                                    "content": line.strip()[:200],
                                    "keyword": keyword,
                                })
                except (OSError, UnicodeDecodeError) as e:
                    # Skip unreadable files during search
                    logger.debug(f"Could not search file {path}: {e}")

            if name_match or content_matches:
                relevance = 1.0 if name_match else 0.5
                relevance += min(len(content_matches) * 0.1, 0.5)

                results.append(SearchResult(
                    file_path=str(path.relative_to(self.working_dir)),
                    matches=content_matches[:10],
                    relevance_score=relevance,
                ))

            if len(results) >= max_results:
                break

        # Sort by relevance
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        return results

    async def _parse_search_query(self, query: str) -> dict:
        """Use LLM to parse natural language search query."""
        prompt = f"""Parse this file search query into structured parameters.

Query: {query}

Return JSON:
{{
  "patterns": ["filename patterns to match"],
  "keywords": ["content keywords to search"],
  "file_types": ["code", "config", etc. or empty for all],
  "directory": "specific directory or empty"
}}"""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            return json.loads(self._extract_json(response))

        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            return {"patterns": [query], "keywords": [query]}

    def _basic_query_parse(self, query: str) -> dict:
        """Basic query parsing without LLM."""
        words = query.lower().split()
        return {
            "patterns": words,
            "keywords": [w for w in words if len(w) > 2],
        }

    async def get_file_info(self, path: str) -> Optional[FileInfo]:
        """Get detailed information about a file."""
        file_path = self.working_dir / path

        if not file_path.exists():
            return None

        stat = file_path.stat()
        ext = file_path.suffix.lower()

        info = FileInfo(
            path=str(file_path.relative_to(self.working_dir)),
            name=file_path.name,
            extension=ext,
            file_type=self._ext_to_type.get(ext, FileType.UNKNOWN),
            size_bytes=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            language=self._ext_to_lang.get(ext, ""),
        )

        # Read content for analysis
        if info.file_type != FileType.ASSET and stat.st_size < 1_000_000:
            try:
                content = file_path.read_text(encoding='utf-8')
                info.line_count = content.count('\n') + 1
                info.is_binary = False

                # Extract code structure
                if info.language == "python":
                    info.classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                    info.functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
                    info.imports = re.findall(r'^(?:from\s+\S+\s+)?import\s+(\S+)', content, re.MULTILINE)

                elif info.language in ("javascript", "typescript"):
                    info.classes = re.findall(r'class\s+(\w+)', content)
                    info.functions = re.findall(r'function\s+(\w+)', content)
                    info.exports = re.findall(r'export\s+(?:default\s+)?(?:class|function|const|let|var)\s+(\w+)', content)

            except UnicodeDecodeError:
                info.is_binary = True

        return info

    async def read_file(
        self,
        path: str,
        encoding: str = "utf-8",
    ) -> Optional[str]:
        """Read file contents."""
        file_path = self.working_dir / path

        try:
            return file_path.read_text(encoding=encoding)
        except Exception as e:
            logger.error(f"Read failed: {e}")
            return None

    async def write_file(
        self,
        path: str,
        content: str,
        create_backup: Optional[bool] = None,
        validate: bool = True,
    ) -> FileChange:
        """
        Write content to a file with optional validation.

        Returns FileChange with status and diff.
        """
        file_path = self.working_dir / path
        do_backup = create_backup if create_backup is not None else self.create_backups

        change = FileChange(
            operation=OperationType.WRITE if file_path.exists() else OperationType.CREATE,
            path=path,
            content=content,
        )

        # Read original for diff
        if file_path.exists():
            try:
                change.original_content = file_path.read_text(encoding='utf-8')
                change.diff = self._generate_diff(change.original_content, content)
            except (OSError, UnicodeDecodeError) as e:
                logger.debug(f"Could not read original file for diff: {e}")
                change.original_content = None
                change.diff = None

        # Validate if requested
        if validate:
            errors = await self._validate_content(path, content)
            if errors:
                change.is_valid = False
                change.validation_errors = errors
                return change

        # Create backup
        if do_backup and file_path.exists():
            backup_path = file_path.with_suffix(file_path.suffix + ".bak")
            try:
                import shutil
                shutil.copy2(file_path, backup_path)
                change.backup_path = str(backup_path)
            except Exception as e:
                logger.warning(f"Backup failed: {e}")

        # Write file
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            change.is_valid = True
        except Exception as e:
            change.is_valid = False
            change.validation_errors.append(str(e))

        return change

    async def _validate_content(
        self,
        path: str,
        content: str,
    ) -> list[str]:
        """Validate file content."""
        errors = []
        ext = Path(path).suffix.lower()

        # JSON validation
        if ext == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON: {e}")

        # Python syntax validation
        elif ext == ".py":
            try:
                compile(content, path, 'exec')
            except SyntaxError as e:
                errors.append(f"Python syntax error: {e}")

        # YAML validation
        elif ext in (".yaml", ".yml"):
            try:
                import yaml
                yaml.safe_load(content)
            except yaml.YAMLError as e:
                errors.append(f"Invalid YAML: {e}")

        return errors

    def _generate_diff(self, original: str, new: str) -> str:
        """Generate a simple diff."""
        import difflib

        original_lines = original.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            new_lines,
            fromfile='original',
            tofile='new',
            lineterm='',
        )

        return ''.join(diff)

    async def find_and_replace(
        self,
        pattern: str,
        replacement: str,
        file_pattern: str = "*",
        dry_run: bool = True,
    ) -> list[FileChange]:
        """
        Find and replace across files.

        Args:
            pattern: Regex pattern to find
            replacement: Replacement string
            file_pattern: Glob pattern for files to search
            dry_run: If True, only preview changes

        Returns:
            List of changes (applied or previewed)
        """
        changes = []
        regex = re.compile(pattern)

        for path in self.working_dir.rglob(file_pattern):
            if not path.is_file() or self._should_ignore(path):
                continue

            try:
                content = path.read_text(encoding='utf-8')

                if not regex.search(content):
                    continue

                new_content = regex.sub(replacement, content)

                change = FileChange(
                    operation=OperationType.WRITE,
                    path=str(path.relative_to(self.working_dir)),
                    content=new_content,
                    original_content=content,
                    diff=self._generate_diff(content, new_content),
                    reason=f"Replace '{pattern}' with '{replacement}'",
                )

                if not dry_run:
                    path.write_text(new_content, encoding='utf-8')

                changes.append(change)

            except Exception as e:
                logger.debug(f"Error processing {path}: {e}")

        return changes

    async def move_file(
        self,
        source: str,
        destination: str,
        update_imports: bool = True,
    ) -> FileChange:
        """
        Move a file with optional import updates.
        """
        source_path = self.working_dir / source
        dest_path = self.working_dir / destination

        change = FileChange(
            operation=OperationType.MOVE,
            path=source,
            new_path=destination,
        )

        if not source_path.exists():
            change.is_valid = False
            change.validation_errors.append(f"Source not found: {source}")
            return change

        try:
            # Create destination directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Read content for backup
            if source_path.is_file():
                change.original_content = source_path.read_text(encoding='utf-8')

            # Move file
            import shutil
            shutil.move(str(source_path), str(dest_path))

            # Update imports if requested
            if update_imports:
                await self._update_imports(source, destination)

            change.is_valid = True

        except Exception as e:
            change.is_valid = False
            change.validation_errors.append(str(e))

        return change

    async def _update_imports(self, old_path: str, new_path: str):
        """Update imports after file move."""
        # Convert paths to module names
        old_module = old_path.replace("/", ".").replace("\\", ".").replace(".py", "")
        new_module = new_path.replace("/", ".").replace("\\", ".").replace(".py", "")

        # Find and replace imports
        await self.find_and_replace(
            pattern=rf'\b{re.escape(old_module)}\b',
            replacement=new_module,
            file_pattern="*.py",
            dry_run=False,
        )

    async def suggest_refactoring(
        self,
        path: str,
    ) -> list[dict]:
        """
        Suggest refactoring for a file.

        Uses LLM to analyze code and suggest improvements.
        """
        content = await self.read_file(path)
        if not content:
            return []

        if not self.llm_fn:
            return []

        prompt = f"""Analyze this code and suggest refactoring improvements.

File: {path}
```
{content[:4000]}
```

Return JSON array of suggestions:
[
  {{
    "type": "extract_function/rename/simplify/etc",
    "location": "line number or description",
    "current": "current code snippet",
    "suggested": "improved code snippet",
    "reason": "why this improves the code"
  }}
]"""

        try:
            if asyncio.iscoroutinefunction(self.llm_fn):
                response = await self.llm_fn(prompt)
            else:
                response = self.llm_fn(prompt)

            return json.loads(self._extract_json(response))

        except Exception as e:
            logger.error(f"Refactoring analysis failed: {e}")
            return []

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text."""
        start = text.find('[')
        end = text.rfind(']') + 1
        if start >= 0 and end > start:
            return text[start:end]

        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return text[start:end]

        return '[]'

    def get_stats(self) -> dict:
        """Get file system agent statistics."""
        return {
            "working_dir": str(self.working_dir),
            "cached_projects": len(self._project_cache),
            "cached_files": len(self._file_cache),
        }
