"""
Cosmos Codebase Indexer - AST-Based Memory Population

Extracts structured data from every .py file using AST parsing and stores
it across the 7-layer memory system (archival, knowledge graph, virtual
context, episodic). This gives agents deep codebase awareness when planning
or implementing tasks.

Usage:
    from Cosmos.memory.codebase_indexer import get_codebase_indexer
    indexer = get_codebase_indexer()
    stats = await indexer.index_codebase()
"""

import ast
import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from loguru import logger


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FunctionInfo:
    name: str
    docstring: str
    signature: str        # "async def foo(bar: int) -> dict"
    is_async: bool
    is_public: bool       # Not starting with _
    lineno: int


@dataclass
class ClassInfo:
    name: str
    docstring: str
    methods: List[FunctionInfo]
    bases: List[str]
    lineno: int


@dataclass
class ModuleInfo:
    filepath: str         # "cosmos/core/nexus.py"
    module_name: str      # "cosmos.core.nexus"
    docstring: str
    classes: List[ClassInfo]
    functions: List[FunctionInfo]
    imports: List[str]
    internal_imports: List[str]   # Only cosmos.* imports
    line_count: int
    category: str         # "core", "memory", "web", etc.
    tags: List[str]       # ["codebase", "module", "core", ...]


# =============================================================================
# AST EXTRACTOR
# =============================================================================

class ASTExtractor:
    """Extracts structured module information from Python source files via AST."""

    def extract_module(self, filepath: str, project_root: str) -> Optional[ModuleInfo]:
        """Parse one .py file and extract module-level information."""
        try:
            source = Path(filepath).read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"Cannot read {filepath}: {e}")
            return None

        try:
            tree = ast.parse(source, filename=filepath)
        except SyntaxError as e:
            logger.debug(f"Syntax error in {filepath}: {e}")
            return None

        # Relative path from project root
        rel_path = os.path.relpath(filepath, project_root).replace("\\", "/")
        module_name = rel_path.replace("/", ".").replace(".py", "")

        # Module docstring
        docstring = ast.get_docstring(tree) or ""

        classes = []
        functions = []
        imports = []
        internal_imports = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(self._extract_class(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._extract_function(node))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
                    if alias.name.startswith("cosmos"):
                        internal_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    if node.module.startswith("cosmos"):
                        internal_imports.append(node.module)

        line_count = len(source.splitlines())
        category = self._categorize_path(rel_path)
        tags = self._generate_tags(
            category, classes, functions, module_name
        )

        return ModuleInfo(
            filepath=rel_path,
            module_name=module_name,
            docstring=docstring[:500],
            classes=classes,
            functions=functions,
            imports=imports,
            internal_imports=list(set(internal_imports)),
            line_count=line_count,
            category=category,
            tags=tags,
        )

    def _extract_class(self, node: ast.ClassDef) -> ClassInfo:
        """Extract class name, docstring, bases, methods."""
        docstring = ast.get_docstring(node) or ""
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                bases.append("?")

        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(self._extract_function(item))

        return ClassInfo(
            name=node.name,
            docstring=docstring[:300],
            methods=methods,
            bases=bases,
            lineno=node.lineno,
        )

    def _extract_function(self, node) -> FunctionInfo:
        """Extract function name, signature, docstring, async flag."""
        docstring = ast.get_docstring(node) or ""
        is_async = isinstance(node, ast.AsyncFunctionDef)

        # Build signature string
        try:
            # Reconstruct just the function def line
            args_str = ast.unparse(node.args)
            returns_str = ""
            if node.returns:
                returns_str = f" -> {ast.unparse(node.returns)}"
            prefix = "async def" if is_async else "def"
            signature = f"{prefix} {node.name}({args_str}){returns_str}"
        except Exception:
            signature = f"{'async ' if is_async else ''}def {node.name}(...)"

        return FunctionInfo(
            name=node.name,
            docstring=docstring[:200],
            signature=signature,
            is_async=is_async,
            is_public=not node.name.startswith("_"),
            lineno=node.lineno,
        )

    def _categorize_path(self, relative_path: str) -> str:
        """Map first directory after cosmos/ to category."""
        parts = relative_path.replace("\\", "/").split("/")
        # e.g. cosmos/core/nexus.py -> "core"
        if len(parts) >= 2 and parts[0] == "cosmos":
            return parts[1]
        return "other"

    def _generate_tags(
        self,
        category: str,
        classes: List[ClassInfo],
        functions: List[FunctionInfo],
        module_name: str,
    ) -> List[str]:
        """Generate tags for the module."""
        tags = ["codebase", "module", category]
        # Add class names as tags
        for cls in classes:
            tags.append(cls.name)
        # Add public function names as tags
        for fn in functions:
            if fn.is_public:
                tags.append(fn.name)
        # Add the short module name
        short_name = module_name.split(".")[-1]
        if short_name not in tags:
            tags.append(short_name)
        return tags


# =============================================================================
# CODEBASE INDEXER
# =============================================================================

class CodebaseIndexer:
    """
    Indexes the Cosmos codebase into the 7-layer memory system.

    Uses AST parsing to extract structured data from every .py file and
    stores it across archival memory, knowledge graph, virtual context,
    and episodic memory layers.
    """

    def __init__(
        self,
        project_root: str = None,
        scan_dir: str = "cosmos",
        reindex_interval_hours: float = 6.0,
    ):
        if project_root is None:
            # Auto-detect: go up from this file to the project root
            project_root = str(Path(__file__).parent.parent.parent)
        self.project_root = project_root
        self.scan_dir = scan_dir
        self.reindex_interval_hours = reindex_interval_hours

        self._extractor = ASTExtractor()
        self._indexed_modules: Dict[str, str] = {}  # filepath -> SHA256 hash
        self._last_indexed: Optional[datetime] = None
        self._indexing_in_progress = False
        self._background_task: Optional[asyncio.Task] = None
        self._stats: Dict[str, Any] = {}

    async def index_codebase(
        self,
        memory_system=None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Scan all .py files, extract AST data, and store across memory layers.

        Args:
            memory_system: MemorySystem instance (uses global if None)
            force: If True, re-index all files even if unchanged

        Returns:
            Stats dict with indexing results
        """
        if self._indexing_in_progress:
            return {"status": "already_running"}

        self._indexing_in_progress = True
        start_time = time.time()

        try:
            # Get memory system
            if memory_system is None:
                try:
                    from Cosmos.memory.memory_system import get_memory_system
                    memory_system = get_memory_system()
                    if not memory_system._initialized:
                        await memory_system.initialize()
                except Exception as e:
                    logger.warning(f"Memory system unavailable: {e}")
                    memory_system = None

            # Scan .py files
            scan_path = Path(self.project_root) / self.scan_dir
            py_files = self._find_py_files(scan_path)

            modules_indexed = 0
            modules_skipped = 0
            modules_failed = 0
            all_modules: List[ModuleInfo] = []
            entities_created = 0
            relationships_created = 0

            for filepath in py_files:
                # Hash content for change detection
                try:
                    content = Path(filepath).read_bytes()
                    content_hash = hashlib.sha256(content).hexdigest()
                except Exception:
                    modules_failed += 1
                    continue

                rel_path = os.path.relpath(filepath, self.project_root).replace("\\", "/")

                # Skip unchanged files unless forced
                if not force and rel_path in self._indexed_modules:
                    if self._indexed_modules[rel_path] == content_hash:
                        modules_skipped += 1
                        continue

                # Extract module info via AST
                module = self._extractor.extract_module(filepath, self.project_root)
                if module is None:
                    modules_failed += 1
                    continue

                all_modules.append(module)

                # Store to archival memory
                if memory_system:
                    try:
                        await self._store_archival_entry(memory_system, module)
                    except Exception as e:
                        logger.debug(f"Archival store failed for {rel_path}: {e}")

                    # Store to knowledge graph
                    try:
                        e_count, r_count = await self._store_graph_data(
                            memory_system, module
                        )
                        entities_created += e_count
                        relationships_created += r_count
                    except Exception as e:
                        logger.debug(f"Graph store failed for {rel_path}: {e}")

                # Track indexed hash
                self._indexed_modules[rel_path] = content_hash
                modules_indexed += 1

            # Store virtual context tier map
            if memory_system and all_modules:
                try:
                    await self._store_virtual_context(memory_system, all_modules)
                except Exception as e:
                    logger.debug(f"Virtual context store failed: {e}")

            # Record episodic memory event
            if memory_system:
                try:
                    await self._record_indexing_event(memory_system, {
                        "modules_indexed": modules_indexed,
                        "modules_skipped": modules_skipped,
                        "modules_failed": modules_failed,
                        "entities_created": entities_created,
                        "relationships_created": relationships_created,
                    })
                except Exception as e:
                    logger.debug(f"Episodic record failed: {e}")

            elapsed = time.time() - start_time
            self._last_indexed = datetime.now()

            self._stats = {
                "status": "completed",
                "modules_indexed": modules_indexed,
                "modules_skipped": modules_skipped,
                "modules_failed": modules_failed,
                "total_files": len(py_files),
                "entities_created": entities_created,
                "relationships_created": relationships_created,
                "elapsed_seconds": round(elapsed, 2),
                "last_indexed": self._last_indexed.isoformat(),
            }

            logger.info(
                f"Codebase indexing complete: {modules_indexed} modules, "
                f"{entities_created} entities, {relationships_created} relationships "
                f"({elapsed:.1f}s)"
            )

            return self._stats

        except Exception as e:
            logger.error(f"Codebase indexing failed: {e}")
            return {"status": "error", "error": str(e)}

        finally:
            self._indexing_in_progress = False

    def _find_py_files(self, scan_path: Path) -> List[str]:
        """Find all .py files under scan_path, excluding __pycache__."""
        py_files = []
        for root, dirs, files in os.walk(str(scan_path)):
            # Exclude __pycache__ and hidden dirs
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for f in files:
                if f.endswith(".py") and not f.endswith(".pyc"):
                    py_files.append(os.path.join(root, f))
        return sorted(py_files)

    async def _store_archival_entry(
        self, memory_system, module: ModuleInfo
    ):
        """Store one archival entry per module with structured content."""
        # Build content string (capped at 3000 chars)
        parts = [
            f"MODULE: {module.module_name}",
            f"FILE: {module.filepath} ({module.line_count} lines)",
            f"CATEGORY: {module.category}",
        ]

        if module.docstring:
            parts.append(f"\nDESCRIPTION: {module.docstring[:300]}")

        if module.classes:
            parts.append("\nCLASSES:")
            for cls in module.classes:
                bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
                desc = f": {cls.docstring[:80]}" if cls.docstring else ""
                parts.append(f"- {cls.name}{bases_str}{desc}")
                # List public methods
                public_methods = [m for m in cls.methods if m.is_public]
                for m in public_methods[:8]:
                    parts.append(f"  Methods: {m.signature[:100]}")

        public_fns = [f for f in module.functions if f.is_public]
        if public_fns:
            parts.append("\nPUBLIC FUNCTIONS:")
            for fn in public_fns[:10]:
                parts.append(f"- {fn.signature[:120]}")

        if module.internal_imports:
            parts.append(
                f"\nINTERNAL DEPS: {', '.join(module.internal_imports[:10])}"
            )

        content = "\n".join(parts)
        # Cap at 3000 chars for large modules
        if len(content) > 3000:
            content = content[:2950] + "\n... (truncated)"

        await memory_system.remember(
            content=content,
            tags=module.tags,
            importance=0.6,
            metadata={
                "type": "codebase_module",
                "category": module.category,
                "filepath": module.filepath,
                "line_count": module.line_count,
            },
            extract_entities=False,  # We handle graph storage ourselves
        )

    async def _store_graph_data(
        self, memory_system, module: ModuleInfo
    ) -> tuple:
        """Store entities and relationships in the knowledge graph."""
        entities_created = 0
        relationships_created = 0
        graph = memory_system.knowledge_graph

        # File entity
        try:
            docstring_preview = module.docstring[:100] if module.docstring else ""
            await graph.add_entity(
                name=module.filepath,
                entity_type="file",
                properties={
                    "category": module.category,
                    "line_count": module.line_count,
                    "docstring_preview": docstring_preview,
                },
            )
            entities_created += 1
        except Exception as e:
            logger.debug(f"File entity failed for {module.filepath}: {e}")

        # Class and public function entities
        for cls in module.classes:
            try:
                await graph.add_entity(
                    name=f"{module.module_name}.{cls.name}",
                    entity_type="code",
                    properties={
                        "kind": "class",
                        "signature": cls.name,
                        "module_path": module.filepath,
                        "bases": cls.bases,
                    },
                )
                entities_created += 1

                # part_of relationship: class -> module
                try:
                    await graph.add_relationship(
                        f"{module.module_name}.{cls.name}",
                        module.filepath,
                        "part_of",
                    )
                    relationships_created += 1
                except Exception:
                    pass

                # is_a relationships for inheritance
                for base in cls.bases:
                    if base and base not in ("object", "Exception", "Enum"):
                        try:
                            await graph.add_relationship(
                                f"{module.module_name}.{cls.name}",
                                base,
                                "is_a",
                            )
                            relationships_created += 1
                        except Exception:
                            pass

            except Exception as e:
                logger.debug(f"Class entity failed for {cls.name}: {e}")

        for fn in module.functions:
            if fn.is_public:
                try:
                    await graph.add_entity(
                        name=f"{module.module_name}.{fn.name}",
                        entity_type="code",
                        properties={
                            "kind": "function",
                            "signature": fn.signature[:150],
                            "module_path": module.filepath,
                        },
                    )
                    entities_created += 1

                    # part_of relationship
                    try:
                        await graph.add_relationship(
                            f"{module.module_name}.{fn.name}",
                            module.filepath,
                            "part_of",
                        )
                        relationships_created += 1
                    except Exception:
                        pass

                except Exception as e:
                    logger.debug(f"Function entity failed for {fn.name}: {e}")

        # depends_on relationships from imports
        for imp in module.internal_imports:
            # Convert import to file path
            imp_path = imp.replace(".", "/") + ".py"
            try:
                await graph.add_relationship(
                    module.filepath, imp_path, "depends_on"
                )
                relationships_created += 1
            except Exception:
                pass

        return entities_created, relationships_created

    async def _store_virtual_context(
        self, memory_system, modules: List[ModuleInfo]
    ):
        """Store tiered directory listings in virtual context."""
        from Cosmos.memory.virtual_context import MemoryBlock, MemoryTier

        # Group modules by category
        by_category: Dict[str, List[ModuleInfo]] = {}
        for m in modules:
            by_category.setdefault(m.category, []).append(m)

        # Define tier mapping
        hot_categories = {"core", "memory"}
        warm_categories = {"integration", "web", "agents", "trading"}
        # Everything else is cold

        tier_configs = [
            ("HOT", hot_categories, 0.9),
            ("WARM", warm_categories, 0.6),
            ("COLD", None, 0.3),  # None = everything else
        ]

        for tier_name, cat_set, importance in tier_configs:
            lines = [f"CODEBASE {tier_name} TIER:"]
            categories = (
                [c for c in by_category if c in cat_set]
                if cat_set
                else [c for c in by_category if c not in hot_categories and c not in warm_categories]
            )

            for cat in sorted(categories):
                mods = by_category[cat]
                lines.append(f"\n  {cat}/ ({len(mods)} modules):")
                for m in sorted(mods, key=lambda x: x.filepath)[:20]:
                    desc = m.docstring.split("\n")[0][:60] if m.docstring else ""
                    short = m.filepath.split("/")[-1]
                    lines.append(f"    {short} - {desc}")

            content = "\n".join(lines)
            if len(content) > 100:
                block = MemoryBlock(
                    id=f"codebase_{tier_name.lower()}",
                    content=content[:2000],
                    importance_score=importance,
                    tags=["codebase", f"tier_{tier_name.lower()}"],
                )
                memory_system.virtual_context.context_window.add_block(block)

    async def _record_indexing_event(
        self, memory_system, stats: Dict
    ):
        """Record an episodic memory event for this indexing run."""
        await memory_system.remember(
            content=(
                f"Codebase indexing completed at {datetime.now().isoformat()}. "
                f"Indexed {stats['modules_indexed']} modules, "
                f"created {stats['entities_created']} entities and "
                f"{stats['relationships_created']} relationships. "
                f"Skipped {stats['modules_skipped']} unchanged, "
                f"{stats['modules_failed']} failed."
            ),
            tags=["codebase", "indexing_event", "system"],
            importance=0.4,
            metadata={"type": "codebase_indexing_event", **stats},
            extract_entities=False,
        )

    async def start_background_indexing(self):
        """Run index_codebase() immediately, then every reindex_interval_hours."""
        if self._background_task and not self._background_task.done():
            logger.debug("Background indexing already running")
            return

        self._background_task = asyncio.create_task(self._background_loop())

    async def _background_loop(self):
        """Background loop: index immediately, then periodically."""
        # Initial index
        try:
            await self.index_codebase()
        except Exception as e:
            logger.error(f"Initial codebase indexing failed: {e}")

        # Periodic re-index
        interval = self.reindex_interval_hours * 3600
        while True:
            try:
                await asyncio.sleep(interval)
                await self.index_codebase()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic codebase indexing failed: {e}")
                await asyncio.sleep(300)  # Retry in 5 min on error

    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics."""
        return {
            "indexed_modules": len(self._indexed_modules),
            "last_indexed": self._last_indexed.isoformat() if self._last_indexed else None,
            "indexing_in_progress": self._indexing_in_progress,
            "reindex_interval_hours": self.reindex_interval_hours,
            "project_root": self.project_root,
            **self._stats,
        }


# =============================================================================
# GLOBAL SINGLETON
# =============================================================================

_codebase_indexer: Optional[CodebaseIndexer] = None


def get_codebase_indexer() -> CodebaseIndexer:
    """Get or create the global CodebaseIndexer instance."""
    global _codebase_indexer
    if _codebase_indexer is None:
        _codebase_indexer = CodebaseIndexer()
    return _codebase_indexer


async def index_codebase_into_memory() -> Dict:
    """Convenience: index codebase using global memory system."""
    indexer = get_codebase_indexer()
    return await indexer.index_codebase()
