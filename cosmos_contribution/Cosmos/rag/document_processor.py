"""
cosmos Document Processor - Ingestion Pipeline

Handles:
- Text extraction from various formats
- Smart chunking strategies
- Metadata extraction
- Embedding generation
"""

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from loguru import logger

from Cosmos.rag.embeddings import EmbeddingManager
from Cosmos.rag.hybrid_retriever import Document


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    chunk_size: int = 512  # Target chunk size in tokens
    chunk_overlap: int = 50  # Overlap between chunks
    min_chunk_size: int = 100  # Minimum chunk size
    split_on: list[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    preserve_sentences: bool = True


@dataclass
class ProcessedDocument:
    """A processed document with chunks."""
    source_id: str
    source_path: Optional[str]
    chunks: list[Document]
    metadata: dict = field(default_factory=dict)
    processed_at: datetime = field(default_factory=datetime.now)


class DocumentProcessor:
    """
    Document processing pipeline for RAG.

    Features:
    - Multiple format support (txt, md, pdf, code)
    - Smart semantic chunking
    - Overlap handling
    - Metadata extraction
    """

    def __init__(
        self,
        embedding_manager: Optional[EmbeddingManager] = None,
        chunking_config: Optional[ChunkingConfig] = None,
    ):
        self.embedding_manager = embedding_manager
        self.config = chunking_config or ChunkingConfig()

        # Format handlers
        self._handlers = {
            ".txt": self._process_text,
            ".md": self._process_markdown,
            ".py": self._process_code,
            ".js": self._process_code,
            ".ts": self._process_code,
            ".json": self._process_json,
            ".yaml": self._process_yaml,
            ".yml": self._process_yaml,
        }

    async def process_file(
        self,
        file_path: str,
        metadata: Optional[dict] = None,
    ) -> ProcessedDocument:
        """Process a file and return chunks."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get handler
        suffix = path.suffix.lower()
        handler = self._handlers.get(suffix, self._process_text)

        # Read and process
        content = path.read_text(encoding='utf-8', errors='ignore')
        processed_content, extracted_metadata = handler(content, path)

        # Merge metadata
        final_metadata = {
            "source_path": str(path),
            "filename": path.name,
            "extension": suffix,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            **(extracted_metadata or {}),
            **(metadata or {}),
        }

        # Chunk the content
        chunks = self._chunk_text(processed_content, str(path), final_metadata)

        # Generate embeddings
        if self.embedding_manager:
            chunks = await self._add_embeddings(chunks)

        return ProcessedDocument(
            source_id=hashlib.md5(str(path).encode()).hexdigest()[:12],
            source_path=str(path),
            chunks=chunks,
            metadata=final_metadata,
        )

    async def process_text(
        self,
        text: str,
        source_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> ProcessedDocument:
        """Process raw text content."""
        source_id = source_id or hashlib.md5(text[:100].encode()).hexdigest()[:12]

        chunks = self._chunk_text(text, source_id, metadata or {})

        if self.embedding_manager:
            chunks = await self._add_embeddings(chunks)

        return ProcessedDocument(
            source_id=source_id,
            source_path=None,
            chunks=chunks,
            metadata=metadata or {},
        )

    def _chunk_text(
        self,
        text: str,
        source_id: str,
        base_metadata: dict,
    ) -> list[Document]:
        """Chunk text into overlapping segments."""
        if not text.strip():
            return []

        # Estimate tokens (rough: 4 chars per token)
        char_chunk_size = self.config.chunk_size * 4
        char_overlap = self.config.chunk_overlap * 4
        char_min_size = self.config.min_chunk_size * 4

        chunks = []
        current_pos = 0
        chunk_idx = 0

        while current_pos < len(text):
            # Find chunk end
            chunk_end = min(current_pos + char_chunk_size, len(text))

            # Try to split at a natural boundary
            if chunk_end < len(text):
                chunk_end = self._find_split_point(
                    text, current_pos, chunk_end
                )

            chunk_text = text[current_pos:chunk_end].strip()

            if len(chunk_text) >= char_min_size:
                doc = Document(
                    id=f"{source_id}_chunk_{chunk_idx}",
                    content=chunk_text,
                    metadata={
                        **base_metadata,
                        "chunk_index": chunk_idx,
                        "char_start": current_pos,
                        "char_end": chunk_end,
                    },
                )
                chunks.append(doc)
                chunk_idx += 1

            # Move position with overlap
            current_pos = chunk_end - char_overlap
            if current_pos <= 0 or current_pos >= len(text) - char_min_size:
                current_pos = chunk_end

        return chunks

    def _find_split_point(
        self,
        text: str,
        start: int,
        end: int,
    ) -> int:
        """Find a good split point in text."""
        search_text = text[start:end]

        # Try split points in order of preference
        for delimiter in self.config.split_on:
            last_pos = search_text.rfind(delimiter)
            if last_pos > len(search_text) * 0.3:  # At least 30% through
                return start + last_pos + len(delimiter)

        return end

    async def _add_embeddings(self, chunks: list[Document]) -> list[Document]:
        """Add embeddings to chunks."""
        texts = [c.content for c in chunks]
        results = await self.embedding_manager.embed_batch(texts)

        for chunk, result in zip(chunks, results):
            chunk.embedding = result.embedding

        return chunks

    def _process_text(self, content: str, path: Path) -> tuple[str, dict]:
        """Process plain text."""
        return content, {}

    def _process_markdown(self, content: str, path: Path) -> tuple[str, dict]:
        """Process markdown with structure extraction."""
        metadata = {}

        # Extract title (first H1)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1)

        # Extract headers for structure
        headers = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
        metadata["headers"] = [h[1] for h in headers]

        # Remove code blocks for cleaner chunking (preserve as separate chunks)
        code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', content, re.DOTALL)
        metadata["has_code"] = len(code_blocks) > 0

        return content, metadata

    def _process_code(self, content: str, path: Path) -> tuple[str, dict]:
        """Process code files with structure extraction."""
        metadata = {
            "language": path.suffix[1:],
        }

        # Extract function/class definitions
        if path.suffix == ".py":
            functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
            metadata["functions"] = functions
            metadata["classes"] = classes
        elif path.suffix in (".js", ".ts"):
            functions = re.findall(r'(?:function|const|let|var)\s+(\w+)\s*[=\(]', content)
            classes = re.findall(r'class\s+(\w+)', content)
            metadata["functions"] = functions
            metadata["classes"] = classes

        # Add docstrings/comments as context
        docstrings = re.findall(r'"""(.*?)"""', content, re.DOTALL)
        metadata["has_docstrings"] = len(docstrings) > 0

        return content, metadata

    def _process_json(self, content: str, path: Path) -> tuple[str, dict]:
        """Process JSON files."""
        import json
        metadata = {"format": "json"}

        try:
            data = json.loads(content)
            if isinstance(data, dict):
                metadata["keys"] = list(data.keys())[:10]
            elif isinstance(data, list):
                metadata["item_count"] = len(data)
        except json.JSONDecodeError:
            pass

        return content, metadata

    def _process_yaml(self, content: str, path: Path) -> tuple[str, dict]:
        """Process YAML files."""
        import yaml
        metadata = {"format": "yaml"}

        try:
            data = yaml.safe_load(content)
            if isinstance(data, dict):
                metadata["keys"] = list(data.keys())[:10]
        except yaml.YAMLError:
            pass

        return content, metadata

    async def process_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        patterns: Optional[list[str]] = None,
    ) -> list[ProcessedDocument]:
        """Process all files in a directory."""
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        patterns = patterns or ["*.txt", "*.md", "*.py", "*.js"]
        results = []

        for pattern in patterns:
            if recursive:
                files = path.rglob(pattern)
            else:
                files = path.glob(pattern)

            for file_path in files:
                if file_path.is_file():
                    try:
                        doc = await self.process_file(str(file_path))
                        results.append(doc)
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")

        return results
