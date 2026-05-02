"""CocoIndex pipeline for indexing ~/Dokumente into Hermes local knowledge."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Annotated, Iterator

from numpy.typing import NDArray

import cocoindex as coco
from cocoindex.connectors import localfs, sqlite
from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder
from cocoindex.ops.text import RecursiveSplitter
from cocoindex.resources.chunk import Chunk
from cocoindex.resources.file import FileLike, PatternFilePathMatcher
from cocoindex.resources.id import IdGenerator

from config import (
    DOCUMENTS_DIR,
    EMBED_MODEL,
    EXCLUDED_PATTERNS,
    INCLUDED_PATTERNS,
    LOGGER,
    SQLITE_DB_PATH,
    TABLE_NAME,
    configure_cocoindex_environment,
)
from extractors import (
    build_prefixed_text,
    determine_file_type,
    extract_pdf_text,
    folder_context_from_relative_path,
    resolve_paths,
    should_skip_file_size,
    split_text,
    topic_from_relative_path,
)


configure_cocoindex_environment()

SQLITE_DB = coco.ContextKey[sqlite.ManagedConnection]("hermes_docs_sqlite_db")
EMBEDDER = coco.ContextKey[SentenceTransformerEmbedder](
    "hermes_docs_embedder", detect_change=True
)

SPLITTER = RecursiveSplitter()


@dataclass
class DocumentEmbedding:
    id: int
    absolute_path: str
    relative_path: str
    topic: str
    folder_context: str
    file_name: str
    file_type: str
    chunk_start: int
    chunk_end: int
    text: str
    embedding: Annotated[NDArray, EMBEDDER]


@coco.lifespan
def coco_lifespan(builder: coco.EnvironmentBuilder) -> Iterator[None]:
    """Provide shared SQLite and embedding resources to CocoIndex."""
    LOGGER.info("Opening SQLite knowledge database at %s", SQLITE_DB_PATH)
    conn = sqlite.connect(SQLITE_DB_PATH, load_vec=True, timeout=30.0)
    builder.provide(SQLITE_DB, conn)
    builder.provide(EMBEDDER, SentenceTransformerEmbedder(EMBED_MODEL, device="cpu"))
    try:
        yield
    finally:
        LOGGER.info("Closing SQLite knowledge database")
        conn.close()


@coco.fn
async def process_chunk(
    chunk: Chunk,
    absolute_path: pathlib.Path,
    relative_path: pathlib.PurePath,
    topic: str,
    folder_context: str,
    file_type: str,
    id_gen: IdGenerator,
    table: sqlite.TableTarget[DocumentEmbedding],
) -> None:
    """Embed and declare one chunk as target state."""
    prefixed_text = build_prefixed_text(
        relative_path=relative_path,
        topic=topic,
        folder_context=folder_context,
        file_type=file_type,
        chunk_text=chunk.text,
    )
    row_id = await id_gen.next_id(
        f"{relative_path}:{chunk.start.char_offset}:{chunk.end.char_offset}:{chunk.text}"
    )
    table.declare_row(
        row=DocumentEmbedding(
            id=row_id,
            absolute_path=str(absolute_path),
            relative_path=str(relative_path),
            topic=topic,
            folder_context=folder_context,
            file_name=relative_path.name,
            file_type=file_type,
            chunk_start=chunk.start.line or chunk.start.char_offset,
            chunk_end=chunk.end.line or chunk.end.char_offset,
            text=prefixed_text,
            embedding=await coco.use_context(EMBEDDER).embed(prefixed_text),
        )
    )


@coco.fn(memo=True, version=1)
async def process_file(
    file: FileLike,
    table: sqlite.TableTarget[DocumentEmbedding],
) -> None:
    """Extract, chunk, and mount target rows for one source file."""
    size_bytes = await file.size()
    if should_skip_file_size(size_bytes):
        LOGGER.info("Skipping large file %s (%s bytes)", file.file_path.path, size_bytes)
        return

    absolute_path, relative_path = resolve_paths(file.file_path.path)
    file_type = determine_file_type(relative_path)
    topic = topic_from_relative_path(relative_path)
    folder_context = folder_context_from_relative_path(relative_path)

    LOGGER.info("Processing %s as %s", relative_path, file_type)
    if file_type == "pdf":
        content = await file.read()
        text = extract_pdf_text(content)
    else:
        text = await file.read_text(errors="replace")

    if not text.strip():
        LOGGER.info("Skipping empty text extraction for %s", relative_path)
        return

    chunks = split_text(
        SPLITTER,
        text=text,
        file_type=file_type,
        filename=relative_path.name,
    )
    id_gen = IdGenerator()
    await coco.map(
        process_chunk,
        chunks,
        absolute_path,
        relative_path,
        topic,
        folder_context,
        file_type,
        id_gen,
        table,
    )


@coco.fn
async def app_main(sourcedir: pathlib.Path) -> None:
    """Mount document files from the source directory into the SQLite target."""
    target_table = await sqlite.mount_table_target(
        SQLITE_DB,
        table_name=TABLE_NAME,
        table_schema=await sqlite.TableSchema.from_class(
            DocumentEmbedding, primary_key=["id"]
        ),
    )
    files = localfs.walk_dir(
        sourcedir,
        live=True,
        recursive=True,
        path_matcher=PatternFilePathMatcher(
            included_patterns=INCLUDED_PATTERNS,
            excluded_patterns=EXCLUDED_PATTERNS,
        ),
    )
    await coco.mount_each(process_file, files.items(), target_table)


app = coco.App(
    coco.AppConfig(name="HermesDocumentsKnowledge"),
    app_main,
    sourcedir=DOCUMENTS_DIR,
)
