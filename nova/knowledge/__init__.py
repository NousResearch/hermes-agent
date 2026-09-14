"""NOVA Knowledge — declared corpora, indexed once, searched under policy.

The capability is four separable pieces, in the order a document travels through them:

``sources``   what a corpus is: which files, from where, labelled how.
``chunk``     splitting a document into retrievable pieces without losing where they came from.
``index``     an SQLite FTS5 store of those chunks — NOVA's own database, never the runtime's.
``ingest``    the pipeline that walks a source, extracts text, chunks it and writes the index.

Retrieval itself lives with the runtime, in ``nova.runtime.<adapter>``: the agent reaches a
corpus through a tool, and what a tool is differs per runtime. What does not differ is the
index format, which is why everything above this line is runtime-agnostic.

Two rules hold throughout and are worth stating once:

**Scope is enforced at the query, not by the model.** An agent's readable sources are
compiled into the tool's own configuration and applied as a SQL filter. A model that asks
for a corpus it was not granted does not get a refusal it might argue with — it gets a
search over the corpora it does have.

**Retrieved text is untrusted.** It enters as a tool result with citations, never as part of
the system prompt, and it is scanned for injection before a model sees it. A customer
document is exactly as hostile an input as a scraped web page.
"""

from nova.knowledge.chunk import Chunk, chunk_document
from nova.knowledge.index import KnowledgeIndex, SearchHit
from nova.knowledge.ingest import IngestReport, SourceReport, ingest
from nova.knowledge.sources import (
    KnowledgeCatalog,
    KnowledgeSource,
    iter_documents,
    load_catalog,
)

__all__ = [
    "Chunk",
    "IngestReport",
    "KnowledgeCatalog",
    "KnowledgeIndex",
    "KnowledgeSource",
    "SearchHit",
    "SourceReport",
    "chunk_document",
    "ingest",
    "iter_documents",
    "load_catalog",
]
