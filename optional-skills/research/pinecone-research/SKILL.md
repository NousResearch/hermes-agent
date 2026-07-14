---
name: pinecone-research
description: Agent RAG and long-term memory with Pinecone
version: 1.0.0
author: immuhammadfurqan
license: MIT
dependencies: [pinecone-client, langchain-pinecone]
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [RAG, Pinecone, Memory, Research, Vector Database, Agent, Retrieval]

---

# Pinecone Research — Agent RAG & Long-Term Memory

Use Pinecone as a retrieval-augmented generation (RAG) backend for agent
conversations: persist embeddings, retrieve relevant context from past
sessions, and build long-term memory.

## When to use this skill

**Use when:**
- Building agent RAG pipelines with Pinecone as the vector store
- Need persistent long-term memory across agent sessions
- Combining retrieval with agent tool use
- Researching or prototyping semantic search workflows

**Use the mlops/pinecone skill instead when:**
- Need a general Pinecone reference (index management, CRUD, hybrid search)
- Working on production infrastructure without agent integration

## Quick start

### Setup

```bash
pip install pinecone-client langchain-pinecone langchain-openai
```

Set your API key:
```bash
export PINECONE_API_KEY="your-api-key"
```

### Basic RAG pipeline

```python
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Initialize Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create or connect to index
index_name = "agent-memory"
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Build vector store
vectorstore = PineconeVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    index_name=index_name,
)

# Retrieve relevant context
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
results = retriever.invoke("What did the agent discuss yesterday?")
```

### Namespace-based session memory

```python
# Store per-session memory
vectorstore = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=OpenAIEmbeddings(),
    namespace=f"session-{session_id}",
)

# Query across all sessions (no namespace filter)
all_memory = PineconeVectorStore(
    index=pc.Index(index_name),
    embedding=OpenAIEmbeddings(),
)
results = all_memory.similarity_search("relevant query", k=10)
```

## Best practices

1. **Namespace by session or user** — isolate data for multi-tenant agents
2. **Batch upserts** — 100–200 vectors per batch for efficiency
3. **Metadata filtering** — tag vectors with session ID, timestamp, topic
4. **Prune old memory** — delete stale namespaces to control costs
5. **Use serverless** — auto-scaling, pay-per-use pricing

## Resources

- **Pinecone Docs**: https://docs.pinecone.io
- **LangChain Integration**: https://python.langchain.com/docs/integrations/vectorstores/pinecone
- **Free Tier**: 1 index, 100K vectors (1536 dimensions)
