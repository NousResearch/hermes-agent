import faiss
import numpy as np
import networkx as nx
import pickle
import os
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from utils.config import Config

class RecursiveMemory:
    """
    Hyper-Recursive Memory System mimicking MemGPT/Letta OS-paging.
    Manages Core Memory (Context) and Archival Memory (Vector Store + Graph).
    """
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL):
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.graph = nx.Graph()
        self.core_memory_limit = 2000  # Token-ish limit for active context (simplified)
        self.short_term_memory: List[Dict[str, Any]] = []
        
        # FAISS Setup
        self.index = faiss.IndexFlatL2(self.dimension)
        self.doc_store: Dict[int, str] = {} # Map ID to content
        
        # Load state
        self.load_state()

    def add_memory(self, content: str, role: str = "user", importance: float = 1.0):
        """Add new interaction to memory. Triggers self-organization."""
        entry = {
            "content": content,
            "role": role,
            "importance": importance,
            "embedding": self.embedding_model.encode(content).tolist()
        }
        
        self.short_term_memory.append(entry)
        self._check_paging()
        self._update_graph(entry)

    def _check_paging(self):
        """Move old/low-importance memories from RAM to Disk (FAISS)."""
        # Simple heuristic: if too many items, move oldest/least important
        if len(self.short_term_memory) > 10: # Arbitrary small number for demo
            # Sort by importance (ascending) and age (oldest first)
            to_archive = self.short_term_memory.pop(0) 
            self._archive(to_archive)

    def _archive(self, entry: Dict[str, Any]):
        """Write to FAISS and Graph."""
        vector = np.array([entry['embedding']]).astype('float32')
        self.index.add(vector)
        doc_id = self.index.ntotal - 1
        self.doc_store[doc_id] = json.dumps(entry)
        
        # Graph connections
        self.graph.add_node(doc_id, content=entry['content'][:50], importance=entry['importance'])
        # Find nearest neighbors to link
        D, I = self.index.search(vector, 3)
        for idx in I[0]:
            if idx != -1 and idx != doc_id:
                self.graph.add_edge(doc_id, int(idx), weight=float(1.0))

    def _update_graph(self, entry: Dict[str, Any]):
        """RL-driven graph evolution (Simulated)."""
        # FUTURE: Add DEAP/RL logic here to prune weak edges or merge nodes
        pass

    def retrieve(self, query: str, k: int = 5) -> str:
        """Hybrid retrieval: semantic search + graph traversal."""
        query_vec = self.embedding_model.encode([query]).astype('float32')
        D, I = self.index.search(query_vec, k)
        
        results = []
        for idx in I[0]:
            if idx != -1:
                data = json.loads(self.doc_store[idx])
                results.append(f"[{data['role']}]: {data['content']}")
                
                # Retrieve one hop neighbors
                neighbors = list(self.graph.neighbors(int(idx)))
                for n_idx in neighbors[:2]: # Limit expansion
                    n_data = json.loads(self.doc_store[n_idx])
                    results.append(f"Related: {n_data['content']}")
        
        return "\n".join(list(set(results))) # Deduplicate

    def get_core_context(self) -> str:
        """Return formatted short-term memory."""
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.short_term_memory])

    def save_state(self):
        faiss.write_index(self.index, os.path.join(Config.VECTOR_STORE_PATH, "index.faiss"))
        with open(os.path.join(Config.VECTOR_STORE_PATH, "docs.pkl"), 'wb') as f:
            pickle.dump(self.doc_store, f)
        nx.write_gpickle(self.graph, os.path.join(Config.VECTOR_STORE_PATH, "memory_graph.gpickle"))

    def load_state(self):
        Config.ensure_dirs()
        try:
            self.index = faiss.read_index(os.path.join(Config.VECTOR_STORE_PATH, "index.faiss"))
            with open(os.path.join(Config.VECTOR_STORE_PATH, "docs.pkl"), 'rb') as f:
                self.doc_store = pickle.load(f)
            self.graph = nx.read_gpickle(os.path.join(Config.VECTOR_STORE_PATH, "memory_graph.gpickle"))
        except:
            # New instance
            pass
