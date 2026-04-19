#!/usr/bin/env python3
"""
cpp_bridge.py - ctypes wrapper for the C++ Neural Memory library
Provides Pythonic interface to libneural_memory.so
"""

import ctypes
import ctypes.util
from pathlib import Path
from typing import Optional

# ============================================================================
# Find library
# ============================================================================

def _find_lib() -> str:
    candidates = [
        Path(__file__).parent.parent / "build" / "libneural_memory.so",
        Path.home() / "projects" / "neural-memory-adapter" / "build" / "libneural_memory.so",
        Path("/usr/local/lib/libneural_memory.so"),
        Path("/usr/lib/libneural_memory.so"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    # Try LD_LIBRARY_PATH
    lib = ctypes.util.find_library("neural_memory")
    if lib:
        return lib
    raise FileNotFoundError(
        "libneural_memory.so not found. Build first:\n"
        "  cd ~/projects/neural-memory-adapter/build && cmake --build . -j$(nproc)"
    )

# ============================================================================
# C API types (matching c_api.h)
# ============================================================================

class CSearchResult(ctypes.Structure):
    """Must match NeuralMemoryResult in c_api.h exactly."""
    _fields_ = [
        ("id", ctypes.c_uint64),
        ("embedding", ctypes.POINTER(ctypes.c_float)),  # float* — caller must NOT free
        ("embedding_dim", ctypes.c_int),
        ("label", ctypes.c_char * 256),
        ("content", ctypes.c_char * 4096),
        ("similarity", ctypes.c_float),
        ("salience", ctypes.c_float),
    ]

class CStats(ctypes.Structure):
    _fields_ = [
        ("total_stores", ctypes.c_uint64),
        ("total_retrieves", ctypes.c_uint64),
        ("total_searches", ctypes.c_uint64),
        ("total_consolidations", ctypes.c_uint64),
        ("avg_store_us", ctypes.c_uint64),
        ("avg_retrieve_us", ctypes.c_uint64),
        ("graph_nodes", ctypes.c_size_t),
        ("graph_edges", ctypes.c_size_t),
        ("hopfield_patterns", ctypes.c_size_t),
        ("hopfield_occupancy", ctypes.c_float),
    ]

# ============================================================================
# C++ Bridge
# ============================================================================

class NeuralMemoryCpp:
    """
    Pythonic wrapper for the C++ Neural Memory library.
    
    Usage:
        mem = NeuralMemoryCpp()
        mem.initialize(dim=384)
        id = mem.store([0.1, 0.2, ...], "label", "content")
        results = mem.retrieve([0.1, 0.2, ...], k=10)
        mem.consolidate()
        mem.shutdown()
    """
    
    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = _find_lib()
        
        self._lib = ctypes.CDLL(lib_path)
        self._setup_functions()
        self._handle = None
    
    def _setup_functions(self):
        lib = self._lib
        
        # void* neural_memory_create_dim(int dim)
        lib.neural_memory_create_dim.argtypes = [ctypes.c_int]
        lib.neural_memory_create_dim.restype = ctypes.c_void_p
        
        # void* neural_memory_create(void)
        lib.neural_memory_create.argtypes = []
        lib.neural_memory_create.restype = ctypes.c_void_p
        
        # void neural_memory_destroy(void* handle)
        lib.neural_memory_destroy.argtypes = [ctypes.c_void_p]
        lib.neural_memory_destroy.restype = None
        
        # uint64_t neural_memory_store(void* handle, const float* vec, int dim,
        #                               const char* label, const char* content)
        lib.neural_memory_store.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_char_p, ctypes.c_char_p
        ]
        lib.neural_memory_store.restype = ctypes.c_uint64
        
        # int neural_memory_retrieve_full(void* handle, const float* query, int dim,
        #                                  int k, NeuralMemoryResult* results)
        lib.neural_memory_retrieve_full.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(CSearchResult)
        ]
        lib.neural_memory_retrieve_full.restype = ctypes.c_int
        
        # int neural_memory_search(void* handle, const char* query, int k,
        #                           NeuralMemoryResult* results)
        lib.neural_memory_search.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
            ctypes.POINTER(CSearchResult)
        ]
        lib.neural_memory_search.restype = ctypes.c_int
        
        # size_t neural_memory_consolidate(void* handle)
        lib.neural_memory_consolidate.argtypes = [ctypes.c_void_p]
        lib.neural_memory_consolidate.restype = ctypes.c_size_t
        
        # void neural_memory_stats(void* handle, NeuralMemoryStats* stats)
        lib.neural_memory_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(CStats)]
        lib.neural_memory_stats.restype = None
        
        # int neural_memory_think(void* handle, uint64_t start_id, int depth,
        #                          uint64_t* ids, float* activations, int max_results)
        lib.neural_memory_think.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]
        lib.neural_memory_think.restype = ctypes.c_int
    
    def initialize(self, dim: int = 384, hopfield_capacity: int = 1024,
                   episodic_capacity: int = 10000) -> bool:
        """Initialize the memory system."""
        self._handle = self._lib.neural_memory_create_dim(dim)
        return self._handle is not None
    
    def shutdown(self):
        """Shutdown and free resources."""
        if self._handle:
            self._lib.neural_memory_destroy(self._handle)
            self._handle = None
    
    def store(self, embedding: list[float], label: str = "", content: str = "") -> int:
        """Store a memory with embedding. Returns memory ID."""
        assert self._handle, "Not initialized. Call initialize() first."
        
        arr = (ctypes.c_float * len(embedding))(*embedding)
        label_bytes = label.encode('utf-8') if label else None
        content_bytes = content.encode('utf-8') if content else None
        
        return self._lib.neural_memory_store(
            self._handle, arr, len(embedding), label_bytes, content_bytes
        )
    
    def retrieve(self, query: list[float], k: int = 10) -> list[dict]:
        """Retrieve memories by embedding similarity."""
        assert self._handle, "Not initialized."
        
        arr = (ctypes.c_float * len(query))(*query)
        results = (CSearchResult * k)()
        
        count = self._lib.neural_memory_retrieve_full(
            self._handle, arr, len(query), k, results
        )
        
        return [
            {
                'id': results[i].id,
                'score': results[i].similarity,
                'similarity': results[i].similarity,
                'salience': results[i].salience,
                'label': results[i].label.decode('utf-8', errors='replace'),
                'content': results[i].content.decode('utf-8', errors='replace'),
            }
            for i in range(count)
        ]

    def search(self, query_text: str, k: int = 10) -> list[dict]:
        """Text-based search (uses internal embedding)."""
        assert self._handle, "Not initialized."
        
        results = (CSearchResult * k)()
        count = self._lib.neural_memory_search(
            self._handle, query_text.encode('utf-8'), k, results
        )
        
        return [
            {
                'id': results[i].id,
                'score': results[i].similarity,
                'similarity': results[i].similarity,
                'label': results[i].label.decode('utf-8', errors='replace'),
                'content': results[i].content.decode('utf-8', errors='replace'),
            }
            for i in range(count)
        ]

    def think(self, start_id: int, depth: int = 3, max_results: int = 20) -> list[dict]:
        """Spreading activation from a memory."""
        assert self._handle, "Not initialized."
        
        ids = (ctypes.c_uint64 * max_results)()
        activations = (ctypes.c_float * max_results)()
        
        count = self._lib.neural_memory_think(
            self._handle, start_id, depth, ids, activations, max_results
        )
        
        return [
            {
                'id': ids[i],
                'activation': activations[i],
                'label': f"node_{ids[i]}",
            }
            for i in range(count)
        ]
    
    def consolidate(self) -> int:
        """Run memory consolidation. Returns count of consolidated memories."""
        assert self._handle, "Not initialized."
        return self._lib.neural_memory_consolidate(self._handle)
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        assert self._handle, "Not initialized."
        
        stats = CStats()
        self._lib.neural_memory_stats(self._handle, ctypes.byref(stats))
        
        return {
            'total_stores': stats.total_stores,
            'total_retrieves': stats.total_retrieves,
            'total_searches': stats.total_searches,
            'total_consolidations': stats.total_consolidations,
            'avg_store_us': stats.avg_store_us,
            'avg_retrieve_us': stats.avg_retrieve_us,
            'graph_nodes': stats.graph_nodes,
            'graph_edges': stats.graph_edges,
            'hopfield_patterns': stats.hopfield_patterns,
            'hopfield_occupancy': stats.hopfield_occupancy,
        }
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, *args):
        self.shutdown()


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    try:
        mem = NeuralMemoryCpp()
        mem.initialize(dim=384)
        print(f"Initialized C++ adapter")
        
        # Store some test data
        import random
        for i in range(5):
            vec = [random.uniform(-1, 1) for _ in range(384)]
            norm = sum(x*x for x in vec) ** 0.5
            vec = [x/norm for x in vec]
            mid = mem.store(vec, f"test_{i}", f"Test memory {i}")
            print(f"  Stored: [{mid}] test_{i}")
        
        # Retrieve
        query = [random.uniform(-1, 1) for _ in range(384)]
        norm = sum(x*x for x in query) ** 0.5
        query = [x/norm for x in query]
        results = mem.retrieve(query, k=3)
        print(f"\n  Retrieved {len(results)} results")
        for r in results:
            print(f"    [{r['id']}] {r['label']}: {r['score']:.3f}")
        
        # Stats
        stats = mem.get_stats()
        print(f"\n  Stats: {stats}")
        
        mem.shutdown()
        print("\nC++ bridge: OK")
    except FileNotFoundError as e:
        print(f"C++ library not found: {e}")
        print("Build first: cd ~/projects/neural-memory-adapter/build && cmake --build . -j$(nproc)")
