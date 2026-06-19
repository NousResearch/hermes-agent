import pytest
from pathlib import Path

MODEL_PATH = Path("~/.aingram/models/nomic-embed-text-v1.5/onnx/model.onnx").expanduser()


class TestEmbedder:
    def test_import_without_onnx(self):
        from plugins.memory.holographic.embedder import get_embedder, cosine_similarity
        assert get_embedder is not None
        assert cosine_similarity is not None

    def test_cosine_similarity_identical(self):
        from plugins.memory.holographic.embedder import cosine_similarity
        import numpy as np
        vec = np.random.rand(768).astype(np.float32)
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        from plugins.memory.holographic.embedder import cosine_similarity
        import numpy as np
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = cosine_similarity(vec1, vec2)
        assert abs(sim) < 1e-6

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="ONNX model not available")
    def test_embed_returns_vector(self):
        from plugins.memory.holographic.embedder import get_embedder
        embedder = get_embedder()
        if embedder is None:
            pytest.skip("ONNX model not available")
        vec = embedder.embed("hello world")
        assert vec is not None
        assert vec.shape == (768,)
        assert vec.dtype.name == "float32"

    @pytest.mark.skipif(not MODEL_PATH.exists(), reason="ONNX model not available")
    def test_embed_empty_string(self):
        from plugins.memory.holographic.embedder import get_embedder
        embedder = get_embedder()
        if embedder is None:
            pytest.skip("ONNX model not available")
        vec = embedder.embed("")
        assert vec is None
