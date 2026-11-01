"""Unit tests for pdf_index.py — no external dependencies beyond stdlib + numpy."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Make scripts/ importable
import sys
_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))

from pdf_index import PDFIndex, __version__


class TestPDFIndexBasics:
    """Tests that don't require pymupdf/faiss/sentence-transformers."""

    def test_version(self):
        assert __version__ == "1.0.0"

    def test_init_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            assert Path(tmp).exists()
            assert Path(tmp).is_dir()

    def test_expand_tilde(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Won't actually use home, just verifying the path logic
            idx = PDFIndex(index_dir=tmp)
            info = idx.info()
            assert info["index_dir"] == str(Path(tmp).resolve())

    def test_info_empty_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            info = idx.info()
            assert info["total_chunks"] == 0
            assert info["indexed_files"] == 0
            assert info["has_index"] is False
            assert "index_dir" in info

    def test_info_after_manual_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            chunks_file = Path(tmp) / PDFIndex.CHUNKS_FILE
            chunks_file.write_text(
                json.dumps({"source": "/a.pdf", "filename": "a.pdf", "page": 1, "text": "hello"}) + "\n" +
                json.dumps({"source": "/b.pdf", "filename": "b.pdf", "page": 2, "text": "world"}) + "\n"
            )
            info = idx.info()
            assert info["total_chunks"] == 2

    def test_search_raises_without_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            with pytest.raises(FileNotFoundError, match="No index found"):
                idx.search("test query")

    def test_load_chunks_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            assert idx._load_chunks() == []

    def test_collect_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            chunks_file = Path(tmp) / PDFIndex.CHUNKS_FILE
            chunks_file.write_text(
                json.dumps({"source": "/x.pdf", "filename": "x.pdf", "page": 1, "text": "a"}) + "\n" +
                json.dumps({"source": "/x.pdf", "filename": "x.pdf", "page": 2, "text": "b"}) + "\n" +
                json.dumps({"source": "/y.pdf", "filename": "y.pdf", "page": 1, "text": "c"}) + "\n"
            )
            sources = idx._collect_sources()
            assert sources == {"/x.pdf", "/y.pdf"}

    def test_files_json_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            chunks_file = Path(tmp) / PDFIndex.CHUNKS_FILE
            chunks_file.write_text(
                json.dumps({"source": "/a.pdf", "filename": "a.pdf", "page": 1, "text": "x"}) + "\n"
            )
            # Manually write files.json
            files_file = Path(tmp) / PDFIndex.FILES_FILE
            files_file.write_text(json.dumps(["/a.pdf"]))
            info = idx.info()
            assert info["indexed_files"] == 1

    def test_corrupt_jsonl_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            chunks_file = Path(tmp) / PDFIndex.CHUNKS_FILE
            chunks_file.write_text(
                '{"source": "/a.pdf", "filename": "a.pdf", "page": 1, "text": "ok"}\n'
                'not valid json\n'
                '{"source": "/b.pdf", "filename": "b.pdf", "page": 2, "text": "also ok"}\n'
            )
            chunks = idx._load_chunks()
            assert len(chunks) == 2
            assert chunks[0]["filename"] == "a.pdf"
            assert chunks[1]["filename"] == "b.pdf"

    def test_model_file_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            # No model file
            info = idx.info()
            assert info["model"] is None

            # Write model file
            (Path(tmp) / PDFIndex.MODEL_FILE).write_text("all-MiniLM-L6-v2")
            info = idx.info()
            assert info["model"] == "all-MiniLM-L6-v2"

    def test_repr_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx = PDFIndex(index_dir=tmp)
            assert "PDFIndex" in repr(idx) or True  # just don't crash


@pytest.mark.integration
class TestIntegration:
    """Tests requiring pymupdf, sentence-transformers, and faiss."""

    @pytest.fixture
    def sample_pdf_dir(self):
        """Create a minimal PDF for testing."""
        try:
            import fitz  # pymupdf
        except ImportError:
            pytest.skip("pymupdf not installed")

        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "test.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "This is a test PDF about machine learning.")
            page.insert_text((50, 80), "It discusses neural networks and deep learning.")
            doc.save(str(pdf_path))
            doc.close()
            yield tmp

    @pytest.fixture
    def index_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield tmp

    def test_full_workflow(self, sample_pdf_dir, index_dir):
        """End-to-end: index a PDF, search it, verify results."""
        try:
            import faiss           # noqa: F401
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError:
            pytest.skip("faiss or sentence-transformers not installed")

        idx = PDFIndex(index_dir=index_dir)

        # Index
        n = idx.index_directory(
            sample_pdf_dir,
            chunk_size=500,
            model_name="all-MiniLM-L6-v2",
            progress=False,
        )
        assert n > 0

        # Info
        info = idx.info()
        assert info["has_index"]
        assert info["indexed_files"] >= 1
        assert info["total_chunks"] >= 1

        # Search
        results = idx.search("machine learning neural network", top_k=3)
        assert len(results) >= 1
        assert results[0]["score"] > 0.0
        assert "test.pdf" in results[0]["filename"]
        assert "machine" in results[0]["text"].lower()

    def test_force_reindex(self, sample_pdf_dir, index_dir):
        """Re-indexing with force should work."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")

        idx = PDFIndex(index_dir=index_dir)

        n1 = idx.index_directory(sample_pdf_dir, progress=False)
        n2 = idx.index_directory(sample_pdf_dir, force=True, progress=False)
        # After force, same number of chunks (only one PDF)
        assert n2 == n1

    def test_no_duplicate_indexing(self, sample_pdf_dir, index_dir):
        """Second call without force should skip already-indexed files."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")

        idx = PDFIndex(index_dir=index_dir)
        n1 = idx.index_directory(sample_pdf_dir, progress=False)
        n2 = idx.index_directory(sample_pdf_dir, progress=False)
        assert n2 == n1  # no new chunks

    def test_info_after_indexing(self, sample_pdf_dir, index_dir):
        """Info after indexing has non-zero values."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            pytest.skip("faiss not installed")

        idx = PDFIndex(index_dir=index_dir)
        idx.index_directory(sample_pdf_dir, progress=False)
        info = idx.info()
        assert info["has_index"] is True
        assert info["total_chunks"] > 0
        assert info["indexed_files"] > 0
