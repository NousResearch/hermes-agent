import pytest
from unittest.mock import AsyncMock
from agent.semantic_chunker import (
    split_sentences, 
    cosine_similarity, 
    savitzky_golay, 
    semantic_chunk_text
)

def test_split_sentences():
    text = "Hello world! This is a test. Is it working? Yes, it is."
    s = split_sentences(text)
    assert len(s) == 4
    assert s[0] == "Hello world!"
    assert s[1] == "This is a test."
    assert s[2] == "Is it working?"
    assert s[3] == "Yes, it is."

def test_cosine_similarity():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    # Orthogonal
    assert abs(cosine_similarity(v1, v2)) < 0.001
    
    # Identical
    assert abs(cosine_similarity([1, 1], [1, 1]) - 1.0) < 0.001
    
    # Opposite
    assert abs(cosine_similarity([1, 1], [-1, -1]) + 1.0) < 0.001

def test_savitzky_golay_math():
    # Provide a simple known signal
    data = [0.0, 1.0, 2.0, 3.0, 4.0]
    # The first derivative of a line y=x should be a constant 1.0
    deriv = savitzky_golay(data, window_size=5, poly_order=2, deriv_order=1)
    
    # The boundaries will be somewhat skewed by the filter, but the center points should be close to 1.
    assert abs(deriv[2] - 1.0) < 0.1

@pytest.mark.asyncio
async def test_semantic_chunk_text():
    text = "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5."
    
    # Mock an embedder that creates distinct orthogonal bounds between sentence 2 and 3
    async def mock_embed(sentences):
        # First two sentences are topic A
        # Next three are topic B
        embeds = []
        for i, s in enumerate(sentences):
            if i < 2:
                embeds.append([1.0, 0.0, 0.0])
            else:
                embeds.append([0.0, 1.0, 0.0])
        return embeds
        
    chunks = await semantic_chunk_text(text, embed_fn=mock_embed)
    
    assert len(chunks) >= 2
    assert "Sentence 1" in chunks[0]
    assert "Sentence 3" in chunks[1]
