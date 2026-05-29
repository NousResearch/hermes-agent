import pytest

from provider_gateway.guardrails import PIISanitizer


def test_pii_sanitizer_basic() -> None:
    """Test basic sanitization and restoration of various PII types."""
    sanitizer = PIISanitizer()
    prompt = "Halo, silakan hubungi saya di email void@example.com atau IP server 192.168.1.100. Kunci API saya sk-1234567890abcdef1234567890abcdef."
    
    # 1. Sanitize
    sanitized = sanitizer.sanitize_prompt(prompt)
    assert "void@example.com" not in sanitized
    assert "192.168.1.100" not in sanitized
    assert "[REDACTED_EMAIL_1]" in sanitized
    assert "[REDACTED_IP_ADDRESS_2]" in sanitized

    # 2. Restore
    response = "Hasil analisis untuk [REDACTED_EMAIL_1] dan server [REDACTED_IP_ADDRESS_2]."
    restored = sanitizer.restore_response(response)
    assert "void@example.com" in restored
    assert "192.168.1.100" in restored
    assert "[REDACTED_EMAIL_1]" not in restored


def test_pii_sanitizer_code_block_bypass() -> None:
    """Test that markdown code blocks are bypassed during sanitization."""
    sanitizer = PIISanitizer()
    prompt = (
        "Email saya void@example.com\n"
        "```python\n"
        "# Jangan sensor email di dalam kode ini:\n"
        "test_email = 'keep_this@example.com'\n"
        "```\n"
        "Tapi sensor email ini: other@example.com"
    )

    sanitized = sanitizer.sanitize_prompt(prompt)
    assert "void@example.com" not in sanitized
    assert "other@example.com" not in sanitized
    assert "keep_this@example.com" in sanitized  # preserved in code block


def test_streaming_deanonimizer_sliding_buffer() -> None:
    """Test real-time de-anonymization of sliding buffer chunks."""
    sanitizer = PIISanitizer()
    prompt = "Tolong analisis email void@example.com"
    sanitized = sanitizer.sanitize_prompt(prompt)
    placeholder = "[REDACTED_EMAIL_1]"
    assert placeholder in sanitized

    deanonimizer = sanitizer.get_deanonimizer()
    
    # Simulate stream chunks coming in
    chunk1 = "Hasil untuk "
    out1 = deanonimizer.process_chunk(chunk1)
    assert out1 == "Hasil untuk "

    # Chunk splitting the placeholder: "[REDACTED_EM"
    chunk2 = "[REDACTED_EM"
    out2 = deanonimizer.process_chunk(chunk2)
    assert out2 == ""  # should hold back the open bracket part!

    # Next chunk completing the placeholder: "AIL_1] aman."
    chunk3 = "AIL_1] aman."
    out3 = deanonimizer.process_chunk(chunk3)
    assert out3 == "void@example.com aman."

    # Final flush
    out4 = deanonimizer.flush()
    assert out4 == ""
