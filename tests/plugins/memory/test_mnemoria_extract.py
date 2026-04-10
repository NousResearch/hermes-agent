from plugins.memory.mnemoria.extract import ExtractedFact, extract_from_text


class TestExtractFromTextErrors:
    def test_extracts_error_from_traceback(self):
        text = "Traceback (most recent call last):\n  File test_auth.py\nAssertionError: 3 tests failed"
        facts = extract_from_text(text, source="tool_result")
        assert any("?[error]" in f.content for f in facts)

    def test_extracts_url_near_error(self):
        text = "Error: connection refused\nFailed to reach https://api.example.com:3005\nRetrying..."
        facts = extract_from_text(text, source="tool_result")
        assert any("V[url]" in f.content and "api.example.com" in f.content for f in facts)

    def test_extracts_file_path_near_error(self):
        text = "FAILED tests/test_auth.py::test_jwt - AssertionError\nError in /src/auth/jwt.py line 42"
        facts = extract_from_text(text, source="tool_result")
        assert any("V[file]" in f.content and "/src/auth/jwt.py" in f.content for f in facts)

    def test_ignores_urls_without_error_context(self):
        text = "Documentation available at https://docs.example.com\nAll tests passed."
        facts = extract_from_text(text, source="tool_result")
        url_facts = [f for f in facts if "V[url]" in f.content]
        assert len(url_facts) == 0


class TestExtractFromTextUserStatements:
    def test_extracts_always_use_directive(self):
        text = "always use TypeScript for new code"
        facts = extract_from_text(text, source="user_statement")
        assert any("C[user.pref]" in f.content and "TypeScript" in f.content for f in facts)

    def test_extracts_never_use_directive(self):
        text = "never use var in JavaScript"
        facts = extract_from_text(text, source="user_statement")
        assert any("C[user.pref]" in f.content for f in facts)

    def test_ignores_conversational_dont(self):
        text = "I don't think that's right"
        facts = extract_from_text(text, source="user_statement")
        pref_facts = [f for f in facts if "C[user.pref]" in f.content]
        assert len(pref_facts) == 0

    def test_extracts_url_from_user_message(self):
        text = "check out https://docs.example.com/api for reference"
        facts = extract_from_text(text, source="user_statement")
        assert any("V[url]" in f.content and "docs.example.com" in f.content for f in facts)


class TestExtractFromTextEdgeCases:
    def test_empty_text_returns_empty(self):
        assert extract_from_text("", source="tool_result") == []

    def test_no_patterns_returns_empty(self):
        assert extract_from_text("Everything is fine.", source="tool_result") == []

    def test_confidence_is_set(self):
        text = "Error: connection failed to https://api.example.com"
        facts = extract_from_text(text, source="tool_result")
        for f in facts:
            assert 0.0 <= f.confidence <= 1.0
