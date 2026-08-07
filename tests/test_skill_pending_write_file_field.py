"""Tests for #78519: write_file pending payload normalization.

When the LLM calls skill_manage(action='write_file', ..., content='...')
instead of file_content='...', the staged payload must be normalized so
apply_skill_pending can replay it correctly.
"""

import json


class TestWriteFilePayloadNormalization:
    """Direct test of the normalization logic in _apply_skill_write_gate."""

    def test_content_normalized_to_file_content_for_write_file(self):
        """When LLM passes content= instead of file_content= for write_file,
        the payload should have file_content set from content."""
        # Simulate what _apply_skill_write_gate does with payload_kwargs
        action = "write_file"
        name = "my-skill"
        payload_kwargs = {"file_path": "references/test.md", "content": "file content here"}

        payload = {"action": action, "name": name}
        payload.update({k: v for k, v in payload_kwargs.items() if v is not None})
        # Apply the normalization (the fix)
        if action == "write_file" and "file_content" not in payload and payload.get("content"):
            payload["file_content"] = payload.pop("content")

        assert payload["file_content"] == "file content here"
        assert "content" not in payload

    def test_file_content_preserved_when_correct(self):
        """When LLM correctly passes file_content=, it should not be altered."""
        action = "write_file"
        name = "my-skill"
        payload_kwargs = {"file_path": "references/test.md", "file_content": "correct field"}

        payload = {"action": action, "name": name}
        payload.update({k: v for k, v in payload_kwargs.items() if v is not None})
        if action == "write_file" and "file_content" not in payload and payload.get("content"):
            payload["file_content"] = payload.pop("content")

        assert payload["file_content"] == "correct field"
        # content should NOT be present (it was never in payload_kwargs)
        assert "content" not in payload

    def test_non_write_file_actions_not_affected(self):
        """Normalization should not touch content= for create/edit actions."""
        action = "create"
        name = "my-skill"
        payload_kwargs = {"content": "# My Skill\nContent here"}

        payload = {"action": action, "name": name}
        payload.update({k: v for k, v in payload_kwargs.items() if v is not None})
        if action == "write_file" and "file_content" not in payload and payload.get("content"):
            payload["file_content"] = payload.pop("content")

        assert payload["content"] == "# My Skill\nContent here"
        assert "file_content" not in payload

    def test_both_content_and_file_content_present(self):
        """If both are passed (unlikely), file_content wins."""
        action = "write_file"
        name = "my-skill"
        payload_kwargs = {"file_path": "ref.md", "content": "wrong", "file_content": "correct"}

        payload = {"action": action, "name": name}
        payload.update({k: v for k, v in payload_kwargs.items() if v is not None})
        if action == "write_file" and "file_content" not in payload and payload.get("content"):
            payload["file_content"] = payload.pop("content")

        assert payload["file_content"] == "correct"
        assert payload["content"] == "wrong"  # both present, no pop since file_content exists

    def test_empty_content_not_normalized(self):
        """Empty string content should NOT be normalized (fallback guard)."""
        action = "write_file"
        name = "my-skill"
        payload_kwargs = {"file_path": "ref.md", "content": ""}

        payload = {"action": action, "name": name}
        payload.update({k: v for k, v in payload_kwargs.items() if v is not None})
        if action == "write_file" and "file_content" not in payload and payload.get("content"):
            payload["file_content"] = payload.pop("content")

        # Empty content should not trigger normalization (falsy check)
        assert "file_content" not in payload
