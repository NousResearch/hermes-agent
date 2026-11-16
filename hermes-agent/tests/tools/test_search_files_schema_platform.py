"""search_files advertises macOS protected-folder pruning only on the host that prunes."""

import pytest

from tools.file_tools import SEARCH_FILES_SCHEMA


@pytest.mark.platforms("macos")
def test_macos_schema_explains_protected_folder_pruning():
    assert "TCC-protected folders" in SEARCH_FILES_SCHEMA["description"]


@pytest.mark.platforms("not macos")
def test_other_hosts_schema_omits_macos_pruning():
    description = SEARCH_FILES_SCHEMA["description"]
    assert "macOS" not in description and "TCC" not in description
