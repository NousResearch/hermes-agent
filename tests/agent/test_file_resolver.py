import os
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent.file_resolver import FileResolver

@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "asset.png"
    f.write_text("fake binary struct")
    return f

def test_file_resolver_mirror(tmp_file):
    mock_s3 = MagicMock()
    resolver = FileResolver(s3_client=mock_s3, bucket_name="test-bucket")
    
    assert resolver.mirror(str(tmp_file), "remote/asset.png") is True
    mock_s3.upload_file.assert_called_once_with(str(tmp_file), "test-bucket", "remote/asset.png")

def test_file_resolver_mirror_fail(tmp_file):
    resolver = FileResolver() # No s3 client
    assert resolver.mirror(str(tmp_file), "remote/asset.png") is False

def test_file_resolver_redirect(tmp_file):
    mock_s3 = MagicMock()
    resolver = FileResolver(s3_client=mock_s3, bucket_name="test-bucket")
    
    res = resolver.redirect(str(tmp_file), "remote/asset.png")
    assert res is True
    assert not tmp_file.exists()
    
    redir_file = tmp_file.with_suffix(".redirect")
    assert redir_file.exists()
    
    data = yaml.safe_load(redir_file.read_text())
    assert data["target"] == "s3://test-bucket/remote/asset.png"
    assert data["original_name"] == "asset.png"
    assert "hash" in data

def test_file_resolver_resolve_local(tmp_file):
    resolver = FileResolver()
    path = resolver.resolve(str(tmp_file))
    assert path == str(tmp_file)

def test_file_resolver_resolve_remote(tmp_path):
    redir_file = tmp_path / "asset.redirect"
    breadcrumb = {
        "target": "s3://test-bucket/remote/asset.png"
    }
    redir_file.write_text(yaml.dump(breadcrumb))
    
    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://s3.aws/pre-signed"
    resolver = FileResolver(s3_client=mock_s3, bucket_name="test-bucket")
    
    original_target = tmp_path / "asset.png"
    url = resolver.resolve(str(original_target))
    assert url == "https://s3.aws/pre-signed"
    mock_s3.generate_presigned_url.assert_called_once()
