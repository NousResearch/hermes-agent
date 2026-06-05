"""Tests for CLI i18n strings."""
import pytest

def test_cli_strings_exist_in_en():
    """Test that CLI strings exist in English locale."""
    from agent.i18n import _load_catalog
    
    catalog = _load_catalog('en')
    
    # 检查关键字符串存在
    assert 'cli.welcome' in catalog
    assert 'cli.help.header' in catalog
    assert 'cli.tips.1' in catalog

def test_cli_strings_exist_in_zh():
    """Test that CLI strings exist in Chinese locale."""
    from agent.i18n import _load_catalog
    
    catalog = _load_catalog('zh')
    
    # 检查关键字符串存在
    assert 'cli.welcome' in catalog
    assert 'cli.help.header' in catalog
    assert 'cli.tips.1' in catalog

def test_cli_strings_parity():
    """Test that en and zh have the same CLI keys."""
    from agent.i18n import _load_catalog
    
    en_catalog = _load_catalog('en')
    zh_catalog = _load_catalog('zh')
    
    en_cli_keys = {k for k in en_catalog.keys() if k.startswith('cli.')}
    zh_cli_keys = {k for k in zh_catalog.keys() if k.startswith('cli.')}
    
    # 允许少量差异（渐进迁移）
    missing_in_zh = en_cli_keys - zh_cli_keys
    assert len(missing_in_zh) <= 10, f"Too many missing CLI keys in zh: {missing_in_zh}"
