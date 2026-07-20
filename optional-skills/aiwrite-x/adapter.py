"""
AIWriteX 适配器

封装 AIWriteX 核心功能，供 Hermes Agent 调用。
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# AIWriteX 项目路径
AIWRITE_X_PATH = os.environ.get(
    "AIWRITE_X_PATH",
    str(Path.home() / "Downloads" / "workspace" / "AIWriteX")
)

# 延迟导入标志
_aiwrite_initialized = False


def _ensure_aiwrite_path():
    """确保 AIWriteX 路径在 sys.path 中"""
    global _aiwrite_initialized
    if _aiwrite_initialized:
        return
    
    aiwrite_path = str(Path(AIWRITE_X_PATH).resolve())
    if aiwrite_path not in sys.path:
        sys.path.insert(0, aiwrite_path)
    
    # 激活 AIWriteX 的虚拟环境
    venv_site = Path(AIWRITE_X_PATH) / ".venv" / "lib"
    if venv_site.exists():
        for py_dir in venv_site.iterdir():
            site_packages = py_dir / "site-packages"
            if site_packages.exists() and str(site_packages) not in sys.path:
                sys.path.insert(0, str(site_packages))
    
    _aiwrite_initialized = True
    logger.info(f"AIWriteX adapter initialized, path: {aiwrite_path}")


def _import_aiwrite_modules():
    """延迟导入 AIWriteX 模块"""
    _ensure_aiwrite_path()
    
    from src.ai_write_x.core.unified_workflow import UnifiedContentWorkflow
    from src.ai_write_x.adapters.platform_adapters import PlatformType
    
    return UnifiedContentWorkflow, PlatformType


def generate_article(
    topic: str,
    platform: str = "wechat",
    urls: Optional[List[str]] = None,
    reference_ratio: float = 0.0,
    min_len: int = 1000,
    max_len: int = 2000,
    use_template: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    生成文章
    
    Args:
        topic: 文章主题
        platform: 目标平台 (wechat/xiaohongshu/baijiahao/zhihu/douban)
        urls: 参考文章 URL 列表
        reference_ratio: 参考比例 (0.0-1.0)
        min_len: 最小字数
        max_len: 最大字数
        use_template: 是否使用模板
        **kwargs: 其他参数
    
    Returns:
        Dict[str, Any]: 生成结果
        {
            "success": bool,
            "title": str,
            "content": str,
            "save_path": str,
            "publish_result": Dict,
            "message": str
        }
    """
    try:
        UnifiedContentWorkflow, PlatformType = _import_aiwrite_modules()
    except ImportError as e:
        return {
            "success": False,
            "title": "",
            "content": "",
            "save_path": "",
            "publish_result": None,
            "message": f"AIWriteX 模块导入失败: {e}"
        }
    
    # 验证平台
    try:
        platform_enum = PlatformType(platform)
    except ValueError:
        valid_platforms = [p.value for p in PlatformType]
        return {
            "success": False,
            "title": "",
            "content": "",
            "save_path": "",
            "publish_result": None,
            "message": f"不支持的平台: {platform}，有效平台: {valid_platforms}"
        }
    
    # 创建工作流
    workflow = UnifiedContentWorkflow()
    
    # 构建参数
    workflow_kwargs = {
        "publish_platform": platform,
        "urls": urls or [],
        "reference_ratio": reference_ratio,
        "min_len": min_len,
        "max_len": max_len,
        "use_template": use_template,
        **kwargs
    }
    
    try:
        logger.info(f"Starting AIWriteX article generation: topic={topic}, platform={platform}")
        
        # 执行工作流
        result = workflow.execute(topic, **workflow_kwargs)
        
        # 提取结果
        title = result.get("title", topic)
        content = result.get("formatted_content", "")
        save_result = result.get("save_result", {})
        publish_result = result.get("publish_result")
        
        return {
            "success": True,
            "title": title,
            "content": content,
            "save_path": save_result.get("path", ""),
            "publish_result": publish_result,
            "message": "文章生成成功"
        }
        
    except Exception as e:
        logger.error(f"AIWriteX article generation failed: {e}")
        return {
            "success": False,
            "title": "",
            "content": "",
            "save_path": "",
            "publish_result": None,
            "message": f"文章生成失败: {e}"
        }


def get_supported_platforms() -> List[str]:
    """获取支持的平台列表"""
    try:
        _, PlatformType = _import_aiwrite_modules()
        return [p.value for p in PlatformType]
    except ImportError as e:
        logger.error(f"Failed to get supported platforms: {e}")
        return ["wechat", "xiaohongshu", "baijiahao", "zhihu", "douban"]


def check_dependencies() -> Dict[str, Any]:
    """检查 AIWriteX 依赖是否满足"""
    result = {
        "ok": True,
        "issues": []
    }
    
    try:
        _import_aiwrite_modules()
    except ImportError as e:
        result["ok"] = False
        result["issues"].append(f"AIWriteX 模块导入失败: {e}")
    
    return result
