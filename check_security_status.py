#!/usr/bin/env python3
"""
Hermes 版本安全标记
显示当前版本的安全状态和官方代码库警告
"""

import sys
import os

# 添加 hermes 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from hermes_cli import __version__, __release_date__
    
    print("=" * 60)
    print("  🛡️  Hermes Agent - 安全版本标记")
    print("=" * 60)
    print()
    print(f"📦 当前版本: {__version__}")
    print(f"📅 发布日期: {__release_date__}")
    print()
    
    # 检查是否为自定义安全版本
    if "custom" in __version__:
        print("✅ 安全状态: 本地稳定版 (推荐)")
        print("   - 基于经过验证的稳定代码库")
        print("   - 不包含官方未测试修改")
        print("   - 已应用必要的安全补丁")
        print()
        print("⚠️  官方代码库警告:")
        print("   - 官方 main 分支包含大量结构性修改")
        print("   - 可能导致凭证管理、稳定性问题")
        print("   - 建议等待官方下一个稳定版 (v0.19.0)")
        print()
        print("📚 更多信息:")
        print("   - 安全审计报告: SECURITY_AUDIT_REPORT.md")
        print("   - 用户警告: WARNING_TO_USERS.md")
        print("   - 自定义版本信息: CUSTOM_VERSION.md")
    else:
        print("⚠️  安全状态: 官方版本")
        print("   建议检查是否为最新稳定版")
    
    print()
    print("=" * 60)
    
except ImportError as e:
    print(f"❌ 错误: 无法导入 Hermes 模块 ({e})")
    sys.exit(1)
