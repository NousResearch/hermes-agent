#!/usr/bin/env python3
"""
NarratoAI 集成测试脚本

测试 NarratoAI 适配器的核心功能：
1. 依赖检查
2. 语音列表获取
3. 模块导入
"""

import sys
import json
from pathlib import Path

# 添加 NarratoAI 路径
NARRATO_AI_PATH = Path.home() / "Downloads/workspace/NarratoAI"
sys.path.insert(0, str(NARRATO_AI_PATH))

# 添加 NarratoAI 虚拟环境的 site-packages
VENV_SITE_PACKAGES = NARRATO_AI_PATH / ".venv/lib/python3.12/site-packages"
if VENV_SITE_PACKAGES.exists():
    sys.path.insert(0, str(VENV_SITE_PACKAGES))

# 添加适配器路径
ADAPTER_PATH = Path.home() / "Downloads/workspace/hermes-agent/optional-skills/narrato-ai"
sys.path.insert(0, str(ADAPTER_PATH))


def test_dependencies():
    """测试依赖检查"""
    print("=" * 60)
    print("测试 1: 依赖检查")
    print("=" * 60)
    
    try:
        import adapter
        result = adapter.check_dependencies()
        
        print(f"✓ 依赖检查结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        if result.get("ok"):
            print("✓ 所有依赖满足")
            return True
        else:
            print(f"✗ 依赖问题: {result.get('issues', [])}")
            return False
            
    except Exception as e:
        print(f"✗ 依赖检查失败: {e}")
        return False


def test_voice_list():
    """测试语音列表获取"""
    print("\n" + "=" * 60)
    print("测试 2: 语音列表获取")
    print("=" * 60)
    
    try:
        import adapter
        voices = adapter.get_available_voices()
        
        print(f"✓ 获取到 {len(voices)} 个可用语音")
        print(f"  前5个语音: {voices[:5]}")
        
        if len(voices) > 0:
            print("✓ 语音列表获取成功")
            return True
        else:
            print("✗ 语音列表为空")
            return False
            
    except Exception as e:
        print(f"✗ 语音列表获取失败: {e}")
        return False


def test_module_import():
    """测试模块导入"""
    print("\n" + "=" * 60)
    print("测试 3: 模块导入")
    print("=" * 60)
    
    try:
        import adapter
        
        # 检查核心函数是否存在
        required_functions = [
            "check_dependencies",
            "get_available_voices",
            "generate_narration_video",
            "generate_tts_audio",
        ]
        
        missing = []
        for func_name in required_functions:
            if hasattr(adapter, func_name):
                print(f"✓ 函数 {func_name} 存在")
            else:
                print(f"✗ 函数 {func_name} 缺失")
                missing.append(func_name)
        
        if not missing:
            print("✓ 所有核心函数存在")
            return True
        else:
            print(f"✗ 缺失函数: {missing}")
            return False
            
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("NarratoAI 集成测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("依赖检查", test_dependencies()))
    results.append(("语音列表", test_voice_list()))
    results.append(("模块导入", test_module_import()))
    
    # 输出测试报告
    print("\n" + "=" * 60)
    print("测试报告")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✓ 所有测试通过！NarratoAI 集成成功")
        return 0
    else:
        print(f"\n✗ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
